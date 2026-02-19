"""Interactive human play mode for Nethax using Pygame."""

import bz2
import pickle
import time
import argparse
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

try:
    import pygame
except ImportError:
    pygame = None

# The nethax subpackage internally uses lowercase 'nethax.' imports.
# Register Nethax as 'nethax' in sys.modules so transitive imports resolve.
import sys
import Nethax
import Nethax.nethax
sys.modules.setdefault('nethax', Nethax)
sys.modules.setdefault('nethax.nethax', Nethax.nethax)

from Nethax.nethax.nethax_state import EnvParams, StaticEnvParams
from Nethax.nethax.constants import Action
from Nethax.nethax.game_logic import nethax_step, is_game_over
from Nethax.nethax.renderer import render_nethax_text
from Nethax.nethax.world_gen.world_gen import generate_world


# ============================================================================
# Key mappings (NetHack-style)
# ============================================================================

def build_key_maps():
    """Build key->action mappings. Returns (key_map, unicode_map)."""
    if pygame is None:
        return {}, {}

    # Primary key map (event.key based)
    key_map = {
        # Cardinal movement (arrow keys and hjkl)
        pygame.K_UP: Action.MOVE_N,
        pygame.K_DOWN: Action.MOVE_S,
        pygame.K_RIGHT: Action.MOVE_E,
        pygame.K_LEFT: Action.MOVE_W,
        pygame.K_k: Action.MOVE_N,
        pygame.K_j: Action.MOVE_S,
        pygame.K_l: Action.MOVE_E,
        pygame.K_h: Action.MOVE_W,
        # Diagonal movement (yubn)
        pygame.K_y: Action.MOVE_NW,
        pygame.K_u: Action.MOVE_NE,
        pygame.K_b: Action.MOVE_SW,
        pygame.K_n: Action.MOVE_SE,
        # Actions
        pygame.K_COMMA: Action.PICKUP,       # ,
        pygame.K_d: Action.DROP,
        pygame.K_e: Action.EAT,
        pygame.K_q: Action.QUAFF,
        pygame.K_r: Action.READ,
        pygame.K_z: Action.ZAP,
        pygame.K_w: Action.WIELD,
        pygame.K_PERIOD: Action.WAIT,         # .
        pygame.K_o: Action.OPEN_DOOR,
        pygame.K_c: Action.CLOSE_DOOR,
        pygame.K_s: Action.SEARCH,
        pygame.K_t: Action.THROW,
        pygame.K_p: Action.PRAY,
        pygame.K_a: Action.APPLY,
        pygame.K_f: Action.KICK,
    }

    # Unicode map for shift-modified keys (checked first)
    unicode_map = {
        '<': Action.GO_UP,
        '>': Action.GO_DOWN,
        'W': Action.WEAR,
        'R': Action.REMOVE,
    }

    return key_map, unicode_map


# Character sets for colored text rendering
MONSTER_CHARS = frozenset(':dkrxFbGhaBfecdnoETOLHVJ&DEW')
ITEM_CHARS = frozenset(')[$!?/="*')


# ============================================================================
# Trajectory saving
# ============================================================================

def save_compressed_pickle(data, path):
    """Save data as bz2-compressed pickle."""
    with bz2.BZ2File(str(path), 'wb') as f:
        pickle.dump(data, f)


# ============================================================================
# HUD rendering
# ============================================================================

def get_hunger_text(nutrition):
    """Get hunger state text from nutrition value."""
    if nutrition >= 2000:
        return "Satiated", (0, 200, 0)
    elif nutrition >= 1000:
        return "", (200, 200, 200)  # Normal, don't show
    elif nutrition >= 150:
        return "Hungry", (255, 255, 0)
    elif nutrition >= 50:
        return "Weak", (255, 165, 0)
    elif nutrition >= 0:
        return "Fainting", (255, 0, 0)
    else:
        return "Starving", (255, 0, 0)


def draw_hud(screen, font, state, static_params, last_action_name, y_offset):
    """Draw HUD stats bar at bottom of screen."""
    hp = int(state.player_hp)
    max_hp = int(state.player_max_hp)
    dlvl = int(state.player_level) + 1
    xlev = int(state.player_xp_level)
    ac = int(state.player_ac)
    strength = int(state.player_strength)
    gold = int(state.gold)
    score = int(state.score)
    turns = int(state.timestep)
    nutrition = int(state.player_nutrition)
    killed = int(state.monsters_killed)

    # HP color
    hp_ratio = hp / max(max_hp, 1)
    if hp_ratio > 0.5:
        hp_color = (0, 255, 0)
    elif hp_ratio > 0.25:
        hp_color = (255, 255, 0)
    else:
        hp_color = (255, 0, 0)

    hunger_text, hunger_color = get_hunger_text(nutrition)

    # Line 1: vitals
    x = 10
    parts = [
        (f"HP:{hp}/{max_hp}", hp_color),
        (f"  Dlvl:{dlvl}", (200, 200, 200)),
        (f"  Lv:{xlev}", (200, 200, 200)),
        (f"  AC:{ac}", (200, 200, 200)),
        (f"  Str:{strength}", (200, 200, 200)),
        (f"  ${gold}", (255, 255, 0)),
    ]
    if hunger_text:
        parts.append((f"  {hunger_text}", hunger_color))

    for text, color in parts:
        surf = font.render(text, True, color)
        screen.blit(surf, (x, y_offset))
        x += surf.get_width()

    # Line 2: score, turns, action
    line2 = f"Score:{score}  Kills:{killed}  T:{turns}  Action:{last_action_name}"
    surf2 = font.render(line2, True, (150, 150, 150))
    screen.blit(surf2, (10, y_offset + 18))


# ============================================================================
# Main
# ============================================================================

def play(seed=0, god_mode=False, fps=30, save_trajectories=False, debug=False):
    """Launch interactive play session."""
    if pygame is None:
        print("Pygame is required. Install with: pip install pygame")
        return

    if debug:
        jax.config.update("jax_disable_jit", True)

    params = EnvParams(god_mode=god_mode)
    static_params = StaticEnvParams()

    # JIT-compile step and game-over check (params/static_params are static)
    step_fn = jax.jit(nethax_step, static_argnums=(3, 4))
    done_fn = jax.jit(is_game_over, static_argnums=(1, 2))

    rng = jax.random.PRNGKey(seed)
    rng, _rng = jax.random.split(rng)
    print("Generating world...")
    state = generate_world(_rng, params, static_params)

    # Warmup: compile step + done check before game loop
    print("Compiling step function (first step)...")
    _ws, _wr = step_fn(jax.random.PRNGKey(0), state, 0, params, static_params)
    _wr.block_until_ready()
    print("Compiling done check...")
    _wd = done_fn(state, params, static_params)
    jnp.array(_wd).block_until_ready()
    del _ws, _wr, _wd

    print(f"Playing Nethax (seed={seed}, god_mode={god_mode})")
    print()
    print("Controls:")
    print("  Movement:  arrow keys or hjkl (cardinal), yubn (diagonal)")
    print("  ,=pickup  d=drop  e=eat  q=quaff  r=read  z=zap")
    print("  w=wield  W(shift)=wear  R(shift)=remove  a=apply")
    print("  o=open  c=close  f=kick  s=search  t=throw  p=pray")
    print("  .=wait  <=go up  >=go down")
    print("  ESC=quit")
    print()

    # Pygame setup
    pygame.init()
    font_size = 16
    font = pygame.font.SysFont("monospace", font_size)
    hud_font = pygame.font.SysFont("monospace", 14)
    big_font = pygame.font.SysFont("monospace", 28, bold=True)

    map_h, map_w = static_params.map_size
    screen_w = map_w * (font_size // 2 + 2) + 40
    hud_height = 50
    screen_h = (map_h + 4) * font_size + 40 + hud_height
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Nethax")

    clock = pygame.time.Clock()
    key_map, unicode_map = build_key_maps()

    # Trajectory recording
    history = {"action": [], "reward": [], "done": []}

    running = True
    game_over = False
    last_action_name = "---"
    cumulative_reward = 0.0
    last_reward = 0.0
    text_area_height = (map_h + 4) * font_size + 40

    while running:
        # Render text map
        text_obs = render_nethax_text(state, static_params)
        screen.fill((0, 0, 0))

        for i, line in enumerate(text_obs.split("\n")):
            x_offset = 10
            for ch in line:
                if ch == "@":
                    surf = font.render(ch, True, (255, 255, 0))
                elif ch in MONSTER_CHARS:
                    surf = font.render(ch, True, (255, 0, 0))
                elif ch == "$":
                    surf = font.render(ch, True, (255, 255, 0))
                elif ch in ITEM_CHARS:
                    surf = font.render(ch, True, (100, 100, 255))
                elif ch in "<>":
                    surf = font.render(ch, True, (0, 255, 255))
                elif ch == "#":
                    surf = font.render(ch, True, (128, 128, 128))
                elif ch == ".":
                    surf = font.render(ch, True, (180, 180, 180))
                elif ch == "+":
                    surf = font.render(ch, True, (139, 69, 19))
                else:
                    surf = font.render(ch, True, (255, 255, 255))
                screen.blit(surf, (x_offset, 10 + i * font_size))
                x_offset += surf.get_width()

        # Draw HUD
        draw_hud(screen, hud_font, state, static_params, last_action_name, text_area_height)

        # Game over overlay
        if game_over:
            overlay = pygame.Surface((screen_w, text_area_height))
            overlay.set_alpha(100)
            overlay.fill((0, 0, 0))
            screen.blit(overlay, (0, 0))

            msg = f"GAME OVER  Score: {int(state.score)}"
            text_surf = big_font.render(msg, True, (255, 100, 100))
            text_rect = text_surf.get_rect(center=(screen_w // 2, text_area_height // 2 - 15))
            screen.blit(text_surf, text_rect)

            sub = f"Dlvl:{int(state.player_level)+1}  Turns:{int(state.timestep)}  |  SPACE=restart  ESC=quit"
            sub_surf = hud_font.render(sub, True, (200, 200, 200))
            sub_rect = sub_surf.get_rect(center=(screen_w // 2, text_area_height // 2 + 15))
            screen.blit(sub_surf, sub_rect)

        pygame.display.flip()

        # Handle input
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif game_over:
                    if event.key == pygame.K_SPACE:
                        # Restart with new seed
                        rng, _rng = jax.random.split(rng)
                        state = generate_world(_rng, params, static_params)
                        game_over = False
                        last_action_name = "---"
                        cumulative_reward = 0.0
                        last_reward = 0.0
                else:
                    # Check unicode map first (shift-modified keys)
                    act = unicode_map.get(event.unicode)
                    if act is None:
                        act = key_map.get(event.key)
                    if act is not None:
                        action = act

        # Step
        if action is not None and not game_over:
            last_action_name = Action(int(action)).name
            rng, _rng = jax.random.split(rng)
            state, reward = step_fn(_rng, state, action, params, static_params)
            last_reward = float(reward)
            cumulative_reward += last_reward

            if save_trajectories:
                history["action"].append(int(action))
                history["reward"].append(float(reward))
                history["done"].append(False)

            if done_fn(state, params, static_params):
                game_over = True
                if save_trajectories:
                    history["done"][-1] = True

        clock.tick(fps)

    pygame.quit()

    # Save trajectory
    if save_trajectories and len(history["action"]) > 0:
        save_dir = Path("play_data")
        save_dir.mkdir(exist_ok=True)
        timestamp = int(time.time())
        suffix = "_GM" if god_mode else ""
        filename = save_dir / f"nethax_{seed}{suffix}_{timestamp}.pbz2"
        save_compressed_pickle(history, filename)
        print(f"Trajectory saved to {filename} ({len(history['action'])} steps)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Nethax interactively")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--god-mode", action="store_true", help="Enable god mode")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    parser.add_argument("--save", action="store_true", help="Save trajectory on exit")
    parser.add_argument("--debug", action="store_true", help="Disable JIT compilation")
    args = parser.parse_args()

    play(seed=args.seed, god_mode=args.god_mode, fps=args.fps,
         save_trajectories=args.save, debug=args.debug)
