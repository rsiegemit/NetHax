"""Interactive human play mode for MiniHax environments using Pygame."""

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

from Nethax.nethax_env import make_minihax_env_from_name
from Nethax.tiles.renderer import load_tiles
from Nethax.minihax.pixel_renderer import (
    render_pixels_with_monsters,
    render_pixels_no_monsters,
)
from Nethax.minihax.constants import Action, TILE_SIZE


# ============================================================================
# Environment tier detection
# ============================================================================

# Maps env name substrings to tiers (order matters: specific patterns first)
TIER_PATTERNS = [
    # Tier 4 Sokoban
    ("Soko", 4),
    # Tier 3 Combat (before Tier 2 since "Quest" appears in both)
    ("ZombieHorde", 3),
    ("QuestHard", 3),
    ("KeyAndDoor", 3),
    ("ClosedDoor", 3),
    ("Chest", 3),
    ("Memento", 3),
    # Tier 2 Hazards
    ("LavaCrossing", 2),
    ("HideNSeek", 2),
    ("QuestEasy", 2),
    ("QuestMedium", 2),
    ("LockedDoor", 2),
    ("TreasureDash", 2),
    # Tier 1 Navigation
    ("Corridor", 1),
    ("Mazewalk", 1),
    ("ExploreMaze", 1),
]


def get_tier(env_name):
    """Determine environment tier from name."""
    for pattern, tier in TIER_PATTERNS:
        if pattern in env_name:
            return tier
    # Plain "Quest" without Easy/Medium/Hard suffix is Tier 3
    if "Quest" in env_name:
        return 3
    return 1


def resolve_env_name(short_name):
    """Resolve short env name to full factory name. Returns (full_name, env)."""
    for pattern in [f"Minihax-{short_name}-v0", f"Minihax-{short_name}-Symbolic-v0"]:
        try:
            env = make_minihax_env_from_name(pattern)
            return pattern, env
        except ValueError:
            continue
    raise ValueError(
        f"Unknown env: {short_name}\n"
        f"Examples: ZombieHorde, Corridor5, Soko1a, LavaCrossing, Quest, MementoEasy"
    )


def get_static_params(env):
    """Get static params from env, handling attribute name differences."""
    if hasattr(env, 'static_params'):
        return env.static_params
    elif hasattr(env, 'static_env_params'):
        return env.static_env_params
    raise RuntimeError("Cannot find static params on env")


# ============================================================================
# Key mapping (NetHack vi-style)
# ============================================================================

def build_key_maps():
    """Build key->action mappings. Returns (key_map, unicode_map)."""
    if pygame is None:
        return {}, {}

    key_map = {
        # Cardinal movement (arrow keys)
        pygame.K_UP: Action.MOVE_N,
        pygame.K_DOWN: Action.MOVE_S,
        pygame.K_RIGHT: Action.MOVE_E,
        pygame.K_LEFT: Action.MOVE_W,
        # Vi-style cardinal (hjkl)
        pygame.K_h: Action.MOVE_W,
        pygame.K_j: Action.MOVE_S,
        pygame.K_k: Action.MOVE_N,
        pygame.K_l: Action.MOVE_E,
        # Vi-style diagonal (yubn)
        pygame.K_y: Action.MOVE_NW,
        pygame.K_u: Action.MOVE_NE,
        pygame.K_b: Action.MOVE_SW,
        pygame.K_n: Action.MOVE_SE,
        # Actions
        pygame.K_s: Action.SEARCH,
        pygame.K_e: Action.EAT,
        pygame.K_COMMA: Action.PICKUP,
        pygame.K_a: Action.USE_ITEM,
        pygame.K_f: Action.KICK,
        pygame.K_o: Action.OPEN_DOOR,
        pygame.K_z: Action.ZAP,
        pygame.K_1: Action.SLOT_0,
        pygame.K_2: Action.SLOT_1,
        pygame.K_3: Action.SLOT_2,
    }

    # Unicode map for shift-modified keys (checked first)
    unicode_map = {
        '>': Action.GO_DOWN_STAIRS,
        'U': Action.UNLOCK_DOOR,
    }

    return key_map, unicode_map


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

def draw_hud(screen, font, env_name, tier, state, last_reward, cumulative_reward,
             last_action_name, y_offset):
    """Draw HUD info bar at bottom of screen."""
    # Line 1: env info and action
    timestep = int(state.timestep)
    line1 = (f"{env_name}  T:{timestep}  Action:{last_action_name}  "
             f"R:{last_reward:+.2f}  Total:{cumulative_reward:.2f}")
    surf1 = font.render(line1, True, (200, 200, 200))
    screen.blit(surf1, (10, y_offset))

    # Line 2: tier-specific stats
    if tier in (2, 3):
        hp = int(state.player_stats.hp)
        max_hp = int(state.player_stats.max_hp)
        xlev = int(state.player_stats.xp_level)
        ac = int(state.player_stats.ac)
        score = int(state.player_stats.score)
        killed = int(state.player_stats.monsters_killed)
        hp_ratio = hp / max(max_hp, 1)
        if hp_ratio > 0.5:
            hp_color = (0, 255, 0)
        elif hp_ratio > 0.25:
            hp_color = (255, 255, 0)
        else:
            hp_color = (255, 0, 0)
        # Zap phase indicator (Tier 3 only)
        zap_str = ""
        if tier == 3 and hasattr(state, 'zap_phase'):
            zp = int(state.zap_phase)
            if zp == 1:
                zap_str = "  [ZAP: select slot 1/2/3]"
            elif zp == 2:
                zap_str = "  [ZAP: select direction]"
        line2 = f"HP:{hp}/{max_hp}  Lv:{xlev}  AC:{ac}  Score:{score}  Kills:{killed}{zap_str}"
        surf2 = font.render(line2, True, hp_color)
        screen.blit(surf2, (10, y_offset + 18))

        # Line 3: Inventory
        if hasattr(state, 'inventory'):
            from Nethax.minihax.constants import ItemType
            inv_parts = []
            for i in range(state.inventory.item_mask.shape[0]):
                if bool(state.inventory.item_mask[i]):
                    item_id = int(state.inventory.item_ids[i])
                    name = ItemType(item_id).name.replace('_', ' ').title()
                    inv_parts.append(f"{i+1}:{name}")
                else:
                    inv_parts.append(f"{i+1}:---")
            inv_str = "Inv: " + "  ".join(inv_parts)
            lev_str = ""
            if hasattr(state, 'player_levitating') and bool(state.player_levitating):
                lev_str = f"  [LEV:{int(state.levitation_turns)}t]"
            key_str = ""
            if hasattr(state, 'player_has_key') and bool(state.player_has_key):
                key_str = "  [KEY]"
            surf3 = font.render(inv_str + lev_str + key_str, True, (180, 180, 255))
            screen.blit(surf3, (10, y_offset + 36))
    elif tier == 4:
        pits = int(state.pits_remaining)
        line2 = f"Pits remaining: {pits}"
        surf2 = font.render(line2, True, (200, 200, 200))
        screen.blit(surf2, (10, y_offset + 18))


# ============================================================================
# Main game loop
# ============================================================================

def main():
    if pygame is None:
        print("Pygame is required. Install with: pip install pygame")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Play MiniHax interactively")
    parser.add_argument("--env", type=str, required=True,
                        help="Environment short name (e.g. ZombieHorde, Corridor5, Soko1a)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: random)")
    parser.add_argument("--fps", type=int, default=60, help="Target FPS")
    parser.add_argument("--scale", type=int, default=2, help="Pixel upscale factor")
    parser.add_argument("--save", action="store_true", help="Save trajectory on exit")
    parser.add_argument("--debug", action="store_true", help="Disable JIT compilation")
    parser.add_argument("--wizard", action="store_true", help="Show hidden traps (Tier 3)")
    parser.add_argument("--max-steps", type=int, default=10000,
                        help="Max timesteps per episode")
    args = parser.parse_args()

    if args.debug:
        jax.config.update("jax_disable_jit", True)

    # Resolve env and detect tier
    full_name, env = resolve_env_name(args.env)
    tier = get_tier(args.env)
    static_params = get_static_params(env)

    # Load tiles for pixel rendering
    print("Loading tiles...")
    tiles_array = load_tiles()

    # Build JIT-compiled render function (includes upscaling for zero-copy display)
    scale = args.scale
    if tier in (1, 4):
        @jax.jit
        def render_fn(state):
            pixels = render_pixels_no_monsters(state, static_params, tiles_array)
            return jnp.repeat(jnp.repeat(pixels, scale, axis=0), scale, axis=1)
    else:
        max_monsters = static_params.max_monsters
        if args.wizard and tier == 3:
            @jax.jit
            def render_fn(state):
                pixels = render_pixels_with_monsters(
                    state, static_params, tiles_array, max_monsters,
                    wizard_mode=True, traps=state.traps)
                return jnp.repeat(jnp.repeat(pixels, scale, axis=0), scale, axis=1)
        else:
            @jax.jit
            def render_fn(state):
                pixels = render_pixels_with_monsters(
                    state, static_params, tiles_array, max_monsters)
                return jnp.repeat(jnp.repeat(pixels, scale, axis=0), scale, axis=1)

    # Env params with higher timestep limit for human play
    env_params = env.default_params.replace(max_timesteps=args.max_steps)

    # Setup seed
    seed = args.seed if args.seed is not None else int(np.random.randint(2**31))
    rng = jax.random.PRNGKey(seed)

    # Reset env
    print("Compiling environment (first reset)...")
    rng, _rng = jax.random.split(rng)
    obs, state = env.reset(_rng, env_params)

    # Warmup: compile render + step before game loop starts
    print("Compiling renderer (first render)...")
    pixels = render_fn(state)
    pixels.block_until_ready()

    print("Compiling step function (first step)...")
    _warmup = env.step(jax.random.PRNGKey(0), state, jnp.int32(0), env_params)
    _warmup[0].block_until_ready()
    del _warmup

    print(f"\nPlaying: {full_name} (Tier {tier})")
    print(f"Seed: {seed}")
    print(f"Actions available: {env.num_actions}")
    print(f"Map size: {static_params.map_height}x{static_params.map_width}")
    print()
    print("Controls:")
    print("  Movement:  arrow keys or hjkl (cardinal), yubn (diagonal)")
    print("  s=search  e=eat  >=stairs  ,=pickup  a=use item")
    print("  z=zap wand  1/2/3=select slot  (then direction to fire)")
    print("  f=kick  o=open door  U(shift+u)=unlock door")
    print("  ESC=quit")
    print()

    # Pygame setup
    pygame.init()
    scale = args.scale
    pixel_h = static_params.map_height * TILE_SIZE * scale
    pixel_w = static_params.map_width * TILE_SIZE * scale
    hud_height = 70
    screen = pygame.display.set_mode((pixel_w, pixel_h + hud_height))
    pygame.display.set_caption(f"MiniHax - {args.env}")

    font = pygame.font.SysFont("monospace", 14)
    big_font = pygame.font.SysFont("monospace", 32, bold=True)
    clock = pygame.time.Clock()
    key_map, unicode_map = build_key_maps()

    # Trajectory recording
    history = {"action": [], "reward": [], "done": []}

    running = True
    game_over = False
    last_reward = 0.0
    cumulative_reward = 0.0
    last_action_name = "---"

    while running:
        # Render pixels to screen (upscaling already done in JIT render_fn)
        pixels_np = np.asarray(pixels)
        # Transpose [H, W, 3] -> [W, H, 3] for pygame
        surface = pygame.surfarray.make_surface(pixels_np.transpose(1, 0, 2))

        screen.fill((0, 0, 0))
        screen.blit(surface, (0, 0))

        # Draw HUD
        draw_hud(screen, font, args.env, tier, state, last_reward, cumulative_reward,
                 last_action_name, pixel_h + 5)

        # Game over overlay
        if game_over:
            overlay = pygame.Surface((pixel_w, pixel_h))
            overlay.set_alpha(128)
            overlay.fill((0, 0, 0))
            screen.blit(overlay, (0, 0))

            msg = "EPISODE OVER"
            color = (0, 255, 0) if cumulative_reward > 0 else (255, 100, 100)
            text_surf = big_font.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(pixel_w // 2, pixel_h // 2 - 20))
            screen.blit(text_surf, text_rect)

            sub_text = font.render(
                f"Reward: {cumulative_reward:.2f}  |  SPACE=restart  ESC=quit",
                True, (200, 200, 200))
            sub_rect = sub_text.get_rect(center=(pixel_w // 2, pixel_h // 2 + 20))
            screen.blit(sub_text, sub_rect)

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
                        # Restart episode
                        rng, _rng = jax.random.split(rng)
                        obs, state = env.reset(_rng, env_params)
                        pixels = render_fn(state)
                        game_over = False
                        last_reward = 0.0
                        cumulative_reward = 0.0
                        last_action_name = "---"
                else:
                    # Check unicode map first (shift-modified keys)
                    act = unicode_map.get(event.unicode)
                    if act is None:
                        act = key_map.get(event.key)
                    if act is not None:
                        action = act

        # Step env if we have a valid action
        if action is not None and not game_over:
            action_id = int(action)
            if action_id < env.num_actions:
                last_action_name = Action(action_id).name
                rng, _rng = jax.random.split(rng)
                obs, state, reward, done, info = env.step(
                    _rng, state, jnp.int32(action_id), env_params)

                last_reward = float(reward)
                cumulative_reward += last_reward

                if args.save:
                    history["action"].append(action_id)
                    history["reward"].append(float(reward))
                    history["done"].append(bool(done))

                # Re-render
                pixels = render_fn(state)

                if done:
                    game_over = True

        clock.tick(args.fps)

    pygame.quit()

    # Save trajectory
    if args.save and len(history["action"]) > 0:
        save_dir = Path("play_data")
        save_dir.mkdir(exist_ok=True)
        timestamp = int(time.time())
        filename = save_dir / f"minihax_{args.env}_{timestamp}.pbz2"
        save_compressed_pickle(history, filename)
        print(f"Trajectory saved to {filename} ({len(history['action'])} steps)")


if __name__ == "__main__":
    main()
