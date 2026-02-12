"""Interactive human play mode for Nethax using Pygame."""

import jax
import jax.numpy as jnp

try:
    import pygame
except ImportError:
    pygame = None

from nethax.nethax.nethax_state import EnvParams, StaticEnvParams
from nethax.nethax.constants import Action
from nethax.nethax.game_logic import nethax_step, is_game_over
from nethax.nethax.renderer import render_nethax_text
from nethax.nethax.world_gen.world_gen import generate_world


# Key mappings (NetHack-style: yuhjklbn for diagonal, arrow keys for cardinal)
KEY_ACTION_MAP = {}
if pygame is not None:
    KEY_ACTION_MAP = {
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
        pygame.K_LESS: Action.GO_UP,          # <
        pygame.K_GREATER: Action.GO_DOWN,     # >
        pygame.K_o: Action.OPEN_DOOR,
        pygame.K_c: Action.CLOSE_DOOR,
        pygame.K_s: Action.SEARCH,
        pygame.K_t: Action.THROW,
        pygame.K_p: Action.PRAY,
    }


def play(seed: int = 0, god_mode: bool = False):
    """Launch interactive play session."""
    if pygame is None:
        print("Pygame is required for interactive play. Install with: pip install pygame")
        return

    params = EnvParams(god_mode=god_mode)
    static_params = StaticEnvParams()

    rng = jax.random.PRNGKey(seed)
    rng, _rng = jax.random.split(rng)
    state = generate_world(_rng, params, static_params)

    # Pygame setup
    pygame.init()
    font_size = 16
    font = pygame.font.SysFont("monospace", font_size)

    map_h, map_w = static_params.map_size
    screen_w = map_w * (font_size // 2 + 2) + 40
    screen_h = (map_h + 4) * font_size + 40
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Nethax")

    clock = pygame.time.Clock()
    running = True

    while running:
        # Render
        text_obs = render_nethax_text(state, static_params)
        screen.fill((0, 0, 0))

        for i, line in enumerate(text_obs.split("\n")):
            color = (255, 255, 255)
            # Character sets for detection
            MONSTER_CHARS = set(':dkrxFbGhaBfecdnoETOLHVJ&DEW')
            ITEM_CHARS = set(')[$!?/="*')

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
                    surf = font.render(ch, True, color)
                screen.blit(surf, (x_offset, 10 + i * font_size))
                x_offset += surf.get_width()

        pygame.display.flip()

        # Check game over
        if is_game_over(state, params, static_params):
            print("Game Over!")
            print(f"Score: {int(state.score)}")
            print(f"Dungeon level: {int(state.player_level) + 1}")
            print(f"Turns: {int(state.timestep)}")
            pygame.time.wait(3000)
            running = False
            continue

        # Handle input
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in KEY_ACTION_MAP:
                    action = KEY_ACTION_MAP[event.key]

        if action is not None:
            rng, _rng = jax.random.split(rng)
            state, reward = nethax_step(_rng, state, action, params, static_params)

        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Play Nethax interactively")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--god-mode", action="store_true", help="Enable god mode")
    args = parser.parse_args()

    play(seed=args.seed, god_mode=args.god_mode)
