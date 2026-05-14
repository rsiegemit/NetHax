"""Pygame UI for the Nethax environment.

Launch with:
    python -m Nethax.ui.pygame_app

Controls:
    Movement:  arrow keys or hjkl (cardinal), yubn (diagonal)
    </>        stairs up/down
    .          wait
    s          search
    ,          pickup
    a          apply
    e          eat
    q          quaff
    r          read
    d          drop
    z          zap
    i          toggle inventory pane
    Q          quit
    ESC        quit
"""

from __future__ import annotations

import sys

try:
    import pygame
except ImportError:
    pygame = None

import numpy as np
import jax
import jax.numpy as jnp

from Nethax.nethax.env import NethaxEnv
from Nethax.nethax.constants.actions import ACTIONS, Action
from Nethax.tiles import render_pixels, GLYPH2TILE, TILE_SIZE
from Nethax.tiles.renderer import load_tiles


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

MAP_ROWS = 21
MAP_COLS = 79  # NLE glyphs width (env uses 79; tile pane uses 80 for display)
TILE_PANE_W = 80 * TILE_SIZE   # 1280 px
TILE_PANE_H = MAP_ROWS * TILE_SIZE  # 336 px
STATUS_PANEL_H = 120            # pixel height of the bottom status panel
INVENTORY_PANEL_W = 360         # pixel width of the right inventory overlay

WINDOW_W = TILE_PANE_W
WINDOW_H = TILE_PANE_H + STATUS_PANEL_H

FPS = 60

# ANSI colour index -> RGB (standard terminal palette)
ANSI_COLORS = [
    (0,   0,   0),    # 0  black
    (170, 0,   0),    # 1  red
    (0,   170, 0),    # 2  green
    (170, 85,  0),    # 3  brown/dark yellow
    (0,   0,   170),  # 4  blue
    (170, 0,   170),  # 5  magenta
    (0,   170, 170),  # 6  cyan
    (170, 170, 170),  # 7  light gray
    (85,  85,  85),   # 8  dark gray (bright black)
    (255, 85,  85),   # 9  orange / bright red
    (85,  255, 85),   # 10 bright green
    (255, 255, 85),   # 11 yellow
    (85,  85,  255),  # 12 bright blue
    (255, 85,  255),  # 13 bright magenta
    (85,  255, 255),  # 14 bright cyan
    (255, 255, 255),  # 15 white
]


# ---------------------------------------------------------------------------
# Keybinding: pygame key/unicode -> action value (Action enum int value)
# ---------------------------------------------------------------------------

def _build_keymap():
    """Return (key_map, unicode_map) if pygame is available."""
    if pygame is None:
        return {}, {}

    # Build a quick lookup from action int value -> action.
    # We map keys to Action enum values directly.
    key_map = {
        # Cardinal movement — arrow keys (map to vi equivalents)
        pygame.K_UP:    int(Action.COMPASS_N),
        pygame.K_DOWN:  int(Action.COMPASS_S),
        pygame.K_RIGHT: int(Action.COMPASS_E),
        pygame.K_LEFT:  int(Action.COMPASS_W),
        # vi keys
        pygame.K_k:     int(Action.COMPASS_N),
        pygame.K_j:     int(Action.COMPASS_S),
        pygame.K_l:     int(Action.COMPASS_E),
        pygame.K_h:     int(Action.COMPASS_W),
        pygame.K_y:     int(Action.COMPASS_NW),
        pygame.K_u:     int(Action.COMPASS_NE),
        pygame.K_b:     int(Action.COMPASS_SW),
        pygame.K_n:     int(Action.COMPASS_SE),
        # Misc directions
        pygame.K_PERIOD: int(Action.WAIT),
        pygame.K_LESS:   int(Action.UP),
        pygame.K_GREATER: int(Action.DOWN),
        # Commands
        pygame.K_s:     int(Action.SEARCH),
        pygame.K_COMMA: int(Action.PICKUP),
        pygame.K_a:     int(Action.APPLY),
        pygame.K_e:     int(Action.EAT),
        pygame.K_q:     int(Action.QUAFF),
        pygame.K_r:     int(Action.READ),
        pygame.K_d:     int(Action.DROP),
        pygame.K_z:     int(Action.ZAP),
        pygame.K_w:     int(Action.WIELD),
        pygame.K_p:     int(Action.PAY),
        pygame.K_t:     int(Action.THROW),
        pygame.K_c:     int(Action.CLOSE),
        pygame.K_o:     int(Action.OPEN),
    }

    # Unicode map: checked before key_map; covers shift-modified chars.
    unicode_map = {
        ">": int(Action.DOWN),
        "<": int(Action.UP),
    }

    return key_map, unicode_map


# ---------------------------------------------------------------------------
# Blstats field indices (from NLE / Nethax constants)
# ---------------------------------------------------------------------------

BL_X         = 0
BL_Y         = 1
BL_STR25     = 2
BL_DEX       = 4
BL_CON       = 5
BL_INT       = 6
BL_WIS       = 7
BL_CHA       = 8
BL_SCORE     = 9
BL_HP        = 10
BL_HPMAX     = 11
BL_DEPTH     = 12
BL_GOLD      = 13
BL_ENE       = 14
BL_ENEMAX    = 15
BL_AC        = 16
BL_XP        = 18
BL_TIME      = 20
BL_HUNGER    = 21

_HUNGER_NAMES = {0: "Satiated", 1: "Not Hungry", 2: "Hungry",
                 3: "Weak", 4: "Fainting", 5: "Fainted", 6: "Starved"}


# ---------------------------------------------------------------------------
# Player-name autogen (vendor parity).  Vendor NetHack picks a default name
# from the user's login when none is given (config.c::askname).  In our
# self-contained env we have no login, so seed-derive a stable name.
# ---------------------------------------------------------------------------

_NAME_PREFIXES = (
    "Adventurer", "Hero", "Wanderer", "Pilgrim", "Seeker", "Voyager",
    "Champion", "Knight", "Mystic", "Rogue", "Sage", "Scout",
)


def autogen_player_name(seed: int) -> str:
    """Generate a stable nethack-like default name from an int seed.

    Mirrors vendor `config.c::askname` fallback behavior (use the OS login
    when no name is supplied).  Produces e.g. "Hero742".
    """
    s = abs(int(seed))
    prefix = _NAME_PREFIXES[s % len(_NAME_PREFIXES)]
    suffix = s % 1000
    return f"{prefix}{suffix}"


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def _render_tile_pane(obs, tiles_array):
    """Return an (H, W, 3) uint8 numpy array for the tile pane."""
    glyphs = np.asarray(obs["glyphs"])  # (21, 79)
    # Pad to 80 cols so tile pane width = 80 * TILE_SIZE
    glyphs_80 = np.pad(glyphs, ((0, 0), (0, 1)), constant_values=0)

    # JAX render
    pixels_jax = render_pixels(
        jnp.array(glyphs_80, dtype=jnp.int16),
        GLYPH2TILE,
        tiles_array,
    )
    return np.asarray(pixels_jax)


_ALIGN_NAMES = {1: "Lawful", 0: "Neutral", -1: "Chaotic"}


def _format_status_lines(obs, last_action_name=None, role_idx=None, player_name=None):
    """Return (name_title_stats, dungeon_stats, message, action_label).

    Mirrors vendor/nethack/src/botl.c::do_statusline1 / do_statusline2.
    Pure Python, used by both the live HUD and the parity tests.

    Parameters
    ----------
    obs : NLE observation dict
    last_action_name : str | None — appended to dlvl_line for debugging
    role_idx : int | None — role index into _ROLE_RANK_TITLES (Valkyrie=10
        by default).  If None, falls back to 0 (Archeologist) for back-compat
        with callers that don't supply state.player_role.
    player_name : str | None — overrides the default "Player" name prefix.
    """
    blstats = np.asarray(obs["blstats"])
    message_bytes = np.asarray(obs["message"])
    msg_text = bytes(message_bytes[message_bytes != 0]).decode("ascii", errors="replace")

    hp     = int(blstats[BL_HP])
    hpmax  = int(blstats[BL_HPMAX])
    pw     = int(blstats[BL_ENE])
    pwmax  = int(blstats[BL_ENEMAX])
    ac     = int(blstats[BL_AC])
    xp     = int(blstats[BL_XP])
    depth  = int(blstats[BL_DEPTH])
    gold   = int(blstats[BL_GOLD])
    time_  = int(blstats[BL_TIME])
    st     = int(blstats[BL_STR25])
    dx     = int(blstats[BL_DEX])
    co     = int(blstats[BL_CON])
    in_    = int(blstats[BL_INT])
    wi     = int(blstats[BL_WIS])
    ch     = int(blstats[BL_CHA])
    align  = int(blstats[26])    # BL_ALIGN
    align_name = _ALIGN_NAMES.get(align, "Unaligned")

    # Resolve role rank title (Python-only lookup, not JIT-traced).
    # Wave 8 fix: use the actual role_idx passed in.  Previously the call
    # site hard-coded role=0, producing "Digger" for a level-1 Valkyrie
    # instead of the correct "Stripling".
    from Nethax.nethax.obs.nle_obs import role_rank_title
    try:
        title = role_rank_title(int(role_idx) if role_idx is not None else 0, xp)
    except Exception:
        title = "Adventurer"
    name_prefix = player_name if player_name else "Player"
    name_line = (f"{name_prefix} the {title}    "
                 f"St:{st} Dx:{dx} Co:{co} In:{in_} Wi:{wi} Ch:{ch}  "
                 f"{align_name}")

    dlvl_line = (f"Dlvl:{depth} $:{gold} HP:{hp}({hpmax}) "
                 f"Pw:{pw}({pwmax}) AC:{ac} Xp:{xp} T:{time_}")
    if last_action_name:
        dlvl_line += f"  Last:{last_action_name}"
    return name_line, dlvl_line, msg_text


def _draw_status_panel(screen, font, font_large, obs, last_action_name, y_offset,
                       role_idx=None, player_name=None):
    """Draw the bottom status panel onto screen, mirroring the NLE bot lines."""
    name_line, dlvl_line, msg_text = _format_status_lines(
        obs, last_action_name, role_idx=role_idx, player_name=player_name
    )

    blstats = np.asarray(obs["blstats"])
    hp = int(blstats[BL_HP])
    hpmax = int(blstats[BL_HPMAX])
    if hpmax > 0:
        ratio = hp / hpmax
    else:
        ratio = 1.0
    if ratio > 0.5:
        hp_color = (85, 255, 85)
    elif ratio > 0.25:
        hp_color = (255, 255, 85)
    else:
        hp_color = (255, 85, 85)

    panel_rect = pygame.Rect(0, y_offset, WINDOW_W, STATUS_PANEL_H)
    pygame.draw.rect(screen, (20, 20, 30), panel_rect)

    if msg_text:
        msg_surf = font.render(msg_text[:120], True, (230, 230, 180))
        screen.blit(msg_surf, (8, y_offset + 4))

    surf1 = font.render(name_line, True, (200, 200, 230))
    screen.blit(surf1, (8, y_offset + 26))

    surf2 = font.render(dlvl_line, True, hp_color)
    screen.blit(surf2, (8, y_offset + 48))


def _draw_inventory_pane(screen, font, obs, x_offset):
    """Draw an inventory overlay panel on the right."""
    pane_rect = pygame.Rect(x_offset, 0, INVENTORY_PANEL_W, TILE_PANE_H)
    overlay = pygame.Surface((INVENTORY_PANEL_W, TILE_PANE_H), pygame.SRCALPHA)
    overlay.fill((10, 10, 40, 210))
    screen.blit(overlay, (x_offset, 0))

    title = font.render("INVENTORY", True, (255, 255, 100))
    screen.blit(title, (x_offset + 8, 6))

    inv_letters = np.asarray(obs["inv_letters"])   # (55,)
    inv_strs    = np.asarray(obs["inv_strs"])       # (55, 80)

    y = 28
    shown = 0
    for i in range(55):
        letter = int(inv_letters[i])
        if letter == 0:
            continue
        item_bytes = inv_strs[i]
        item_text = bytes(item_bytes[item_bytes != 0]).decode("ascii", errors="replace")
        if not item_text:
            continue
        line = f"{chr(letter)}) {item_text[:38]}"
        surf = font.render(line, True, (200, 200, 230))
        screen.blit(surf, (x_offset + 8, y))
        y += 18
        shown += 1
        if y + 18 > TILE_PANE_H:
            break

    if shown == 0:
        empty_surf = font.render("(empty)", True, (120, 120, 120))
        screen.blit(empty_surf, (x_offset + 8, 28))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    if pygame is None:
        print("pygame is not installed.  Run: pip install pygame", file=sys.stderr)
        sys.exit(1)

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Nethax")
    clock = pygame.time.Clock()

    font       = pygame.font.SysFont("monospace", 15)
    font_large = pygame.font.SysFont("monospace", 24, bold=True)

    # Load tile sprites once.
    tiles_array = load_tiles()

    # Build key maps.
    key_map, unicode_map = _build_keymap()

    # Build reverse lookup: action value -> name string.
    action_names = {int(a): a.name for a in ACTIONS}

    # Initialise env.
    env = NethaxEnv()
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    print("Compiling nethax environment (first reset) — may take ~30-60s …")
    state, obs = env.reset(init_rng)
    print("Ready.")

    # Seed-derived player name (vendor parity for config.c::askname fallback).
    player_name = autogen_player_name(0)

    show_inventory = False
    last_action_name = "---"
    running = True

    while running:
        # ---- event handling ----
        action_val = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                # Quit keys
                if event.key == pygame.K_ESCAPE or (
                        event.key == pygame.K_q and
                        (event.mod & pygame.KMOD_SHIFT)):
                    running = False
                    continue

                # Inventory toggle
                if event.key == pygame.K_i:
                    show_inventory = not show_inventory
                    continue

                # Action lookup: unicode map first, then key map.
                av = unicode_map.get(event.unicode)
                if av is None:
                    av = key_map.get(event.key)
                if av is not None:
                    action_val = av

        # ---- env step ----
        if action_val is not None:
            last_action_name = action_names.get(action_val, str(action_val))
            rng, step_rng = jax.random.split(rng)
            state, obs, reward, done, _info = env.step(
                state, jnp.int32(action_val), step_rng
            )
            if done:
                # Auto-reset on death/ascension.
                rng, reset_rng = jax.random.split(rng)
                state, obs = env.reset(reset_rng)
                last_action_name = "---"

        # ---- render ----
        # Tile pane
        pixels = _render_tile_pane(obs, tiles_array)
        # pygame expects (W, H, 3) from surfarray
        tile_surface = pygame.surfarray.make_surface(pixels.transpose(1, 0, 2))
        screen.blit(tile_surface, (0, 0))

        # Status panel — pass real role index (from env state) so rank
        # titles render correctly (e.g. Stripling, not Digger, for a
        # level-1 Valkyrie).
        role_idx = int(np.asarray(state.player_role))
        _draw_status_panel(
            screen, font, font_large, obs, last_action_name, TILE_PANE_H,
            role_idx=role_idx, player_name=player_name,
        )

        # Inventory overlay (toggled with 'i')
        if show_inventory:
            inv_x = max(0, WINDOW_W - INVENTORY_PANEL_W)
            _draw_inventory_pane(screen, font, obs, inv_x)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
