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
    i          toggle grouped inventory pane
    ;          look here (show what's at feet)
    +          spell menu
    \\          discoveries menu
    Q          quit
    ESC        quit
"""

from __future__ import annotations

import random
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
from Nethax.nethax.constants.roles import Role
from Nethax.nethax.constants.races import Race
from Nethax.tiles import render_pixels, GLYPH2TILE, TILE_SIZE
from Nethax.tiles.renderer import load_tiles


# ---------------------------------------------------------------------------
# Class selection — vendor parity with role.c::role_init and
# wintty.c::tty_player_selection (sequential prompts).
# ---------------------------------------------------------------------------

# 13 roles in canonical vendor order (role.c::roles[], lines 27-586).
_ROLE_NAMES = [
    "Archeologist", "Barbarian", "Caveman", "Healer", "Knight",
    "Monk", "Priest", "Ranger", "Rogue", "Samurai", "Tourist",
    "Valkyrie", "Wizard",
]
# Maps display index (1-based) -> Role enum value
_ROLE_MAP = {i + 1: Role(i) for i in range(len(_ROLE_NAMES))}

# 5 races (role.c::races[], lines 617-726).
_RACE_NAMES = ["Human", "Elf", "Dwarf", "Gnome", "Orc"]
_RACE_MAP = {i + 1: Race(i) for i in range(len(_RACE_NAMES))}

# Alignments (role.c::aligns[]).
_ALIGN_OPTS = ["Lawful", "Neutral", "Chaotic"]
# env.reset alignment: 0=Lawful, 1=Neutral, 2=Chaotic
_ALIGN_MAP = {1: 0, 2: 1, 3: 2}

_GENDER_OPTS = ["Male", "Female"]

# Defaults used when bailing (Q/ESC) or --no-select.
_DEFAULT_ROLE      = Role.VALKYRIE
_DEFAULT_RACE      = Race.HUMAN
_DEFAULT_ALIGNMENT = 0   # Lawful
_DEFAULT_GENDER    = "Female"


def _draw_selection_menu(screen, font, font_large, title: str, options: list[str]) -> None:
    """Render a centered selection menu onto screen.

    Vendor wintty.c::tty_player_selection draws a sequential prompt with
    numbered choices.  We replicate that as a simple full-screen overlay.
    """
    screen.fill((0, 0, 0))
    title_surf = font_large.render(title, True, (255, 255, 100))
    tw = title_surf.get_width()
    screen.blit(title_surf, ((screen.get_width() - tw) // 2, 40))

    start_y = 100
    for i, opt in enumerate(options, start=1):
        line = f"{i}) {opt}"
        surf = font.render(line, True, (200, 200, 230))
        screen.blit(surf, (80, start_y + (i - 1) * 22))

    hint = font.render("Enter = random    Q/ESC = defaults", True, (120, 120, 120))
    screen.blit(hint, (80, start_y + len(options) * 22 + 20))
    pygame.display.flip()


def run_selection_screen(screen, font, font_large, *, key_iter=None) -> tuple:
    """Run the sequential class-selection prompts.

    Vendor reference: wintty.c::tty_player_selection (sequential prompts
    for role, race, alignment, gender) and role.c::role_init (default
    assignment).

    Parameters
    ----------
    key_iter : optional iterator yielding (key, unicode) tuples.  When
        provided, the screen reads keys from `key_iter` instead of
        pygame.event.get().  Used by headless tests to bypass the SDL
        dummy-driver event queue (which doesn't reliably surface
        ``event.post`` traffic to ``event.get``).

    Returns (role, race, alignment, gender) where:
      role      : Role enum
      race      : Race enum
      alignment : int  0=Lawful 1=Neutral 2=Chaotic
      gender    : str  "Male" | "Female"
    """
    prompts = [
        ("Choose your Role", _ROLE_NAMES, _ROLE_MAP),
        ("Choose your Race", _RACE_NAMES, _RACE_MAP),
        ("Choose your Alignment", _ALIGN_OPTS, _ALIGN_MAP),
        ("Choose your Gender", _GENDER_OPTS, {1: "Male", 2: "Female"}),
    ]

    results = []
    bail = False

    for title, options, mapping in prompts:
        if bail:
            # Fill remaining with None (defaults applied below).
            results.append(None)
            continue

        _draw_selection_menu(screen, font, font_large, title, options)

        chosen = None
        # Test path: read directly from the injected key iterator.
        if key_iter is not None:
            try:
                key, uni = next(key_iter)
            except StopIteration:
                key, uni = (pygame.K_RETURN, "\r")
            if key == pygame.K_ESCAPE or key == pygame.K_q:
                bail = True
            elif key == pygame.K_RETURN:
                chosen = random.choice(list(mapping.values()))
            elif uni and uni.isdigit():
                n = int(uni)
                if n in mapping:
                    chosen = mapping[n]
            results.append(chosen)
            continue

        # Live path: busy-poll pygame's event queue.
        waiting = True
        max_idle_iters = 60 * 30   # ~30s safety bound for headless runs
        idle = 0
        while waiting and idle < max_idle_iters:
            pygame.event.pump()
            had_event = False
            for event in pygame.event.get():
                had_event = True
                if event.type == pygame.QUIT:
                    bail = True
                    waiting = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE,) or (
                        event.key == pygame.K_q
                    ):
                        bail = True
                        waiting = False
                    elif event.key == pygame.K_RETURN:
                        chosen = random.choice(list(mapping.values()))
                        waiting = False
                    elif event.unicode and event.unicode.isdigit():
                        n = int(event.unicode)
                        if n in mapping:
                            chosen = mapping[n]
                            waiting = False
            if not had_event:
                idle += 1
                pygame.time.wait(16)
            else:
                idle = 0

        results.append(chosen)

    # Apply defaults for any None entries (bail path).
    role = results[0] if results[0] is not None else _DEFAULT_ROLE
    race = results[1] if results[1] is not None else _DEFAULT_RACE
    alignment = results[2] if results[2] is not None else _DEFAULT_ALIGNMENT
    gender = results[3] if results[3] is not None else _DEFAULT_GENDER

    return role, race, alignment, gender


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


def _draw_inventory_pane(screen, font, state, x_offset):
    """Draw a grouped inventory overlay panel on the right.

    Uses build_grouped_inv_text (vendor invent.c::display_inventory parity)
    instead of the flat inv_strs approach.
    Vendor citation: inv_strs.py::build_grouped_inv_text.
    """
    from Nethax.nethax.obs.inv_strs import build_grouped_inv_text

    pane_rect = pygame.Rect(x_offset, 0, INVENTORY_PANEL_W, TILE_PANE_H)
    overlay = pygame.Surface((INVENTORY_PANEL_W, TILE_PANE_H), pygame.SRCALPHA)
    overlay.fill((10, 10, 40, 210))
    screen.blit(overlay, (x_offset, 0))

    title = font.render("INVENTORY", True, (255, 255, 100))
    screen.blit(title, (x_offset + 8, 6))

    lines = build_grouped_inv_text(state)

    y = 28
    for line in lines:
        # Class headers rendered in a different colour.
        is_header = not line.startswith((" ", "\t")) and not (" - " in line)
        color = (255, 220, 80) if is_header else (200, 200, 230)
        surf = font.render(line[:46], True, color)
        screen.blit(surf, (x_offset + 8, y))
        y += 18
        if y + 18 > TILE_PANE_H:
            break

    if not lines:
        empty_surf = font.render("(empty)", True, (120, 120, 120))
        screen.blit(empty_surf, (x_offset + 8, 28))


def _draw_overlay_text(screen, font, lines: list[str], title: str = "") -> None:
    """Draw a full-screen text overlay for look/spell/discovery menus."""
    overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
    overlay.fill((0, 0, 20, 220))
    screen.blit(overlay, (0, 0))

    y = 20
    if title:
        t = font.render(title, True, (255, 255, 100))
        screen.blit(t, (20, y))
        y += 24

    for line in lines:
        surf = font.render(line[:100], True, (200, 200, 230))
        screen.blit(surf, (20, y))
        y += 18
        if y + 18 > WINDOW_H - 30:
            break

    hint = font.render("[any key to close]", True, (100, 100, 100))
    screen.blit(hint, (20, WINDOW_H - 24))
    pygame.display.flip()


def _wait_for_keypress() -> None:
    """Block until any key is pressed (for overlay dismiss)."""
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN:
                return


def _draw_tombstone_screen(screen, font, font_large, lines: list[str]) -> None:
    """Render the RIP tombstone as a post-game screen and wait for keypress."""
    screen.fill((0, 0, 0))
    y = 40
    for line in lines:
        surf = font.render(line, True, (180, 180, 180))
        screen.blit(surf, (40, y))
        y += 18
    hint = font_large.render("Press any key to restart", True, (255, 255, 100))
    screen.blit(hint, (40, y + 20))
    pygame.display.flip()
    _wait_for_keypress()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    if pygame is None:
        print("pygame is not installed.  Run: pip install pygame", file=sys.stderr)
        sys.exit(1)

    # --no-select flag: skip class-selection prompts and use defaults.
    no_select = "--no-select" in sys.argv

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

    # Class selection — vendor wintty.c::tty_player_selection.
    # Vendor role.c::role_init provides defaults when skipped.
    if no_select:
        chosen_role      = _DEFAULT_ROLE
        chosen_race      = _DEFAULT_RACE
        chosen_alignment = _DEFAULT_ALIGNMENT
        chosen_gender    = _DEFAULT_GENDER
    else:
        chosen_role, chosen_race, chosen_alignment, chosen_gender = (
            run_selection_screen(screen, font, font_large)
        )

    # Initialise env.
    env = NethaxEnv()
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    print("Compiling nethax environment (first reset) — may take ~30-60s …")
    state, obs = env.reset(init_rng, role=chosen_role, race=chosen_race,
                           alignment=chosen_alignment)
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

                # Inventory toggle — grouped (vendor invent.c::display_inventory).
                if event.key == pygame.K_i:
                    show_inventory = not show_inventory
                    continue

                # ';' — look here (vendor invent.c::look_here).
                if event.unicode == ";":
                    from Nethax.nethax.obs.look import build_look_here_text
                    text = build_look_here_text(state)
                    _draw_overlay_text(screen, font, text.splitlines(), title=";  Look here")
                    _wait_for_keypress()
                    continue

                # '+' — spell menu (vendor spell.c::dospellmenu).
                if event.unicode == "+":
                    from Nethax.nethax.obs.spell_menu import build_spell_menu_text
                    lines = build_spell_menu_text(state)
                    _draw_overlay_text(screen, font, lines, title="+  Spells")
                    _wait_for_keypress()
                    continue

                # '\' — discoveries (vendor o_init.c::dodiscovered).
                if event.unicode == "\\":
                    from Nethax.nethax.obs.discovery import build_discovery_text
                    rows = build_discovery_text(state)
                    import numpy as _np
                    disc_lines = [
                        bytes(_np.asarray(row).tolist()).rstrip(b"\x00").decode("ascii", errors="replace")
                        for row in rows
                    ]
                    _draw_overlay_text(screen, font, disc_lines, title="\\  Discoveries")
                    _wait_for_keypress()
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
                # Game over — show tombstone then reset.
                # Vendor rip.c::genl_outrip.
                from Nethax.nethax.obs.tombstone import build_tombstone
                msg_bytes = np.asarray(obs["message"])
                killer = bytes(msg_bytes[msg_bytes != 0]).decode("ascii", errors="replace") or "unknown cause"
                tombstone_lines = build_tombstone(
                    state,
                    name=player_name,
                    killer=killer,
                )
                _draw_tombstone_screen(screen, font, font_large, tombstone_lines)

                # New game: run selection again (or use no-select defaults).
                if no_select:
                    chosen_role      = _DEFAULT_ROLE
                    chosen_race      = _DEFAULT_RACE
                    chosen_alignment = _DEFAULT_ALIGNMENT
                else:
                    chosen_role, chosen_race, chosen_alignment, _ = (
                        run_selection_screen(screen, font, font_large)
                    )

                rng, reset_rng = jax.random.split(rng)
                state, obs = env.reset(reset_rng, role=chosen_role,
                                       race=chosen_race, alignment=chosen_alignment)
                last_action_name = "---"
                show_inventory = False

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

        # Inventory overlay (toggled with 'i') — grouped by class.
        if show_inventory:
            inv_x = max(0, WINDOW_W - INVENTORY_PANEL_W)
            _draw_inventory_pane(screen, font, state, inv_x)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
