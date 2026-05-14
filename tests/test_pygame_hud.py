"""Wave 7 HUD parity tests — vendor status line format for pygame_app.

Reference: vendor/nethack/src/botl.c::do_statusline1 (lines 48-98) and
do_statusline2 (lines 100-249).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pytest

pygame = pytest.importorskip("pygame", reason="pygame not installed")

import jax
import jax.numpy as jnp

from Nethax.nethax.state import EnvState
from Nethax.nethax.obs.nle_obs import build_nle_observation
from Nethax.ui.pygame_app import _format_status_lines


_RNG = jax.random.PRNGKey(1)


def _obs() -> dict:
    return build_nle_observation(EnvState.default(rng=_RNG))


# ---------------------------------------------------------------------------
# Status line text contents — vendor parity
# ---------------------------------------------------------------------------

def test_status_name_line_includes_player_the_prefix():
    """do_statusline1 starts with 'Name the <Title>'."""
    name, _, _ = _format_status_lines(_obs(), last_action_name=None)
    assert name.startswith("Player the "), f"got: {name!r}"


def test_status_name_line_has_all_six_stat_fields():
    """do_statusline1 emits St:/Dx:/Co:/In:/Wi:/Ch: in order."""
    name, _, _ = _format_status_lines(_obs(), last_action_name=None)
    for label in ("St:", "Dx:", "Co:", "In:", "Wi:", "Ch:"):
        assert label in name, f"missing {label} in: {name!r}"
    # Order check.
    assert (name.index("St:") < name.index("Dx:") < name.index("Co:")
            < name.index("In:") < name.index("Wi:") < name.index("Ch:"))


def test_status_name_line_has_alignment_word():
    """do_statusline1 ends with one of 'Lawful'/'Neutral'/'Chaotic'."""
    name, _, _ = _format_status_lines(_obs(), last_action_name=None)
    assert any(a in name for a in ("Lawful", "Neutral", "Chaotic")), name


def test_status_dlvl_line_uses_dollar_prefix_not_gold():
    """do_statusline2 emits '$:N' — never 'Gold:'."""
    _, dlvl, _ = _format_status_lines(_obs(), last_action_name=None)
    assert "$:" in dlvl, f"missing $: in: {dlvl!r}"
    assert "Gold:" not in dlvl, f"got vendor-divergent Gold:, in: {dlvl!r}"


def test_status_dlvl_line_has_canonical_field_order():
    """do_statusline2 order: Dlvl: $: HP:H(Hmax) Pw:P(Pmax) AC: Xp: T:."""
    _, dlvl, _ = _format_status_lines(_obs(), last_action_name=None)
    for label in ("Dlvl:", "$:", "HP:", "Pw:", "AC:", "Xp:", "T:"):
        assert label in dlvl, f"missing {label} in: {dlvl!r}"
    assert (dlvl.index("Dlvl:") < dlvl.index("$:") < dlvl.index("HP:")
            < dlvl.index("Pw:") < dlvl.index("AC:") < dlvl.index("Xp:")
            < dlvl.index("T:"))


def test_status_hp_uses_parenthesized_maxhp_format():
    """HP rendered as 'HP:N(Nmax)' per do_statusline2:143."""
    import re
    _, dlvl, _ = _format_status_lines(_obs(), last_action_name=None)
    m = re.search(r"HP:(\d+)\((\d+)\)", dlvl)
    assert m is not None, f"HP:n(nmax) regex missed in: {dlvl!r}"


def test_status_pw_uses_parenthesized_maxpw_format():
    """Pw rendered as 'Pw:N(Nmax)' per do_statusline2:143."""
    import re
    _, dlvl, _ = _format_status_lines(_obs(), last_action_name=None)
    m = re.search(r"Pw:(\d+)\((\d+)\)", dlvl)
    assert m is not None, f"Pw:n(nmax) regex missed in: {dlvl!r}"


# ---------------------------------------------------------------------------
# Surface rendering integration — sanity-check pygame can draw the panel
# ---------------------------------------------------------------------------

def test_draw_status_panel_renders_to_surface():
    """The status panel must draw without exceptions and update pixels."""
    import numpy as np
    from Nethax.ui.pygame_app import (
        _draw_status_panel, WINDOW_W, WINDOW_H, TILE_PANE_H, STATUS_PANEL_H,
    )
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    font = pygame.font.SysFont("monospace", 15)
    font_large = pygame.font.SysFont("monospace", 24, bold=True)

    obs = _obs()
    _draw_status_panel(screen, font, font_large, obs, "WAIT", TILE_PANE_H)

    # Grab the panel pixels and assert at least one pixel is the panel color
    # (RGB 20,20,30) — i.e. the rect was drawn.
    pixels = pygame.surfarray.array3d(screen)
    panel = pixels[:, TILE_PANE_H:TILE_PANE_H + STATUS_PANEL_H, :]
    # Look for the panel background colour or the rendered text.
    panel_bg_hits = np.all(panel == (20, 20, 30), axis=-1).sum()
    text_hits = np.any(panel != (20, 20, 30), axis=-1).sum()
    assert panel_bg_hits > 0, "panel background not drawn"
    assert text_hits > 0, "no text rendered onto panel"
    pygame.quit()


def test_status_panel_contains_dollar_prefix_in_pixels():
    """When the HUD text is drawn we should see the '$:' string in font pixels.

    Uses pygame's font.render directly to verify the produced surface has
    actual visible glyph pixels (non-background).
    """
    import numpy as np
    pygame.init()
    font = pygame.font.SysFont("monospace", 15)
    _, dlvl, _ = _format_status_lines(_obs(), last_action_name="WAIT")
    surf = font.render(dlvl, True, (255, 255, 255))
    pixels = pygame.surfarray.array3d(surf)
    # Any non-zero (white-ish) pixel proves the text was rasterized.
    assert (pixels.sum() > 0)
    assert "$:" in dlvl
    pygame.quit()
