"""Engraving polish parity tests — wipe-on-step, blood kind, enum numbering.

Vendor citations:
  engrave.h        — DUST=1, ENGRAVE=2, BURN=3, MARK=4, ENGR_BLOOD=5
  engrave.c::wipe_engr_at (u_wipe_engr) lines 263-290 — per-step erosion
  engrave.c::doengrave line 573 — vampire/demon finger → ENGR_BLOOD
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

from Nethax.nethax.state import EnvState, StaticParams
from Nethax.nethax.constants.tiles import TileType
from Nethax.nethax.constants.monsters import MONSTERS, MonsterSymbol
from Nethax.nethax.subsystems.engrave import (
    ENGR_NONE,
    ENGR_DUST,
    ENGR_ENGRAVE,
    ENGR_BURN,
    ENGR_MARK,
    ENGR_BLOOD,
    handle_engrave,
    wipe_engr_on_step,
)
from Nethax.nethax.subsystems.polymorph import polymorph_player

_RNG = jax.random.PRNGKey(0)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _floor_state(player_pos=(5, 10)):
    static = StaticParams()
    state = EnvState.default(rng=_RNG, static=static)
    floor_map = jnp.full(
        (static.map_h, static.map_w), TileType.FLOOR, dtype=jnp.int8
    )
    return state.replace(
        terrain=state.terrain.at[0, 0].set(floor_map),
        player_pos=jnp.array(player_pos, dtype=jnp.int16),
    )


def _write_kind(state, row, col, kind):
    """Write an Elbereth engraving of the given kind directly into state."""
    from Nethax.nethax.subsystems.engrave import _elbereth_bytes_array
    bytes_vec = _elbereth_bytes_array()
    eng = state.engrave
    return state.replace(engrave=eng.replace(
        text=eng.text.at[row, col, :].set(bytes_vec),
        has_engraving=eng.has_engraving.at[row, col].set(jnp.bool_(True)),
        engraving_kind=eng.engraving_kind.at[row, col].set(jnp.int8(kind)),
    ))


# ---------------------------------------------------------------------------
# test_enum_dust_is_1
# ---------------------------------------------------------------------------

def test_enum_dust_is_1():
    """Enum values must match vendor engrave.h: DUST=1, ENGRAVE=2, BURN=3.

    Cite: vendor/nethack/include/engrave.h ENGR_DUST=1, ENGRAVE=2, BURN=3.
    """
    assert ENGR_DUST    == 1, f"ENGR_DUST expected 1, got {ENGR_DUST}"
    assert ENGR_ENGRAVE == 2, f"ENGR_ENGRAVE expected 2, got {ENGR_ENGRAVE}"
    assert ENGR_BURN    == 3, f"ENGR_BURN expected 3, got {ENGR_BURN}"
    assert ENGR_MARK    == 4, f"ENGR_MARK expected 4, got {ENGR_MARK}"
    assert ENGR_BLOOD   == 5, f"ENGR_BLOOD expected 5, got {ENGR_BLOOD}"


# ---------------------------------------------------------------------------
# test_dust_engraving_erodes_over_steps
# ---------------------------------------------------------------------------

def test_dust_engraving_erodes_over_steps():
    """DUST engraving should erase (has_engraving→False) at least once in 20 steps.

    Cite: engrave.c::wipe_engr_at lines 263-290 — DUST erodes on every step.
    50% per step → P(never erased in 20) = 0.5^20 < 0.0001%.
    """
    row, col = 5, 10
    state = _floor_state(player_pos=(row, col))
    state = _write_kind(state, row, col, ENGR_DUST)
    assert bool(state.engrave.has_engraving[row, col]), "Elbereth must exist before steps"

    erased = False
    for i in range(20):
        if not bool(state.engrave.has_engraving[row, col]):
            erased = True
            break
        rng_i = jax.random.PRNGKey(i + 500)
        state = wipe_engr_on_step(state, rng_i)

    if not erased:
        erased = not bool(state.engrave.has_engraving[row, col])

    assert erased, (
        "DUST Elbereth should have been erased within 20 steps "
        "(p(not erased) < 0.0001%)"
    )


# ---------------------------------------------------------------------------
# test_burn_engraving_does_not_erode
# ---------------------------------------------------------------------------

def test_burn_engraving_does_not_erode():
    """BURN engraving must never erode on player movement (50 steps).

    Cite: engrave.c::wipe_engr_at line 278 — BURN only erodes on magical
    contact, never on plain movement.
    """
    row, col = 5, 10
    state = _floor_state(player_pos=(row, col))
    state = _write_kind(state, row, col, ENGR_BURN)

    for i in range(50):
        rng_i = jax.random.PRNGKey(i + 700)
        state = wipe_engr_on_step(state, rng_i)

    assert bool(state.engrave.has_engraving[row, col]), (
        "BURN Elbereth must not erode on player movement (50 steps)"
    )
    assert int(state.engrave.engraving_kind[row, col]) == ENGR_BURN, (
        "BURN kind must be preserved after 50 steps"
    )


# ---------------------------------------------------------------------------
# test_vampire_polyform_writes_blood
# ---------------------------------------------------------------------------

def test_vampire_polyform_writes_blood():
    """Finger-engraving while polymorphed into a vampire yields ENGR_BLOOD.

    Cite: engrave.c::doengrave line 573 — bloodtype check → ENGR_BLOOD.
    Vendor vampire entry indices: 222 (vampire), 223 (vampire lord),
    224 (Vlad the Impaler).
    """
    # Resolve vampire index dynamically.
    vampire_idx = next(
        i for i, m in enumerate(MONSTERS) if m.name == "vampire"
    )

    row, col = 5, 10
    state = _floor_state(player_pos=(row, col))

    # Polymorph into vampire.
    state = polymorph_player(state, _RNG, jnp.int16(vampire_idx), controlled=True)
    assert bool(state.polymorph.is_polymorphed), "Player should be polymorphed"

    state = handle_engrave(state, _RNG)

    kind = int(state.engrave.engraving_kind[row, col])
    assert kind == ENGR_BLOOD, (
        f"Vampire-polymorphed engraving should be ENGR_BLOOD (5), got {kind}"
    )
    assert bool(state.engrave.has_engraving[row, col]), "has_engraving should be True"
