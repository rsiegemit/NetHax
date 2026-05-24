"""Wave 46b: vendor-deep engraving parity tests.

Covers the helpers added in this wave:

  - is_elbereth(text)           — pure text predicate
  - engrave_scares_monster      — Elbereth + monster-eligibility gate
  - tick_engravings             — coarse per-turn DUST decay

Vendor cites:
  vendor/nethack/src/engrave.c::sengr_at           (lines 250-261)
  vendor/nethack/src/engrave.c::wipe_engr_at       (lines 270-289)
  vendor/nethack/src/monmove.c::onscary            (lines 240-303)
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.engrave import (
    ENGRAVE_TEXT_LEN,
    ENGR_NONE,
    ENGR_DUST,
    ENGR_BLOOD,
    ENGR_ENGRAVE,
    ENGR_BURN,
    ENGR_MARK,
    engrave_scares_monster,
    handle_engrave,
    is_elbereth,
    tick_engravings,
)


_RNG = jax.random.PRNGKey(0)


# --- Helpers ---------------------------------------------------------------

def _bytes_to_textvec(text: bytes) -> jnp.ndarray:
    """Pad ``text`` to ENGRAVE_TEXT_LEN bytes (zero-terminated)."""
    arr = list(text) + [0] * (ENGRAVE_TEXT_LEN - len(text))
    return jnp.asarray(arr[:ENGRAVE_TEXT_LEN], dtype=jnp.int8)


def _fresh_state(row: int = 5, col: int = 7) -> EnvState:
    state = EnvState.default(_RNG)
    return state.replace(player_pos=jnp.array([row, col], dtype=jnp.int16))


def _engrave_tile(state: EnvState, row: int, col: int, text: bytes, kind: int) -> EnvState:
    """Force a specific engraving at (row, col) — bypasses handle_engrave."""
    eng = state.engrave
    text_vec = _bytes_to_textvec(text)
    new_eng = eng.replace(
        text=eng.text.at[row, col, :].set(text_vec),
        has_engraving=eng.has_engraving.at[row, col].set(jnp.bool_(True)),
        engraving_kind=eng.engraving_kind.at[row, col].set(jnp.int8(kind)),
    )
    return state.replace(engrave=new_eng)


def _spawn_monster(
    state: EnvState,
    slot: int = 0,
    entry_idx: int = 12,  # jackal: not mindless, not humanoid, not Rider
    row: int = 4,
    col: int = 7,
    *,
    peaceful: bool = False,
    blind_timer: int = 0,
) -> EnvState:
    """Stamp a single monster into a slot for the scare-test."""
    mai = state.monster_ai
    new_mai = mai.replace(
        alive=mai.alive.at[slot].set(jnp.bool_(True)),
        entry_idx=mai.entry_idx.at[slot].set(jnp.int16(entry_idx)),
        pos=mai.pos.at[slot].set(jnp.array([row, col], dtype=jnp.int16)),
        peaceful=mai.peaceful.at[slot].set(jnp.bool_(peaceful)),
        blind_timer=mai.blind_timer.at[slot].set(jnp.int16(blind_timer)),
    )
    return state.replace(monster_ai=new_mai)


# --- 1. is_elbereth text predicate ----------------------------------------

def test_is_elbereth_text_predicate():
    """is_elbereth returns True only for the exact 'Elbereth' byte sequence."""
    assert bool(is_elbereth(_bytes_to_textvec(b"Elbereth"))) is True
    assert bool(is_elbereth(_bytes_to_textvec(b"elbereth"))) is False  # case
    assert bool(is_elbereth(_bytes_to_textvec(b"Elbereth!"))) is False  # extra
    assert bool(is_elbereth(_bytes_to_textvec(b""))) is False
    assert bool(is_elbereth(_bytes_to_textvec(b"Hello"))) is False


# --- 2. engrave_scares_monster --------------------------------------------

def test_elbereth_scares_hostile_seeing_monster():
    """Elbereth on the player's tile freezes a hostile, sighted, mind-bearing monster.

    Cite: vendor/nethack/src/monmove.c::onscary (line 295: sengr_at("Elbereth", ...)).
    """
    state = _fresh_state(row=5, col=7)
    state = _engrave_tile(state, 5, 7, b"Elbereth", ENGR_DUST)
    state = _spawn_monster(state, slot=0, entry_idx=12)  # jackal
    assert bool(engrave_scares_monster(state, 0)) is True


def test_blind_monster_not_scared():
    """A blind monster ignores Elbereth (vendor onscary: !mtmp->mcansee → False)."""
    state = _fresh_state(row=5, col=7)
    state = _engrave_tile(state, 5, 7, b"Elbereth", ENGR_DUST)
    state = _spawn_monster(state, slot=0, entry_idx=12, blind_timer=10)
    assert bool(engrave_scares_monster(state, 0)) is False


def test_mindless_monster_not_scared():
    """A mindless monster (M1_MINDLESS, e.g. yellow mold) ignores Elbereth."""
    state = _fresh_state(row=5, col=7)
    state = _engrave_tile(state, 5, 7, b"Elbereth", ENGR_DUST)
    state = _spawn_monster(state, slot=0, entry_idx=157)  # yellow mold
    assert bool(engrave_scares_monster(state, 0)) is False


def test_peaceful_monster_not_scared():
    """A peaceful monster ignores Elbereth (vendor onscary line 300)."""
    state = _fresh_state(row=5, col=7)
    state = _engrave_tile(state, 5, 7, b"Elbereth", ENGR_DUST)
    state = _spawn_monster(state, slot=0, entry_idx=12, peaceful=True)
    assert bool(engrave_scares_monster(state, 0)) is False


def test_non_elbereth_engraving_does_not_scare():
    """A non-Elbereth engraving (e.g. 'Hello') doesn't scare anything."""
    state = _fresh_state(row=5, col=7)
    state = _engrave_tile(state, 5, 7, b"Hello", ENGR_DUST)
    state = _spawn_monster(state, slot=0, entry_idx=12)
    assert bool(engrave_scares_monster(state, 0)) is False


# --- 3. tick_engravings: DUST decays, others persist ----------------------

def test_dust_decays_under_many_ticks():
    """Repeated tick_engravings calls eventually erase a DUST engraving.

    With p=1/15 per tick, P(survive 300 ticks) = (14/15)^300 ≈ 1.7e-9,
    so a single deterministic seed is overwhelmingly likely to erase it.
    """
    state = _fresh_state(row=5, col=7)
    state = _engrave_tile(state, 5, 7, b"Elbereth", ENGR_DUST)
    assert bool(state.engrave.has_engraving[5, 7]) is True

    rng = jax.random.PRNGKey(123)
    for _ in range(300):
        rng, sub = jax.random.split(rng)
        state = tick_engravings(state, sub)

    assert bool(state.engrave.has_engraving[5, 7]) is False
    assert int(state.engrave.engraving_kind[5, 7]) == int(ENGR_NONE)


def test_blood_engrave_burn_mark_persist_under_ticks():
    """BLOOD / ENGRAVE / BURN / MARK do NOT decay under tick_engravings.

    Cite: vendor/nethack/src/engrave.c::wipe_engr_at lines 270-289 — only
    DUST wears via plain movement.  BURN never erodes; ENGRAVE/MARK fade
    only via the rare 1/26 gate inside wipe_engr_on_step, not this tick.
    """
    cases = [
        (b"Elbereth", ENGR_BLOOD),
        (b"Elbereth", ENGR_ENGRAVE),
        (b"Elbereth", ENGR_BURN),
        (b"Elbereth", ENGR_MARK),
    ]
    for text, kind in cases:
        state = _fresh_state(row=5, col=7)
        state = _engrave_tile(state, 5, 7, text, kind)
        rng = jax.random.PRNGKey(42)
        for _ in range(500):
            rng, sub = jax.random.split(rng)
            state = tick_engravings(state, sub)
        assert bool(state.engrave.has_engraving[5, 7]) is True, (
            f"kind={kind} should persist but was erased"
        )
        assert int(state.engrave.engraving_kind[5, 7]) == int(kind)


def test_tick_only_touches_player_tile():
    """tick_engravings only affects the tile under the player.

    A DUST engraving at a different tile is unaffected regardless of
    how many ticks elapse.
    """
    state = _fresh_state(row=5, col=7)
    state = _engrave_tile(state, 10, 10, b"Elbereth", ENGR_DUST)  # not under player

    rng = jax.random.PRNGKey(7)
    for _ in range(200):
        rng, sub = jax.random.split(rng)
        state = tick_engravings(state, sub)

    assert bool(state.engrave.has_engraving[10, 10]) is True
    assert int(state.engrave.engraving_kind[10, 10]) == int(ENGR_DUST)


def test_handle_engrave_then_scares():
    """End-to-end: handle_engrave (Elbereth in dust) makes a jackal flee."""
    state = _fresh_state(row=5, col=7)
    state = handle_engrave(state, _RNG)
    state = _spawn_monster(state, slot=0, entry_idx=12)  # jackal
    assert bool(engrave_scares_monster(state, 0)) is True
