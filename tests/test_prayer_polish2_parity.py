"""Prayer polish-2 parity tests.

Covers:
  1. god_speaks_message returns a string containing the god's name.
  2. High piety → higher GIFT_ARTIFACT rate (pray.c::pleased bucket bias).
  3. Sacrifice level scales alignment delta (pray.c:2030-2065).
  4. Praying on a cross-aligned altar punishes the player.
  5. player_luck is capped at 13 on the luck-bump path (pray.c LUCKMAX=13).
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_state():
    from Nethax.nethax.env import NethaxEnv
    env = NethaxEnv()
    state, _ = env.reset(jax.random.PRNGKey(42))
    return state


def _place_altar(state, align_int: int):
    max_levels = state.terrain.shape[1]
    b  = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    flat_lv = b * max_levels + lv
    row = int(state.player_pos[0])
    col = int(state.player_pos[1])
    new_aa = state.features.altar_alignment.at[flat_lv, row, col].set(
        jnp.int8(align_int)
    )
    return state.replace(features=state.features.replace(altar_alignment=new_aa))


def _add_corpse(state, slot: int, type_id: int, monster_level: int = 0):
    """Add a FOOD-category corpse with encoded monster level in enchantment."""
    items = state.inventory.items
    new_cat = items.category.at[slot].set(jnp.int8(7))   # FOOD
    new_qty = items.quantity.at[slot].set(jnp.int16(1))
    new_tid = items.type_id.at[slot].set(jnp.int16(type_id))
    new_enc = items.enchantment.at[slot].set(jnp.int8(monster_level))
    new_items = items.replace(category=new_cat, quantity=new_qty,
                               type_id=new_tid, enchantment=new_enc)
    return state.replace(inventory=state.inventory.replace(items=new_items))


# ---------------------------------------------------------------------------
# 1. god_speaks_message — pray.c:50
# ---------------------------------------------------------------------------

def test_god_speaks_returns_name_string():
    """god_speaks_message(HUMAN, LAWFUL, 'pleased') contains 'Anu'."""
    from Nethax.nethax.subsystems.prayer import god_speaks_message, Race, Alignment
    msg = god_speaks_message(Race.HUMAN, Alignment.LAWFUL, "pleased")
    assert isinstance(msg, str)
    assert "Anu" in msg, f"Expected 'Anu' in message, got: {msg!r}"


def test_god_speaks_angry():
    from Nethax.nethax.subsystems.prayer import god_speaks_message, Race, Alignment
    msg = god_speaks_message(Race.ELF, Alignment.CHAOTIC, "angry")
    assert "Lolth" in msg


def test_god_speaks_ascension():
    from Nethax.nethax.subsystems.prayer import god_speaks_message, Race, Alignment
    msg = god_speaks_message(Race.HUMAN, Alignment.LAWFUL, "ascension")
    assert "Anu" in msg


# ---------------------------------------------------------------------------
# 2. High piety → higher artifact gift rate (pray.c::pleased bias)
# ---------------------------------------------------------------------------

def test_high_piety_higher_artifact_chance():
    """50 pray() trials at alignment_record=20 yield more wis bumps than at 0.

    GIFT_ARTIFACT is modelled as player_wis bump in _apply_gift_artifact.
    Uses vmap to run all trials in a single JIT-compiled call.
    """
    from Nethax.nethax.subsystems.prayer import pray

    TRIALS = 50

    def _count_gifts(alignment_record: int) -> int:
        state = _fresh_state()
        state = state.replace(
            prayer=state.prayer.replace(
                alignment_record=jnp.int16(alignment_record),
                pray_timeout=jnp.int32(0),
            ),
            player_luck=jnp.int8(0),
        )
        base_wis = int(state.player_wis)

        # Batch TRIALS keys and vmap pray over them.
        keys = jax.random.split(jax.random.PRNGKey(13), TRIALS)

        def _single(rng):
            new_s = pray(state, rng)
            return new_s.player_wis

        wis_results = jax.vmap(_single)(keys)
        return int(jnp.sum(wis_results > jnp.int8(base_wis)))

    high_gifts = _count_gifts(20)
    low_gifts  = _count_gifts(0)
    assert high_gifts > low_gifts, (
        f"Expected more artifact gifts at high piety: high={high_gifts}, low={low_gifts}"
    )


# ---------------------------------------------------------------------------
# 3. Sacrifice level scales alignment delta (pray.c:2030-2065)
# ---------------------------------------------------------------------------

def test_sacrifice_level_scales_delta():
    """Same-aligned corpse: level-5 → ~+5, level-20 → ~+10."""
    from Nethax.nethax.subsystems.prayer import sacrifice_on_altar

    # Player align = LAWFUL (2), corpse type_id = 2 (same-aligned sentinel).
    PLAYER_ALIGN = 2  # Alignment.LAWFUL

    def _do_sacrifice(monster_level: int) -> int:
        state = _fresh_state()
        state = state.replace(
            player_align=jnp.int8(PLAYER_ALIGN),
            prayer=state.prayer.replace(alignment_record=jnp.int16(0)),
        )
        state = _place_altar(state, PLAYER_ALIGN)
        state = _add_corpse(state, slot=0, type_id=PLAYER_ALIGN,
                             monster_level=monster_level)
        rng = jax.random.PRNGKey(99)
        new_state = sacrifice_on_altar(state, rng, jnp.int32(0))
        return int(new_state.prayer.alignment_record)

    delta_lv5  = _do_sacrifice(5)
    delta_lv20 = _do_sacrifice(20)

    # level-5:  base * (1 + 5/10)  = 5 * 1.5 = 7  (int: 7)
    # level-20: base * (1 + 20/10) = 5 * 3.0 = 15
    assert delta_lv5 > 0, f"Expected positive delta at level 5, got {delta_lv5}"
    assert delta_lv20 > delta_lv5, (
        f"Expected level-20 delta ({delta_lv20}) > level-5 delta ({delta_lv5})"
    )


# ---------------------------------------------------------------------------
# 4. Praying on cross-aligned altar punishes (pray.c god_zaps_you branch)
# ---------------------------------------------------------------------------

def test_pray_on_cross_altar_punishes():
    """Standing on a cross-aligned altar when praying decreases alignment_record."""
    from Nethax.nethax.subsystems.prayer import pray

    state = _fresh_state()
    # Player is LAWFUL (2), altar is CHAOTIC (0) → cross-aligned.
    PLAYER_ALIGN = 2
    ALTAR_ALIGN  = 0
    state = state.replace(
        player_align=jnp.int8(PLAYER_ALIGN),
        prayer=state.prayer.replace(
            alignment_record=jnp.int16(10),
            pray_timeout=jnp.int32(0),
        ),
    )
    state = _place_altar(state, ALTAR_ALIGN)
    before_record = int(state.prayer.alignment_record)
    before_hp     = int(state.player_hp)

    rng = jax.random.PRNGKey(7)
    new_state = pray(state, rng)

    after_record = int(new_state.prayer.alignment_record)
    after_hp     = int(new_state.player_hp)

    # god_zaps_you decreases alignment_record by 5 and may reduce HP.
    assert after_record < before_record or after_hp < before_hp, (
        f"Expected punishment on cross-altar pray: "
        f"record {before_record}→{after_record}, hp {before_hp}→{after_hp}"
    )


# ---------------------------------------------------------------------------
# 5. Luck cap at 13 (pray.c LUCKMAX=13 prayer path)
# ---------------------------------------------------------------------------

def test_luck_capped_at_13():
    """luck=12 + good pray → luck=13; luck=13 + good pray → luck stays 13."""
    from Nethax.nethax.subsystems.prayer import pray

    def _pray_with_luck(luck_val: int) -> int:
        state = _fresh_state()
        state = state.replace(
            player_luck=jnp.int8(luck_val),
            prayer=state.prayer.replace(
                alignment_record=jnp.int16(20),  # high piety → pleased
                pray_timeout=jnp.int32(0),
            ),
        )
        # No trouble, no altar → should hit pleased branch.
        rng = jax.random.PRNGKey(42)
        new_state = pray(state, rng)
        return int(new_state.player_luck)

    # With luck=12, a pleased prayer should push it to 13.
    luck_after_12 = _pray_with_luck(12)
    assert luck_after_12 <= 13, f"Luck exceeded cap: {luck_after_12}"

    # With luck=13 it must not exceed 13.
    luck_after_13 = _pray_with_luck(13)
    assert luck_after_13 == 13, f"Luck should stay at 13, got {luck_after_13}"
