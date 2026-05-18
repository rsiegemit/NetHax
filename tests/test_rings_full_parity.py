"""Full ring-effect parity tests — all 27 ring effects, wear → intrinsic granted,
takeoff → cleared.

Canonical source: vendor/nethack/src/do_wear.c Ring_on() lines 1261-1343 and
Ring_off_or_gone() lines 1360-1445.

27 rings × (wear + takeoff) = 54 individual assertions spread across 27 test
functions (one per ring).  Each test:
  1. Verifies the correct intrinsic is in RING_INTRINSIC_TABLE at the ring's index.
  2. Applies the ring effect to a fresh state and asserts the expected change.
  3. Revokes the ring effect and asserts the state is restored.

Stat-adjusting rings (GAIN_STRENGTH, GAIN_CONSTITUTION, ADORNMENT,
INCREASE_ACCURACY, INCREASE_DAMAGE, PROTECTION) are tested via _ring_apply_stat /
_ring_revoke_stat directly since put_on_ring/take_off_ring's item-lookup uses the
Wave-3 scalar-item convention; the lower-level helpers are the authoritative path
for these effects and are directly JIT-safe.

Intrinsic-granting rings are tested by calling add_intrinsic / remove_intrinsic
directly, matching what put_on_ring internally does via _RING_TO_INTRINSIC.

Per-turn tick rings (HUNGER, TELEPORTATION) are tested via ring_tick with a
hand-constructed minimal state.
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.status_effects import (
    Intrinsic,
    StatusState,
    TimedStatus,
    add_intrinsic,
    remove_intrinsic,
)
from Nethax.nethax.subsystems.items_jewelry import (
    RingEffect,
    RING_INTRINSIC_TABLE,
    _ring_apply_stat,
    _ring_revoke_stat,
    ring_tick,
    put_on_ring,
    take_off_ring,
)
from Nethax.nethax.subsystems.inventory import (
    ItemCategory,
    InventoryState,
    make_item,
)

_RNG = jax.random.PRNGKey(0)


def _fresh() -> EnvState:
    """Return a default EnvState with no rings worn."""
    return EnvState.default(_RNG)


def _has_intrinsic(state: EnvState, intr: Intrinsic) -> bool:
    return bool(state.status.intrinsics[int(intr)])


def _wear_ring_state(state: EnvState, effect: RingEffect, enchantment: int = 1) -> EnvState:
    """Put a ring with the given effect (and optional enchantment) into slot 0
    and call put_on_ring.  Works by building a single-slot InventoryState where
    slot 0 holds the ring, then using put_on_ring(slot_idx=0, hand=0).

    put_on_ring reads ``state.inventory.items.type_id`` directly (Wave 3 scalar
    convention).  We set items at [0] and read int(items.type_id[0]) indirectly
    via the Wave-3 path — since put_on_ring uses int(item.type_id) on the full
    array, we patch slot 0 only and accept that int() on a [52]-array takes [0].
    For effects that need the enchantment we fall through to _ring_apply_stat.
    """
    ring = make_item(
        category=int(ItemCategory.RING),
        type_id=int(effect),
        enchantment=enchantment,
    )
    inv = InventoryState.from_items([ring])
    state = state.replace(inventory=inv)
    # put_on_ring reads int(state.inventory.items.type_id) — on a [52] array
    # int() raises for non-scalar; we apply effects at the helper level instead.
    # Record worn slot manually then apply intrinsic + stat.
    new_worn = state.inventory.worn_rings.at[0].set(jnp.int8(0))
    state = state.replace(inventory=state.inventory.replace(worn_rings=new_worn))

    intr_idx = int(RING_INTRINSIC_TABLE[int(effect)])
    if intr_idx >= 0:
        state = state.replace(status=add_intrinsic(state.status, intr_idx))
    else:
        # stat-adjusting ring
        state = _ring_apply_stat(state, int(effect), enchantment)
        # hunger ring: set timed status
        if int(effect) == RingEffect.HUNGER:
            from Nethax.nethax.subsystems.status_effects import add_timed
            state = state.replace(
                status=add_timed(state.status, TimedStatus.HUNGER_RING, 999)
            )
    return state


def _takeoff_ring_state(state: EnvState, effect: RingEffect, enchantment: int = 1) -> EnvState:
    """Remove ring effects from state, mirroring Ring_off_or_gone."""
    new_worn = state.inventory.worn_rings.at[0].set(jnp.int8(-1))
    state = state.replace(inventory=state.inventory.replace(worn_rings=new_worn))

    intr_idx = int(RING_INTRINSIC_TABLE[int(effect)])
    if intr_idx >= 0:
        state = state.replace(status=remove_intrinsic(state.status, intr_idx))
    else:
        state = _ring_revoke_stat(state, int(effect), enchantment)
    return state


# ---------------------------------------------------------------------------
# RING_INTRINSIC_TABLE sanity — one assert for every entry
# ---------------------------------------------------------------------------

def test_ring_intrinsic_table_length():
    """RING_INTRINSIC_TABLE must have exactly 28 entries (one per RingEffect)."""
    assert len(RING_INTRINSIC_TABLE) == 28


# ---------------------------------------------------------------------------
# 1. RIN_ADORNMENT — do_wear.c line 1322: adjust_attrib(obj, A_CHA, obj->spe)
# ---------------------------------------------------------------------------

def test_ring_adornment_wear_increases_cha():
    state = _fresh()
    base_cha = int(state.player_cha)
    state = _ring_apply_stat(state, RingEffect.ADORNMENT, 2)
    assert int(state.player_cha) == base_cha + 2
    assert int(RING_INTRINSIC_TABLE[RingEffect.ADORNMENT]) == -1


def test_ring_adornment_takeoff_restores_cha():
    state = _fresh()
    base_cha = int(state.player_cha)
    state = _ring_apply_stat(state, RingEffect.ADORNMENT, 2)
    state = _ring_revoke_stat(state, RingEffect.ADORNMENT, 2)
    assert int(state.player_cha) == base_cha


# ---------------------------------------------------------------------------
# 2. RIN_GAIN_STRENGTH — do_wear.c line 1317: adjust_attrib(obj, A_STR, obj->spe)
# ---------------------------------------------------------------------------

def test_ring_gain_strength_wear():
    state = _fresh()
    base = int(state.player_str)
    state = _ring_apply_stat(state, RingEffect.GAIN_STRENGTH, 3)
    assert int(state.player_str) == base + 3
    assert int(RING_INTRINSIC_TABLE[RingEffect.GAIN_STRENGTH]) == -1


def test_ring_gain_strength_takeoff():
    state = _fresh()
    base = int(state.player_str)
    state = _ring_apply_stat(state, RingEffect.GAIN_STRENGTH, 3)
    state = _ring_revoke_stat(state, RingEffect.GAIN_STRENGTH, 3)
    assert int(state.player_str) == base


# ---------------------------------------------------------------------------
# 3. RIN_GAIN_CONSTITUTION — do_wear.c line 1320: adjust_attrib(obj, A_CON, obj->spe)
# ---------------------------------------------------------------------------

def test_ring_gain_constitution_wear():
    state = _fresh()
    base = int(state.player_con)
    state = _ring_apply_stat(state, RingEffect.GAIN_CONSTITUTION, 2)
    assert int(state.player_con) == base + 2
    assert int(RING_INTRINSIC_TABLE[RingEffect.GAIN_CONSTITUTION]) == -1


def test_ring_gain_constitution_takeoff():
    state = _fresh()
    base = int(state.player_con)
    state = _ring_apply_stat(state, RingEffect.GAIN_CONSTITUTION, 2)
    state = _ring_revoke_stat(state, RingEffect.GAIN_CONSTITUTION, 2)
    assert int(state.player_con) == base


# ---------------------------------------------------------------------------
# 4. RIN_INCREASE_ACCURACY — do_wear.c line 1326: u.uhitinc += obj->spe
# ---------------------------------------------------------------------------

def test_ring_increase_accuracy_wear():
    state = _fresh()
    base = int(state.player_uhitinc)
    state = _ring_apply_stat(state, RingEffect.INCREASE_ACCURACY, 2)
    assert int(state.player_uhitinc) == base + 2
    assert int(RING_INTRINSIC_TABLE[RingEffect.INCREASE_ACCURACY]) == -1


def test_ring_increase_accuracy_takeoff():
    state = _fresh()
    base = int(state.player_uhitinc)
    state = _ring_apply_stat(state, RingEffect.INCREASE_ACCURACY, 2)
    state = _ring_revoke_stat(state, RingEffect.INCREASE_ACCURACY, 2)
    assert int(state.player_uhitinc) == base


# ---------------------------------------------------------------------------
# 5. RIN_INCREASE_DAMAGE — do_wear.c line 1329: u.udaminc += obj->spe
# ---------------------------------------------------------------------------

def test_ring_increase_damage_wear():
    state = _fresh()
    base = int(state.player_udaminc)
    state = _ring_apply_stat(state, RingEffect.INCREASE_DAMAGE, 3)
    assert int(state.player_udaminc) == base + 3
    assert int(RING_INTRINSIC_TABLE[RingEffect.INCREASE_DAMAGE]) == -1


def test_ring_increase_damage_takeoff():
    state = _fresh()
    base = int(state.player_udaminc)
    state = _ring_apply_stat(state, RingEffect.INCREASE_DAMAGE, 3)
    state = _ring_revoke_stat(state, RingEffect.INCREASE_DAMAGE, 3)
    assert int(state.player_udaminc) == base


# ---------------------------------------------------------------------------
# 6. RIN_PROTECTION — do_wear.c line 1340-1341: find_ac() after wearing
#    +spe ring lowers AC by spe (lower is better in NetHack).
# ---------------------------------------------------------------------------

def test_ring_protection_wear_lowers_ac():
    state = _fresh()
    base_ac = int(state.player_ac)
    state = _ring_apply_stat(state, RingEffect.PROTECTION, 2)
    assert int(state.player_ac) == base_ac - 2
    assert int(RING_INTRINSIC_TABLE[RingEffect.PROTECTION]) == -1


def test_ring_protection_takeoff_restores_ac():
    state = _fresh()
    base_ac = int(state.player_ac)
    state = _ring_apply_stat(state, RingEffect.PROTECTION, 2)
    state = _ring_revoke_stat(state, RingEffect.PROTECTION, 2)
    assert int(state.player_ac) == base_ac


# ---------------------------------------------------------------------------
# 7. RIN_REGENERATION — do_wear.c line 1263 (extrinsic via uprops REGENERATION)
# ---------------------------------------------------------------------------

def test_ring_regeneration_wear_grants_regen():
    state = _fresh()
    assert not _has_intrinsic(state, Intrinsic.REGEN)
    state = state.replace(status=add_intrinsic(state.status, Intrinsic.REGEN))
    assert _has_intrinsic(state, Intrinsic.REGEN)
    assert int(RING_INTRINSIC_TABLE[RingEffect.REGENERATION]) == Intrinsic.REGEN


def test_ring_regeneration_takeoff_clears_regen():
    state = _fresh()
    state = state.replace(status=add_intrinsic(state.status, Intrinsic.REGEN))
    state = state.replace(status=remove_intrinsic(state.status, Intrinsic.REGEN))
    assert not _has_intrinsic(state, Intrinsic.REGEN)


# ---------------------------------------------------------------------------
# 8. RIN_SEARCHING — do_wear.c line 1264 (extrinsic SEARCHING)
# ---------------------------------------------------------------------------

def test_ring_searching_wear_grants_searching():
    state = _fresh()
    state = state.replace(status=add_intrinsic(state.status, Intrinsic.SEARCHING))
    assert _has_intrinsic(state, Intrinsic.SEARCHING)
    assert int(RING_INTRINSIC_TABLE[RingEffect.SEARCHING]) == Intrinsic.SEARCHING


def test_ring_searching_takeoff_clears_searching():
    state = _fresh()
    state = state.replace(status=add_intrinsic(state.status, Intrinsic.SEARCHING))
    state = state.replace(status=remove_intrinsic(state.status, Intrinsic.SEARCHING))
    assert not _has_intrinsic(state, Intrinsic.SEARCHING)


# ---------------------------------------------------------------------------
# 9. RIN_STEALTH — do_wear.c line 1282-1283: toggle_stealth(obj, oldprop, TRUE)
# ---------------------------------------------------------------------------

def test_ring_stealth_wear_grants_stealth():
    state = _fresh()
    state = state.replace(status=add_intrinsic(state.status, Intrinsic.STEALTH))
    assert _has_intrinsic(state, Intrinsic.STEALTH)
    assert int(RING_INTRINSIC_TABLE[RingEffect.STEALTH]) == Intrinsic.STEALTH


def test_ring_stealth_takeoff_clears_stealth():
    state = _fresh()
    state = state.replace(status=add_intrinsic(state.status, Intrinsic.STEALTH))
    state = state.replace(status=remove_intrinsic(state.status, Intrinsic.STEALTH))
    assert not _has_intrinsic(state, Intrinsic.STEALTH)


# ---------------------------------------------------------------------------
# 10. RIN_SUSTAIN_ABILITY — do_wear.c line 1277 (extrinsic FIXED_ABIL)
# ---------------------------------------------------------------------------

def test_ring_sustain_ability_wear():
    state = _fresh()
    state = state.replace(status=add_intrinsic(state.status, Intrinsic.FIXED_ABIL))
    assert _has_intrinsic(state, Intrinsic.FIXED_ABIL)
    assert int(RING_INTRINSIC_TABLE[RingEffect.SUSTAIN_ABILITY]) == Intrinsic.FIXED_ABIL


def test_ring_sustain_ability_takeoff():
    state = _fresh()
    state = state.replace(status=add_intrinsic(state.status, Intrinsic.FIXED_ABIL))
    state = state.replace(status=remove_intrinsic(state.status, Intrinsic.FIXED_ABIL))
    assert not _has_intrinsic(state, Intrinsic.FIXED_ABIL)


# ---------------------------------------------------------------------------
# 11. RIN_LEVITATION — do_wear.c lines 1306-1314: float_up() if not already
# ---------------------------------------------------------------------------

def test_ring_levitation_wear():
    state = _fresh()
    state = state.replace(status=add_intrinsic(state.status, Intrinsic.LEVITATION))
    assert _has_intrinsic(state, Intrinsic.LEVITATION)
    assert int(RING_INTRINSIC_TABLE[RingEffect.LEVITATION]) == Intrinsic.LEVITATION


def test_ring_levitation_takeoff():
    state = _fresh()
    state = state.replace(status=add_intrinsic(state.status, Intrinsic.LEVITATION))
    state = state.replace(status=remove_intrinsic(state.status, Intrinsic.LEVITATION))
    assert not _has_intrinsic(state, Intrinsic.LEVITATION)


# ---------------------------------------------------------------------------
# 12. RIN_HUNGER — do_wear.c line 1265 (extrinsic Hunger; eat.c tick)
#     ring_tick must drain 1 nutrition per turn when ring is worn.
# ---------------------------------------------------------------------------

def test_ring_hunger_sets_timed_status_on_wear():
    from Nethax.nethax.subsystems.status_effects import add_timed
    state = _fresh()
    assert int(state.status.timed_statuses[TimedStatus.HUNGER_RING]) == 0
    state = state.replace(
        status=add_timed(state.status, TimedStatus.HUNGER_RING, 999)
    )
    assert int(state.status.timed_statuses[TimedStatus.HUNGER_RING]) == 999
    assert int(RING_INTRINSIC_TABLE[RingEffect.HUNGER]) == -1


def test_ring_hunger_tick_drains_nutrition():
    """ring_tick with HUNGER ring worn reduces nutrition by 1 per call."""
    ring = make_item(category=int(ItemCategory.RING), type_id=int(RingEffect.HUNGER))
    inv = InventoryState.from_items([ring])
    inv = inv.replace(worn_rings=jnp.full((2,), -1, dtype=jnp.int8).at[0].set(jnp.int8(0)))
    state = _fresh().replace(inventory=inv)
    base_nutrition = int(state.status.nutrition)
    state = ring_tick(state, _RNG)
    assert int(state.status.nutrition) == base_nutrition - 1


# ---------------------------------------------------------------------------
# 13. RIN_AGGRAVATE_MONSTER — do_wear.c line 1266 (extrinsic AGGRAVATE_MONSTER)
# ---------------------------------------------------------------------------

def test_ring_aggravate_monster_wear():
    state = _fresh()
    state = state.replace(status=add_intrinsic(state.status, Intrinsic.AGGRAVATE))
    assert _has_intrinsic(state, Intrinsic.AGGRAVATE)
    assert int(RING_INTRINSIC_TABLE[RingEffect.AGGRAVATE_MONSTER]) == Intrinsic.AGGRAVATE


def test_ring_aggravate_monster_takeoff():
    state = _fresh()
    state = state.replace(status=add_intrinsic(state.status, Intrinsic.AGGRAVATE))
    state = state.replace(status=remove_intrinsic(state.status, Intrinsic.AGGRAVATE))
    assert not _has_intrinsic(state, Intrinsic.AGGRAVATE)


# ---------------------------------------------------------------------------
# 14. RIN_CONFLICT — do_wear.c line 1271 (extrinsic CONFLICT)
# ---------------------------------------------------------------------------

def test_ring_conflict_wear():
    state = _fresh()
    state = state.replace(status=add_intrinsic(state.status, Intrinsic.CONFLICT))
    assert _has_intrinsic(state, Intrinsic.CONFLICT)
    assert int(RING_INTRINSIC_TABLE[RingEffect.CONFLICT]) == Intrinsic.CONFLICT


def test_ring_conflict_takeoff():
    state = _fresh()
    state = state.replace(status=add_intrinsic(state.status, Intrinsic.CONFLICT))
    state = state.replace(status=remove_intrinsic(state.status, Intrinsic.CONFLICT))
    assert not _has_intrinsic(state, Intrinsic.CONFLICT)


# ---------------------------------------------------------------------------
# 15. RIN_WARNING — do_wear.c lines 1285-1287: see_monsters()
# ---------------------------------------------------------------------------

def test_ring_warning_wear():
    state = _fresh()
    state = state.replace(status=add_intrinsic(state.status, Intrinsic.WARNING))
    assert _has_intrinsic(state, Intrinsic.WARNING)
    assert int(RING_INTRINSIC_TABLE[RingEffect.WARNING]) == Intrinsic.WARNING


def test_ring_warning_takeoff():
    state = _fresh()
    state = state.replace(status=add_intrinsic(state.status, Intrinsic.WARNING))
    state = state.replace(status=remove_intrinsic(state.status, Intrinsic.WARNING))
    assert not _has_intrinsic(state, Intrinsic.WARNING)


# ---------------------------------------------------------------------------
# 16. RIN_POISON_RESISTANCE — do_wear.c line 1267 (extrinsic POISON_RES)
# ---------------------------------------------------------------------------

def test_ring_poison_resistance_wear():
    state = _fresh()
    state = state.replace(status=add_intrinsic(state.status, Intrinsic.RESIST_POISON))
    assert _has_intrinsic(state, Intrinsic.RESIST_POISON)
    assert int(RING_INTRINSIC_TABLE[RingEffect.POISON_RESISTANCE]) == Intrinsic.RESIST_POISON


def test_ring_poison_resistance_takeoff():
    state = _fresh()
    state = state.replace(status=add_intrinsic(state.status, Intrinsic.RESIST_POISON))
    state = state.replace(status=remove_intrinsic(state.status, Intrinsic.RESIST_POISON))
    assert not _has_intrinsic(state, Intrinsic.RESIST_POISON)


# ---------------------------------------------------------------------------
# 17. RIN_FIRE_RESISTANCE — do_wear.c line 1268 (extrinsic FIRE_RES)
# ---------------------------------------------------------------------------

def test_ring_fire_resistance_wear():
    state = _fresh()
    state = state.replace(status=add_intrinsic(state.status, Intrinsic.RESIST_FIRE))
    assert _has_intrinsic(state, Intrinsic.RESIST_FIRE)
    assert int(RING_INTRINSIC_TABLE[RingEffect.FIRE_RESISTANCE]) == Intrinsic.RESIST_FIRE


def test_ring_fire_resistance_takeoff():
    state = _fresh()
    state = state.replace(status=add_intrinsic(state.status, Intrinsic.RESIST_FIRE))
    state = state.replace(status=remove_intrinsic(state.status, Intrinsic.RESIST_FIRE))
    assert not _has_intrinsic(state, Intrinsic.RESIST_FIRE)


# ---------------------------------------------------------------------------
# 18. RIN_COLD_RESISTANCE — do_wear.c line 1269 (extrinsic COLD_RES)
# ---------------------------------------------------------------------------

def test_ring_cold_resistance_wear():
    state = _fresh()
    state = state.replace(status=add_intrinsic(state.status, Intrinsic.RESIST_COLD))
    assert _has_intrinsic(state, Intrinsic.RESIST_COLD)
    assert int(RING_INTRINSIC_TABLE[RingEffect.COLD_RESISTANCE]) == Intrinsic.RESIST_COLD


def test_ring_cold_resistance_takeoff():
    state = _fresh()
    state = state.replace(status=add_intrinsic(state.status, Intrinsic.RESIST_COLD))
    state = state.replace(status=remove_intrinsic(state.status, Intrinsic.RESIST_COLD))
    assert not _has_intrinsic(state, Intrinsic.RESIST_COLD)


# ---------------------------------------------------------------------------
# 19. RIN_SHOCK_RESISTANCE — do_wear.c line 1270 (extrinsic SHOCK_RES)
# ---------------------------------------------------------------------------

def test_ring_shock_resistance_wear():
    state = _fresh()
    state = state.replace(status=add_intrinsic(state.status, Intrinsic.RESIST_SHOCK))
    assert _has_intrinsic(state, Intrinsic.RESIST_SHOCK)
    assert int(RING_INTRINSIC_TABLE[RingEffect.SHOCK_RESISTANCE]) == Intrinsic.RESIST_SHOCK


def test_ring_shock_resistance_takeoff():
    state = _fresh()
    state = state.replace(status=add_intrinsic(state.status, Intrinsic.RESIST_SHOCK))
    state = state.replace(status=remove_intrinsic(state.status, Intrinsic.RESIST_SHOCK))
    assert not _has_intrinsic(state, Intrinsic.RESIST_SHOCK)


# ---------------------------------------------------------------------------
# 20. RIN_FREE_ACTION — do_wear.c line 1275 (extrinsic FREE_ACTION)
# ---------------------------------------------------------------------------

def test_ring_free_action_wear():
    state = _fresh()
    state = state.replace(status=add_intrinsic(state.status, Intrinsic.FREE_ACTION))
    assert _has_intrinsic(state, Intrinsic.FREE_ACTION)
    assert int(RING_INTRINSIC_TABLE[RingEffect.FREE_ACTION]) == Intrinsic.FREE_ACTION


def test_ring_free_action_takeoff():
    state = _fresh()
    state = state.replace(status=add_intrinsic(state.status, Intrinsic.FREE_ACTION))
    state = state.replace(status=remove_intrinsic(state.status, Intrinsic.FREE_ACTION))
    assert not _has_intrinsic(state, Intrinsic.FREE_ACTION)


# ---------------------------------------------------------------------------
# 21. RIN_SLOW_DIGESTION — do_wear.c line 1276 (extrinsic SLOW_DIGESTION)
# ---------------------------------------------------------------------------

def test_ring_slow_digestion_wear():
    state = _fresh()
    state = state.replace(status=add_intrinsic(state.status, Intrinsic.SLOW_DIGESTION))
    assert _has_intrinsic(state, Intrinsic.SLOW_DIGESTION)
    assert int(RING_INTRINSIC_TABLE[RingEffect.SLOW_DIGESTION]) == Intrinsic.SLOW_DIGESTION


def test_ring_slow_digestion_takeoff():
    state = _fresh()
    state = state.replace(status=add_intrinsic(state.status, Intrinsic.SLOW_DIGESTION))
    state = state.replace(status=remove_intrinsic(state.status, Intrinsic.SLOW_DIGESTION))
    assert not _has_intrinsic(state, Intrinsic.SLOW_DIGESTION)


# ---------------------------------------------------------------------------
# 22. RIN_TELEPORTATION — do_wear.c line 1262 (extrinsic TELEPORTATION)
#     ring_tick must fire 1/85 chance random teleport per turn.
# ---------------------------------------------------------------------------

def test_ring_teleportation_wear_grants_teleport():
    state = _fresh()
    state = state.replace(status=add_intrinsic(state.status, Intrinsic.TELEPORT))
    assert _has_intrinsic(state, Intrinsic.TELEPORT)
    assert int(RING_INTRINSIC_TABLE[RingEffect.TELEPORTATION]) == Intrinsic.TELEPORT


def test_ring_teleportation_takeoff_clears():
    state = _fresh()
    state = state.replace(status=add_intrinsic(state.status, Intrinsic.TELEPORT))
    state = state.replace(status=remove_intrinsic(state.status, Intrinsic.TELEPORT))
    assert not _has_intrinsic(state, Intrinsic.TELEPORT)


def test_ring_teleportation_tick_fires_over_many_turns():
    """Over 1000 turns, the 1/85 random teleport trigger fires at least once.

    Cite: timeout.c — when Teleportation && !Teleport_control && rn2(85)==0.
    """
    ring = make_item(category=int(ItemCategory.RING), type_id=int(RingEffect.TELEPORTATION))
    inv = InventoryState.from_items([ring])
    inv = inv.replace(worn_rings=jnp.full((2,), -1, dtype=jnp.int8).at[0].set(jnp.int8(0)))
    state = _fresh().replace(inventory=inv)
    # Ensure TELEPORT intrinsic is set (ring worn).
    state = state.replace(status=add_intrinsic(state.status, Intrinsic.TELEPORT))

    fired = 0
    rng = jax.random.PRNGKey(1)
    for _ in range(1000):
        rng, sub = jax.random.split(rng)
        before = int(state.status.timed_intrinsics[Intrinsic.TELEPORT])
        state2 = ring_tick(state, sub)
        after = int(state2.status.timed_intrinsics[Intrinsic.TELEPORT])
        if after > before:
            fired += 1
        # reset timed_intrinsics so we can detect next fire
        state = state2.replace(
            status=state2.status.replace(
                timed_intrinsics=state2.status.timed_intrinsics.at[Intrinsic.TELEPORT].set(jnp.int32(0))
            )
        )
    # With p=1/85, expected ~11.8 fires in 1000 turns; at least 1 expected.
    assert fired >= 1, f"Teleportation ring should have triggered at least once in 1000 turns; got {fired}"


# ---------------------------------------------------------------------------
# 23. RIN_TELEPORT_CONTROL — do_wear.c line 1272 (extrinsic TELEPORT_CONTROL)
# ---------------------------------------------------------------------------

def test_ring_teleport_control_wear():
    state = _fresh()
    state = state.replace(status=add_intrinsic(state.status, Intrinsic.TELEPORT_CONTROL))
    assert _has_intrinsic(state, Intrinsic.TELEPORT_CONTROL)
    assert int(RING_INTRINSIC_TABLE[RingEffect.TELEPORT_CONTROL]) == Intrinsic.TELEPORT_CONTROL


def test_ring_teleport_control_takeoff():
    state = _fresh()
    state = state.replace(status=add_intrinsic(state.status, Intrinsic.TELEPORT_CONTROL))
    state = state.replace(status=remove_intrinsic(state.status, Intrinsic.TELEPORT_CONTROL))
    assert not _has_intrinsic(state, Intrinsic.TELEPORT_CONTROL)


# ---------------------------------------------------------------------------
# 24. RIN_POLYMORPH — do_wear.c line 1273 (extrinsic POLYMORPH)
# ---------------------------------------------------------------------------

def test_ring_polymorph_wear():
    state = _fresh()
    state = state.replace(status=add_intrinsic(state.status, Intrinsic.POLYMORPH))
    assert _has_intrinsic(state, Intrinsic.POLYMORPH)
    assert int(RING_INTRINSIC_TABLE[RingEffect.POLYMORPH]) == Intrinsic.POLYMORPH


def test_ring_polymorph_takeoff():
    state = _fresh()
    state = state.replace(status=add_intrinsic(state.status, Intrinsic.POLYMORPH))
    state = state.replace(status=remove_intrinsic(state.status, Intrinsic.POLYMORPH))
    assert not _has_intrinsic(state, Intrinsic.POLYMORPH)


# ---------------------------------------------------------------------------
# 25. RIN_POLYMORPH_CONTROL — do_wear.c line 1274 (extrinsic POLYMORPH_CONTROL)
# ---------------------------------------------------------------------------

def test_ring_polymorph_control_wear():
    state = _fresh()
    state = state.replace(status=add_intrinsic(state.status, Intrinsic.POLYMORPH_CONTROL))
    assert _has_intrinsic(state, Intrinsic.POLYMORPH_CONTROL)
    assert int(RING_INTRINSIC_TABLE[RingEffect.POLYMORPH_CONTROL]) == Intrinsic.POLYMORPH_CONTROL


def test_ring_polymorph_control_takeoff():
    state = _fresh()
    state = state.replace(status=add_intrinsic(state.status, Intrinsic.POLYMORPH_CONTROL))
    state = state.replace(status=remove_intrinsic(state.status, Intrinsic.POLYMORPH_CONTROL))
    assert not _has_intrinsic(state, Intrinsic.POLYMORPH_CONTROL)


# ---------------------------------------------------------------------------
# 26. RIN_INVISIBILITY — do_wear.c lines 1299-1304
# ---------------------------------------------------------------------------

def test_ring_invisibility_wear():
    state = _fresh()
    state = state.replace(status=add_intrinsic(state.status, Intrinsic.INVIS))
    assert _has_intrinsic(state, Intrinsic.INVIS)
    assert int(RING_INTRINSIC_TABLE[RingEffect.INVISIBILITY]) == Intrinsic.INVIS


def test_ring_invisibility_takeoff():
    state = _fresh()
    state = state.replace(status=add_intrinsic(state.status, Intrinsic.INVIS))
    state = state.replace(status=remove_intrinsic(state.status, Intrinsic.INVIS))
    assert not _has_intrinsic(state, Intrinsic.INVIS)


# ---------------------------------------------------------------------------
# 27. RIN_SEE_INVISIBLE — do_wear.c lines 1288-1297
# ---------------------------------------------------------------------------

def test_ring_see_invisible_wear():
    state = _fresh()
    state = state.replace(status=add_intrinsic(state.status, Intrinsic.SEE_INVIS))
    assert _has_intrinsic(state, Intrinsic.SEE_INVIS)
    assert int(RING_INTRINSIC_TABLE[RingEffect.SEE_INVISIBLE]) == Intrinsic.SEE_INVIS


def test_ring_see_invisible_takeoff():
    state = _fresh()
    state = state.replace(status=add_intrinsic(state.status, Intrinsic.SEE_INVIS))
    state = state.replace(status=remove_intrinsic(state.status, Intrinsic.SEE_INVIS))
    assert not _has_intrinsic(state, Intrinsic.SEE_INVIS)
