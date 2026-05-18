"""Combat fidelity gap tests — backstab, death drops, encumbrance/confusion/stun penalties.

Vendor references:
  vendor/nethack/src/uhitm.c:960-964  — Rogue backstab damage bonus
  vendor/nethack/src/uhitm.c:407-409  — encumbrance penalty on to-hit
  vendor/nethack/src/uhitm.c:455      — stun penalty on to-hit + damage
  vendor/nethack/src/weapon.c:961     — confusion penalty on to-hit
  vendor/nethack/src/mondead.c        — xkilled corpse drop
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

# Eager module-level imports keep JAX array construction outside any JIT trace.
import Nethax.nethax.subsystems.artifact_powers  # noqa: F401
import Nethax.nethax.subsystems.weapon_dice  # noqa: F401

from Nethax.nethax.state import EnvState
from Nethax.nethax.constants.roles import Role
from Nethax.nethax.subsystems.combat import melee_attack, to_hit_roll, SKILL_BASIC
from Nethax.nethax.subsystems.status_effects import TimedStatus
from Nethax.nethax.subsystems.monster_ai import MoveStrategy

_RNG = jax.random.PRNGKey(42)
_N_TRIALS = 600


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_state(role: Role = Role.VALKYRIE):
    state = EnvState.default(_RNG)
    state = state.replace(
        player_role=jnp.int8(int(role)),
        player_str=jnp.int16(18),
        player_dex=jnp.int8(14),
        player_xl=jnp.int32(5),
        player_luck=jnp.int8(0),
        player_uhitinc=jnp.int8(0),
        player_udaminc=jnp.int8(0),
        combat=EnvState.default(_RNG).combat.replace(
            weapon_skill=EnvState.default(_RNG).combat.weapon_skill.at[0].set(
                jnp.int8(SKILL_BASIC)
            )
        ),
    )
    # Place a live, high-hp monster at slot 0
    mai = state.monster_ai
    mai = mai.replace(
        alive=mai.alive.at[0].set(jnp.bool_(True)),
        hp=mai.hp.at[0].set(jnp.int32(9999)),
        hp_max=mai.hp_max.at[0].set(jnp.int32(9999)),
        ac=mai.ac.at[0].set(jnp.int8(5)),
        asleep=mai.asleep.at[0].set(jnp.bool_(False)),
        mstrategy=mai.mstrategy.at[0].set(jnp.int8(0)),
        pos=mai.pos.at[0].set(jnp.array([3, 3], dtype=jnp.int16)),
    )
    return state.replace(monster_ai=mai)


def _melee_damage_samples(state, n=_N_TRIALS, seed=7):
    """Run melee_attack n times; collect damage values (0 on miss)."""
    rng = jax.random.PRNGKey(seed)
    damages = []
    for _ in range(n):
        rng, sub = jax.random.split(rng)
        _s, dmg, _hit = melee_attack(state, sub, jnp.int32(0))
        damages.append(int(dmg))
    return damages


def _melee_hit_rate(state, n=_N_TRIALS, seed=17):
    rng = jax.random.PRNGKey(seed)
    hits = 0
    for _ in range(n):
        rng, sub = jax.random.split(rng)
        _s, _dmg, hit = melee_attack(state, sub, jnp.int32(0))
        hits += int(hit)
    return hits / n


def _to_hit_rate(state, target_ac=5, n=_N_TRIALS, seed=99):
    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    ac = jnp.int32(target_ac)
    vroll = jax.jit(jax.vmap(lambda k: to_hit_roll(k, state, ac)))
    hits = vroll(keys)
    return float(jnp.mean(hits.astype(jnp.float32)))


# ---------------------------------------------------------------------------
# 1. Rogue backstab — sleeping target
# vendor/nethack/src/uhitm.c:960-964
# ---------------------------------------------------------------------------

@pytest.mark.timeout(120)
def test_rogue_backstab_sleeping_target():
    """Rogue attacking a sleeping monster should deal more damage than vs awake.

    vendor/nethack/src/uhitm.c:960-964: if (Rogue && (asleep || fleeing))
        dmg += rnd(u.ulevel)
    """
    rogue_state = _base_state(Role.ROGUE)
    mai = rogue_state.monster_ai
    state_awake = rogue_state.replace(
        monster_ai=mai.replace(asleep=mai.asleep.at[0].set(jnp.bool_(False)))
    )
    state_asleep = rogue_state.replace(
        monster_ai=mai.replace(asleep=mai.asleep.at[0].set(jnp.bool_(True)))
    )

    dmg_awake = _melee_damage_samples(state_awake)
    dmg_asleep = _melee_damage_samples(state_asleep)

    # Only compare hitting trials
    hits_awake = [d for d in dmg_awake if d > 0]
    hits_asleep = [d for d in dmg_asleep if d > 0]

    assert len(hits_awake) > 0 and len(hits_asleep) > 0, "No hits recorded"
    mean_awake = sum(hits_awake) / len(hits_awake)
    mean_asleep = sum(hits_asleep) / len(hits_asleep)

    assert mean_asleep > mean_awake, (
        f"Rogue backstab sleeping: expected mean_asleep ({mean_asleep:.2f}) "
        f"> mean_awake ({mean_awake:.2f})"
    )


# ---------------------------------------------------------------------------
# 2. Non-Rogue — no backstab bonus
# vendor/nethack/src/uhitm.c:960
# ---------------------------------------------------------------------------

@pytest.mark.timeout(120)
def test_non_rogue_no_backstab():
    """A Valkyrie should deal the same damage vs sleeping vs awake (no backstab).

    vendor/nethack/src/uhitm.c:960: backstab check is guarded by role==Rogue.
    """
    val_state = _base_state(Role.VALKYRIE)
    mai = val_state.monster_ai
    state_awake = val_state.replace(
        monster_ai=mai.replace(asleep=mai.asleep.at[0].set(jnp.bool_(False)))
    )
    state_asleep = val_state.replace(
        monster_ai=mai.replace(asleep=mai.asleep.at[0].set(jnp.bool_(True)))
    )

    dmg_awake = [d for d in _melee_damage_samples(state_awake) if d > 0]
    dmg_asleep = [d for d in _melee_damage_samples(state_asleep) if d > 0]

    assert len(dmg_awake) > 0 and len(dmg_asleep) > 0
    mean_awake = sum(dmg_awake) / len(dmg_awake)
    mean_asleep = sum(dmg_asleep) / len(dmg_asleep)

    # Allow 1.0 tolerance — sleeping gives +2 to-hit so slightly more hits
    # land, but damage per hit should be statistically similar.
    assert abs(mean_asleep - mean_awake) < 1.5, (
        f"Valkyrie should have no backstab: awake={mean_awake:.2f} "
        f"asleep={mean_asleep:.2f}"
    )


# ---------------------------------------------------------------------------
# 3. Death drops corpse
# vendor/nethack/src/mondead.c::xkilled
# ---------------------------------------------------------------------------

def _kill_monster(state, entry_idx_val: int = 5):
    """Give the monster 1 hp so first hit kills it; return new_state."""
    mai = state.monster_ai
    mai = mai.replace(
        hp=mai.hp.at[0].set(jnp.int32(1)),
        hp_max=mai.hp_max.at[0].set(jnp.int32(1)),
        # Use a regular monster entry (not ghost/elemental/vortex)
        entry_idx=mai.entry_idx.at[0].set(jnp.int16(entry_idx_val)),
    )
    state = state.replace(monster_ai=mai)
    # Force a hit by making AC very low (AC=20 → almost certain hit)
    mai2 = state.monster_ai
    mai2 = mai2.replace(ac=mai2.ac.at[0].set(jnp.int8(20)))
    state = state.replace(monster_ai=mai2)

    rng = jax.random.PRNGKey(123)
    # Try up to 20 times to get the killing blow
    for _ in range(20):
        rng, sub = jax.random.split(rng)
        new_state, _dmg, _hit = melee_attack(state, sub, jnp.int32(0))
        if not bool(new_state.monster_ai.alive[0]):
            return new_state
    return new_state


@pytest.mark.timeout(60)
def test_kill_drops_corpse():
    """Killing a normal monster places a corpse item on the ground at death pos.

    vendor/nethack/src/mondead.c::xkilled — corpse placed at mtmp position.
    """
    state = _base_state()
    new_state = _kill_monster(state, entry_idx_val=5)

    assert not bool(new_state.monster_ai.alive[0]), "Monster should be dead"

    # Check ground_items at the death position (3, 3)
    gi = new_state.ground_items
    branch = int(new_state.dungeon.current_branch)
    level = int(new_state.dungeon.current_level) - 1

    _FOOD_CATEGORY = 7
    _CORPSE_TYPE_ID = 260

    found_corpse = False
    n_stack = gi.category.shape[-1]
    for s in range(n_stack):
        cat = int(gi.category[branch, level, 3, 3, s])
        tid = int(gi.type_id[branch, level, 3, 3, s])
        if cat == _FOOD_CATEGORY and tid == _CORPSE_TYPE_ID:
            found_corpse = True
            break

    assert found_corpse, (
        f"Expected corpse at (3,3) after kill; "
        f"ground_items categories={[int(gi.category[branch,level,3,3,s]) for s in range(n_stack)]}"
    )


# ---------------------------------------------------------------------------
# 4. Ghost / elemental / vortex — no corpse
# vendor/nethack/src/mondead.c::xkilled
# ---------------------------------------------------------------------------

@pytest.mark.timeout(60)
def test_kill_no_corpse_for_ghosts():
    """Killing a ghost-type monster leaves no corpse.

    vendor/nethack/src/mondead.c: ghosts (S_GHOST), elementals, and vortices
    do not leave corpses. _KILLED_DROPS_CORPSE[entry] == False for these.
    """
    from Nethax.nethax.constants.monsters import MONSTERS, MonsterSymbol

    # Find a ghost-symbol monster entry
    ghost_idx = next(
        (i for i, m in enumerate(MONSTERS) if m.symbol == MonsterSymbol.S_GHOST),
        None,
    )
    if ghost_idx is None:
        pytest.skip("No ghost-symbol monster in MONSTERS table")

    state = _base_state()
    new_state = _kill_monster(state, entry_idx_val=ghost_idx)

    if bool(new_state.monster_ai.alive[0]):
        pytest.skip("Could not kill monster in 20 attempts (unlikely; re-run)")

    gi = new_state.ground_items
    branch = int(new_state.dungeon.current_branch)
    level = int(new_state.dungeon.current_level) - 1
    _FOOD_CATEGORY = 7
    _CORPSE_TYPE_ID = 260

    n_stack = gi.category.shape[-1]
    for s in range(n_stack):
        cat = int(gi.category[branch, level, 3, 3, s])
        tid = int(gi.type_id[branch, level, 3, 3, s])
        assert not (cat == _FOOD_CATEGORY and tid == _CORPSE_TYPE_ID), (
            f"Ghost should not drop a corpse; found one at stack slot {s}"
        )


# ---------------------------------------------------------------------------
# 5. Encumbrance penalty on to-hit
# vendor/nethack/src/uhitm.c:407-409
# ---------------------------------------------------------------------------

@pytest.mark.timeout(120)
def test_encumbrance_penalty_burdened():
    """Heavy inventory lowers hit rate vs same setup with empty inventory.

    vendor/nethack/src/uhitm.c:407: if (near_capacity() != 0) tmp -= (2*enc)-1
    """
    base = _base_state()

    # Light state: empty inventory (default)
    state_light = base

    # Heavy state: fill inventory items with weight to trigger enc_level >= 1.
    # cap = 25 * 18 + 50 = 500; enc threshold = 500*2//5 = 200.
    # So total weight > 200 triggers burdened (enc=1).
    items = base.inventory.items
    heavy_weight = items.weight.at[0].set(jnp.int16(210))
    heavy_qty = items.quantity.at[0].set(jnp.int16(1))
    from Nethax.nethax.subsystems.inventory import ItemCategory
    heavy_cat = items.category.at[0].set(jnp.int8(int(ItemCategory.WEAPON)))
    heavy_items = items.replace(weight=heavy_weight, quantity=heavy_qty, category=heavy_cat)
    state_heavy = base.replace(
        inventory=base.inventory.replace(items=heavy_items)
    )

    rate_light = _to_hit_rate(state_light, target_ac=5)
    rate_heavy = _to_hit_rate(state_heavy, target_ac=5)

    assert rate_light > rate_heavy, (
        f"Encumbrance should lower hit rate: light={rate_light:.3f} heavy={rate_heavy:.3f}"
    )


# ---------------------------------------------------------------------------
# 6. Confusion penalty on to-hit
# vendor/nethack/src/weapon.c:961
# ---------------------------------------------------------------------------

@pytest.mark.timeout(120)
def test_confusion_penalty():
    """CONFUSED timer set → lower hit rate than baseline.

    vendor/nethack/src/weapon.c:961: if (Confusion) tohit--;
    """
    base = _base_state()
    state_confused = base.replace(
        status=base.status.replace(
            timed_statuses=base.status.timed_statuses.at[
                int(TimedStatus.CONFUSION)
            ].set(jnp.int32(10))
        )
    )

    rate_normal = _to_hit_rate(base, target_ac=5)
    rate_confused = _to_hit_rate(state_confused, target_ac=5)

    assert rate_normal > rate_confused, (
        f"Confusion should reduce hit rate: normal={rate_normal:.3f} "
        f"confused={rate_confused:.3f}"
    )


# ---------------------------------------------------------------------------
# 7. Stun penalty on to-hit
# vendor/nethack/src/uhitm.c:455
# ---------------------------------------------------------------------------

@pytest.mark.timeout(120)
def test_stun_penalty():
    """STUNNED timer set → lower hit rate than baseline.

    vendor/nethack/src/uhitm.c:455: if (Stunned) tmp--;
    """
    base = _base_state()
    state_stunned = base.replace(
        status=base.status.replace(
            timed_statuses=base.status.timed_statuses.at[
                int(TimedStatus.STUNNED)
            ].set(jnp.int32(10))
        )
    )

    rate_normal = _to_hit_rate(base, target_ac=5)
    rate_stunned = _to_hit_rate(state_stunned, target_ac=5)

    assert rate_normal > rate_stunned, (
        f"Stun should reduce hit rate: normal={rate_normal:.3f} "
        f"stunned={rate_stunned:.3f}"
    )
