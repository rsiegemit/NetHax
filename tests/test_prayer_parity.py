"""Wave 6 Phase B+ — prayer/sacrifice outcome parity vs vendor pray.c.

These tests cover the full vendor outcome tree implemented in
``Nethax/nethax/subsystems/prayer.py`` (Wave 6).  Each test cites the
matching ``vendor/nethack/src/pray.c`` line range.

All imports are lazy inside test bodies so collection never fails.
"""

import pytest


def _fresh_env_state():
    """Return (env, state, rng) after one reset."""
    import jax
    from Nethax.nethax.env import NethaxEnv

    rng = jax.random.PRNGKey(11)
    env = NethaxEnv()
    state, _ = env.reset(rng)
    return env, state, rng


# ---------------------------------------------------------------------------
# in_trouble() parity — vendor priority chain (pray.c:198-284)
# ---------------------------------------------------------------------------

def test_in_trouble_stoning_first_priority():
    """STONED outranks every other trouble (pray.c:206 — checked first)."""
    import jax.numpy as jnp
    from Nethax.nethax.subsystems.prayer import in_trouble, TROUBLE_STONED
    from Nethax.nethax.subsystems.status_effects import TimedStatus

    _env, state, _rng = _fresh_env_state()
    # Set BOTH stoned (priority 14) AND hungry (priority -8).
    ts = state.status.timed_statuses
    ts = ts.at[int(TimedStatus.STONED)].set(jnp.int32(5))
    new_status = state.status.replace(
        timed_statuses=ts,
        hunger_state=jnp.int8(3),   # WEAK
        nutrition=jnp.int32(-80),
    )
    state = state.replace(status=new_status)

    assert int(in_trouble(state)) == TROUBLE_STONED


def test_in_trouble_food_poisoning_priority():
    """SICK outranks STARVING / HIT / HUNGRY (pray.c:214)."""
    import jax.numpy as jnp
    from Nethax.nethax.subsystems.prayer import in_trouble, TROUBLE_SICK

    _env, state, _rng = _fresh_env_state()
    state = state.replace(
        status=state.status.replace(
            sick_kind=jnp.int8(1),       # food poisoning
            hunger_state=jnp.int8(3),    # also starving (lower prio)
        ),
        player_hp=jnp.int32(1),          # also low HP (lower prio)
        player_hp_max=jnp.int32(20),
    )
    assert int(in_trouble(state)) == TROUBLE_SICK


def test_in_trouble_lava_region():
    """LAVA terrain under the player triggers TROUBLE_LAVA (pray.c:212)."""
    import jax.numpy as jnp
    from Nethax.nethax.subsystems.prayer import in_trouble, TROUBLE_LAVA
    from Nethax.nethax.constants.tiles import TileType

    _env, state, _rng = _fresh_env_state()
    pos = state.player_pos
    row, col = int(pos[0]), int(pos[1])
    b = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    new_terrain = state.terrain.at[b, lv, row, col].set(
        jnp.int8(int(TileType.LAVA))
    )
    state = state.replace(terrain=new_terrain)

    assert int(in_trouble(state)) == TROUBLE_LAVA


def test_trouble_priority_matches_vendor_order():
    """Vendor priority list matches pray.c::in_trouble() exact order.

    Validates VENDOR_TROUBLE_PRIORITY constant against the canonical
    pray.c:206-282 chain.
    """
    from Nethax.nethax.subsystems.prayer import (
        VENDOR_TROUBLE_PRIORITY,
        TROUBLE_STONED, TROUBLE_SLIMED, TROUBLE_STRANGLED, TROUBLE_LAVA,
        TROUBLE_SICK, TROUBLE_STARVING, TROUBLE_REGION, TROUBLE_HIT,
        TROUBLE_HUNGRY,
    )
    # STONED first (pray.c:206).
    assert VENDOR_TROUBLE_PRIORITY[0] == TROUBLE_STONED
    # SLIMED second (pray.c:208).
    assert VENDOR_TROUBLE_PRIORITY[1] == TROUBLE_SLIMED
    # STRANGLED third (pray.c:210).
    assert VENDOR_TROUBLE_PRIORITY[2] == TROUBLE_STRANGLED
    # LAVA fourth (pray.c:212).
    assert VENDOR_TROUBLE_PRIORITY[3] == TROUBLE_LAVA
    # SICK fifth (pray.c:214).
    assert VENDOR_TROUBLE_PRIORITY[4] == TROUBLE_SICK
    # STARVING sixth (pray.c:216).
    assert VENDOR_TROUBLE_PRIORITY[5] == TROUBLE_STARVING
    # REGION before HIT (pray.c:218 before :220).
    assert VENDOR_TROUBLE_PRIORITY.index(TROUBLE_REGION) < \
           VENDOR_TROUBLE_PRIORITY.index(TROUBLE_HIT)
    # HUNGRY appears after HIT (pray.c:275 vs :220).
    assert VENDOR_TROUBLE_PRIORITY.index(TROUBLE_HUNGRY) > \
           VENDOR_TROUBLE_PRIORITY.index(TROUBLE_HIT)


# ---------------------------------------------------------------------------
# Pleased outcome buckets (pray.c::pleased lines 1070-1381)
# ---------------------------------------------------------------------------

def test_pleased_intrinsic_gift_resistance():
    """INTRINSIC_GIVING branch (pray.c:1310-1338) sets a resistance flag."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.subsystems.prayer import _apply_intrinsic_gift

    _env, state, _rng = _fresh_env_state()
    # Sample multiple seeds — at least one should set an intrinsic.
    saw_intrinsic = False
    for seed in range(10):
        rng = jax.random.PRNGKey(seed)
        new_state = _apply_intrinsic_gift(state, rng)
        # Compare to baseline — any new True in intrinsics array means a gift.
        diff = (
            new_state.status.intrinsics.astype(jnp.int32)
            - state.status.intrinsics.astype(jnp.int32)
        )
        if int(jnp.any(diff > 0)):
            saw_intrinsic = True
            break
    assert saw_intrinsic, "intrinsic_gift never granted a resistance"


def test_pleased_alignment_bump():
    """ALIGNMENT_BUMP (pray.c:1089 adjalign(+1)) increments alignment_record."""
    import jax.numpy as jnp
    from Nethax.nethax.subsystems.prayer import _apply_alignment_bump

    _env, state, _rng = _fresh_env_state()
    state = state.replace(
        prayer=state.prayer.replace(alignment_record=jnp.int16(3)),
    )
    new_state = _apply_alignment_bump(state)
    assert int(new_state.prayer.alignment_record) == 4


def test_pleased_remove_curse_all_worn():
    """REMOVE_CURSE (pray.c:1283-1309) uncurses ALL cursed items, not just one."""
    import jax.numpy as jnp
    from Nethax.nethax.subsystems.prayer import _apply_remove_curse

    _env, state, _rng = _fresh_env_state()
    # Plant 5 cursed items in slots 0..4 (buc_status = 1).
    items = state.inventory.items
    new_buc = items.buc_status
    new_cat = items.category
    new_qty = items.quantity
    for i in range(5):
        new_buc = new_buc.at[i].set(jnp.int8(1))
        new_cat = new_cat.at[i].set(jnp.int8(2))   # WEAPON category
        new_qty = new_qty.at[i].set(jnp.int16(1))
    new_items = items.replace(
        buc_status=new_buc, category=new_cat, quantity=new_qty,
    )
    state = state.replace(inventory=state.inventory.replace(items=new_items))

    new_state = _apply_remove_curse(state)
    # Verify ALL 5 cursed slots are now uncursed (buc==2).
    for i in range(5):
        assert int(new_state.inventory.items.buc_status[i]) == 2, (
            f"slot {i} still cursed: {int(new_state.inventory.items.buc_status[i])}"
        )


# ---------------------------------------------------------------------------
# Angry outcome buckets (pray.c::angrygods lines 703-784)
# ---------------------------------------------------------------------------

def test_angry_smite_3d6_damage_range():
    """SMITE_3D6 (pray.c:736-743) deals 3..18 HP damage."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.subsystems.prayer import _apply_smite_3d6

    _env, state, _rng = _fresh_env_state()
    state = state.replace(
        player_hp_max=jnp.int32(50),
        player_hp=jnp.int32(50),
    )
    losses = []
    for seed in range(30):
        rng = jax.random.PRNGKey(seed)
        new_state = _apply_smite_3d6(state, rng)
        loss = int(state.player_hp) - int(new_state.player_hp)
        assert 3 <= loss <= 18, f"3d6 damage out of [3,18]: {loss}"
        losses.append(loss)
    # Variance check: at least 3 distinct values across 30 rolls.
    assert len(set(losses)) >= 3


def test_angry_drain_level_decrements_xl():
    """DRAIN_LEVEL (pray.c:742 losexp) reduces XL by 1."""
    import jax.numpy as jnp
    from Nethax.nethax.subsystems.prayer import _apply_drain_level

    _env, state, _rng = _fresh_env_state()
    state = state.replace(
        player_xl=jnp.int32(7),
        player_hp_max=jnp.int32(40),
        player_hp=jnp.int32(40),
    )
    new_state = _apply_drain_level(state)
    assert int(new_state.player_xl) == 6
    # HP_max also drops.
    assert int(new_state.player_hp_max) < 40
    # Floor at XL=1 even with low input.
    state_low = state.replace(player_xl=jnp.int32(1))
    new_state_low = _apply_drain_level(state_low)
    assert int(new_state_low.player_xl) == 1


def test_angry_summon_demon_bumps_god_anger():
    """SUMMON_DEMON (pray.c:760-772) bumps god_anger and drops alignment."""
    import jax.numpy as jnp
    from Nethax.nethax.subsystems.prayer import _apply_summon_demon

    _env, state, _rng = _fresh_env_state()
    state = state.replace(
        prayer=state.prayer.replace(
            god_anger=jnp.int32(2),
            alignment_record=jnp.int16(0),
        ),
    )
    new_state = _apply_summon_demon(state)
    assert int(new_state.prayer.god_anger) == 3
    assert int(new_state.prayer.alignment_record) == -3


# ---------------------------------------------------------------------------
# Sacrifice outcomes (pray.c::dosacrifice / eval_offering)
# ---------------------------------------------------------------------------

def _place_aligned_altar(state):
    """Helper: place an altar of the player's alignment at the player tile."""
    import jax.numpy as jnp

    pos = state.player_pos
    row, col = int(pos[0]), int(pos[1])
    max_levels = state.terrain.shape[1]
    b = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    flat_lv = b * max_levels + lv
    new_altars = state.features.altar_alignment.at[flat_lv, row, col].set(
        state.player_align.astype(jnp.int8)
    )
    return state.replace(features=state.features.replace(altar_alignment=new_altars))


def test_sacrifice_artifact_gift_at_devout_record():
    """Sacrificing same-aligned corpse at record >= DEVOUT triggers gift.

    Mirrors pray.c:2091-2092 (bestow_artifact()) — player_wis bumps as
    surrogate for the artifact grant slot insertion.
    """
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.subsystems.prayer import sacrifice_on_altar, DEVOUT
    from Nethax.nethax.subsystems.inventory import ItemCategory

    _env, state, rng = _fresh_env_state()
    state = _place_aligned_altar(state)

    # Same-aligned fresh corpse.
    items = state.inventory.items
    new_items = items.replace(
        category=items.category.at[0].set(jnp.int8(int(ItemCategory.FOOD))),
        type_id=items.type_id.at[0].set(jnp.int16(int(state.player_align))),
        quantity=items.quantity.at[0].set(jnp.int16(1)),
    )
    starting_wis = int(state.player_wis)
    state = state.replace(
        inventory=state.inventory.replace(items=new_items),
        prayer=state.prayer.replace(
            alignment_record=jnp.int16(DEVOUT + 1),
        ),
    )

    new_state = sacrifice_on_altar(state, rng, jnp.int32(0))
    assert int(new_state.player_wis) == starting_wis + 1, (
        f"WIS not bumped at DEVOUT gift: {int(new_state.player_wis)}"
    )


def test_sacrifice_mighty_monster_grants_wish():
    """Mighty monster sacrifice (type_id >= 1000) grants a wish credit.

    Mirrors pray.c sacrifice_value() bonus path (lines 1909-1955) — wish
    encoded by decrementing god_anger (Wave-6 stand-in for ``u.uevent.wish``).
    """
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.subsystems.prayer import (
        sacrifice_on_altar, MIGHTY_TYPE_THRESHOLD,
    )
    from Nethax.nethax.subsystems.inventory import ItemCategory

    _env, state, rng = _fresh_env_state()
    state = _place_aligned_altar(state)

    items = state.inventory.items
    new_items = items.replace(
        category=items.category.at[0].set(jnp.int8(int(ItemCategory.FOOD))),
        type_id=items.type_id.at[0].set(
            jnp.int16(MIGHTY_TYPE_THRESHOLD + int(state.player_align))
        ),
        quantity=items.quantity.at[0].set(jnp.int16(1)),
    )
    starting_anger = int(state.prayer.god_anger)
    state = state.replace(
        inventory=state.inventory.replace(items=new_items),
    )

    new_state = sacrifice_on_altar(state, rng, jnp.int32(0))
    # god_anger decrements as wish-credit (Wave-6 model).
    assert int(new_state.prayer.god_anger) == starting_anger - 1
    # And the +10 record bonus applies.
    assert int(new_state.prayer.alignment_record) == int(state.prayer.alignment_record) + 10


# ---------------------------------------------------------------------------
# fix_worst() parity
# ---------------------------------------------------------------------------

def test_fix_worst_stoned_clears_timer():
    """fix_worst(TROUBLE_STONED) zeroes the STONED timer (pray.c:382)."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.subsystems.prayer import fix_worst, TROUBLE_STONED
    from Nethax.nethax.subsystems.status_effects import TimedStatus

    _env, state, _rng = _fresh_env_state()
    ts = state.status.timed_statuses.at[int(TimedStatus.STONED)].set(jnp.int32(4))
    state = state.replace(status=state.status.replace(timed_statuses=ts))

    rng = jax.random.PRNGKey(1)
    new_state = fix_worst(state, rng, jnp.int32(TROUBLE_STONED))
    assert int(new_state.status.timed_statuses[int(TimedStatus.STONED)]) == 0


def test_fix_worst_lava_clears_tile():
    """fix_worst(TROUBLE_LAVA) replaces lava under the player with FLOOR (pray.c:397-403)."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.subsystems.prayer import fix_worst, TROUBLE_LAVA
    from Nethax.nethax.constants.tiles import TileType

    _env, state, _rng = _fresh_env_state()
    pos = state.player_pos
    row, col = int(pos[0]), int(pos[1])
    b = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    state = state.replace(
        terrain=state.terrain.at[b, lv, row, col].set(jnp.int8(int(TileType.LAVA))),
    )
    rng = jax.random.PRNGKey(2)
    new_state = fix_worst(state, rng, jnp.int32(TROUBLE_LAVA))
    # No longer LAVA at the player tile.
    assert int(new_state.terrain[b, lv, row, col]) != int(TileType.LAVA)
