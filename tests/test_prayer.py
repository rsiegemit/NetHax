"""Wave 4 tests — prayer subsystem (pray.c port).

All imports are lazy inside test bodies so collection never fails.
"""

import pytest


def _fresh_env_state():
    """Return (env, state, rng) after one reset."""
    import jax
    from Nethax.nethax.env import NethaxEnv

    rng = jax.random.PRNGKey(7)
    env = NethaxEnv()
    state, _ = env.reset(rng)
    return env, state, rng


# ---------------------------------------------------------------------------
# Trouble-fix paths
# ---------------------------------------------------------------------------

def test_pray_when_starving_heals_hunger():
    """pray() on a hungry hero replenishes nutrition (TROUBLE_HUNGRY branch)."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.subsystems.prayer import pray

    _env, state, rng = _fresh_env_state()
    # Force player into WEAK hunger state (hunger_state=3) and low nutrition.
    new_status = state.status.replace(
        hunger_state=jnp.int8(3),
        nutrition=jnp.int32(-80),
    )
    # Clear alignment record (must not trigger anger gate).
    new_prayer = state.prayer.replace(
        alignment_record=jnp.int16(5),
        pray_timeout=jnp.int32(0),
    )
    state = state.replace(status=new_status, prayer=new_prayer)

    new_state = pray(state, rng)

    assert int(new_state.status.nutrition) >= 900, (
        f"Nutrition not restored: got {int(new_state.status.nutrition)}"
    )
    assert int(new_state.status.hunger_state) <= 1


def test_pray_when_low_hp_heals():
    """pray() on a low-HP hero restores HP (TROUBLE_HIT branch)."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.subsystems.prayer import pray

    _env, state, rng = _fresh_env_state()
    state = state.replace(
        player_hp_max=jnp.int32(40),
        player_hp=jnp.int32(3),  # below hp_max/7 = 5
        prayer=state.prayer.replace(
            alignment_record=jnp.int16(5),
            pray_timeout=jnp.int32(0),
        ),
    )

    new_state = pray(state, rng)

    assert int(new_state.player_hp) == 40, (
        f"HP not fully restored: got {int(new_state.player_hp)}"
    )


def test_pray_when_no_trouble_random_outcome():
    """With no trouble and good record, pray() must not crash or drop HP below 0."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.subsystems.prayer import pray

    _env, state, rng = _fresh_env_state()
    state = state.replace(
        player_hp_max=jnp.int32(20),
        player_hp=jnp.int32(20),
        prayer=state.prayer.replace(
            alignment_record=jnp.int16(10),
            pray_timeout=jnp.int32(0),
        ),
    )
    new_state = pray(state, rng)
    assert int(new_state.player_hp) >= 0


def test_pray_increments_timeout():
    """pray() must bump pray_timeout into [300, 1000) (pray.c line 1356)."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.subsystems.prayer import pray, PRAY_TIMEOUT_BASE, PRAY_TIMEOUT_RANGE

    _env, state, rng = _fresh_env_state()
    state = state.replace(
        prayer=state.prayer.replace(
            alignment_record=jnp.int16(5),
            pray_timeout=jnp.int32(0),
        ),
    )
    new_state = pray(state, rng)
    pt = int(new_state.prayer.pray_timeout)
    assert pt >= PRAY_TIMEOUT_BASE
    assert pt < PRAY_TIMEOUT_BASE + PRAY_TIMEOUT_RANGE


def test_pray_when_timeout_active_angers_god():
    """With pray_timeout > 0, alignment_record must drop (-1 angry adjust)."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.subsystems.prayer import pray

    _env, state, rng = _fresh_env_state()
    state = state.replace(
        prayer=state.prayer.replace(
            alignment_record=jnp.int16(10),
            pray_timeout=jnp.int32(500),
        ),
    )
    new_state = pray(state, rng)
    assert int(new_state.prayer.alignment_record) < 10


def test_pray_violates_atheist_conduct():
    """handle_pray() must flag ATHEIST conduct as violated."""
    import jax
    from Nethax.nethax.subsystems.prayer import handle_pray
    from Nethax.nethax.subsystems.conduct import Conduct

    _env, state, rng = _fresh_env_state()
    assert not bool(state.conduct.violations[int(Conduct.ATHEIST)])

    new_state = handle_pray(state, rng)
    assert bool(new_state.conduct.violations[int(Conduct.ATHEIST)])


def test_alignment_record_changes_with_outcome():
    """Pleased prayer (no trouble, positive record) bumps record by +1."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.subsystems.prayer import pray

    _env, state, rng = _fresh_env_state()
    state = state.replace(
        player_hp_max=jnp.int32(20),
        player_hp=jnp.int32(20),
        prayer=state.prayer.replace(
            alignment_record=jnp.int16(7),
            pray_timeout=jnp.int32(0),
        ),
    )
    new_state = pray(state, rng)
    assert int(new_state.prayer.alignment_record) == 8


def test_god_zaps_kills_unresisted_hero():
    """god_zaps_you(state) — two-phase vendor parity (pray.c:610-691).

    Wave-N rebalance: the previous 50/50 d6 / random-inventory-destruction
    coin was a simplified Wave-4 stand-in; vendor pray.c::god_zaps_you is
    a sequential two-phase event:

      Phase 1 (lightning):  fry to crisp unless Shock_resistance / Reflecting.
      Phase 2 (disintegration beam):  disintegrate worn armor; fry unless
                                       Disint_resistance.

    On a fresh hero with no resistances both phases reduce HP to 0, which
    is the vendor outcome — *not* a d6 nibble.
    """
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.subsystems.prayer import god_zaps_you

    _env, state, _rng = _fresh_env_state()
    state = state.replace(
        player_hp_max=jnp.int32(50),
        player_hp=jnp.int32(50),
    )
    rng = jax.random.PRNGKey(0)
    new_state = god_zaps_you(state, rng)
    # Unresisted hero is killed (HP → 0); alignment record drops by 5.
    assert int(new_state.player_hp) == 0
    assert int(new_state.prayer.alignment_record) == int(state.prayer.alignment_record) - 5


def test_god_zaps_spares_disint_resist():
    """Hero with Disintegration resistance survives both phases."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.subsystems.prayer import god_zaps_you
    from Nethax.nethax.subsystems.status_effects import Intrinsic

    _env, state, _rng = _fresh_env_state()
    state = state.replace(
        player_hp_max=jnp.int32(50),
        player_hp=jnp.int32(50),
    )
    intr = state.status.intrinsics
    intr = intr.at[int(Intrinsic.RESIST_SHOCK)].set(True)
    intr = intr.at[int(Intrinsic.RESIST_DISINT)].set(True)
    state = state.replace(status=state.status.replace(intrinsics=intr))
    new_state = god_zaps_you(state, jax.random.PRNGKey(0))
    assert int(new_state.player_hp) == 50, "Resistant hero must survive"


def test_sacrifice_aligned_corpse_increases_record():
    """Sacrificing a same-aligned corpse on an altar bumps alignment_record."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.subsystems.prayer import sacrifice_on_altar
    from Nethax.nethax.subsystems.inventory import ItemCategory

    _env, state, rng = _fresh_env_state()
    # Place an altar of the player's alignment on the player's current tile.
    pos = state.player_pos
    row, col = int(pos[0]), int(pos[1])
    max_levels = state.terrain.shape[1]
    b  = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    flat_lv = b * max_levels + lv
    new_altars = state.features.altar_alignment.at[flat_lv, row, col].set(
        state.player_align.astype(jnp.int8)
    )
    new_feat = state.features.replace(altar_alignment=new_altars)

    # Place a same-aligned (type_id == player_align) FOOD-category corpse.
    items = state.inventory.items
    new_items = items.replace(
        category=items.category.at[0].set(jnp.int8(int(ItemCategory.FOOD))),
        type_id=items.type_id.at[0].set(jnp.int16(int(state.player_align))),
        quantity=items.quantity.at[0].set(jnp.int16(1)),
    )
    state = state.replace(
        features=new_feat,
        inventory=state.inventory.replace(items=new_items),
        prayer=state.prayer.replace(alignment_record=jnp.int16(0)),
    )

    new_state = sacrifice_on_altar(state, rng, jnp.int32(0))
    assert int(new_state.prayer.alignment_record) == 5


def test_sacrifice_unaligned_corpse_decreases_record():
    """Sacrificing an opposite-aligned corpse on the altar drops alignment_record."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.subsystems.prayer import sacrifice_on_altar
    from Nethax.nethax.subsystems.inventory import ItemCategory

    _env, state, rng = _fresh_env_state()
    pos = state.player_pos
    row, col = int(pos[0]), int(pos[1])
    max_levels = state.terrain.shape[1]
    b  = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    flat_lv = b * max_levels + lv
    new_altars = state.features.altar_alignment.at[flat_lv, row, col].set(
        state.player_align.astype(jnp.int8)
    )
    new_feat = state.features.replace(altar_alignment=new_altars)

    # Opposite-aligned corpse: type_id differs from player_align.
    opp = int(state.player_align) + 1
    items = state.inventory.items
    new_items = items.replace(
        category=items.category.at[0].set(jnp.int8(int(ItemCategory.FOOD))),
        type_id=items.type_id.at[0].set(jnp.int16(opp)),
        quantity=items.quantity.at[0].set(jnp.int16(1)),
    )
    state = state.replace(
        features=new_feat,
        inventory=state.inventory.replace(items=new_items),
        prayer=state.prayer.replace(alignment_record=jnp.int16(0)),
    )

    new_state = sacrifice_on_altar(state, rng, jnp.int32(0))
    assert int(new_state.prayer.alignment_record) == -5


def test_altar_buc_sense_reveals_items():
    """Standing on a same-aligned altar reveals BUC of carried items."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.constants import TileType
    from Nethax.nethax.subsystems.features import altar_buc_sense
    from Nethax.nethax.subsystems.inventory import ItemCategory

    if not hasattr(TileType, "ALTAR"):
        pytest.skip("TileType.ALTAR not defined in this build")

    _env, state, _rng = _fresh_env_state()
    pos = state.player_pos
    row, col = int(pos[0]), int(pos[1])
    max_levels = state.terrain.shape[1]
    b  = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    flat_lv = b * max_levels + lv

    new_altars = state.features.altar_alignment.at[flat_lv, row, col].set(
        state.player_align.astype(jnp.int8)
    )
    new_feat = state.features.replace(altar_alignment=new_altars)

    # Plant an unknown-BUC item.
    items = state.inventory.items
    new_items = items.replace(
        category=items.category.at[0].set(jnp.int8(int(ItemCategory.WEAPON))),
        buc_status=items.buc_status.at[0].set(jnp.int8(0)),
        quantity=items.quantity.at[0].set(jnp.int16(1)),
    )
    state = state.replace(
        features=new_feat,
        inventory=state.inventory.replace(items=new_items),
    )

    new_state = altar_buc_sense(state)
    assert int(new_state.inventory.items.buc_status[0]) != 0
    assert bool(new_state.inventory.items.identified[0])


def test_handle_pray_dispatched_through_env_step():
    """Pressing Meta-p through env.step routes to prayer.handle_pray.

    Sanity check: env.step on action=PRAY must (a) not crash, (b) set the
    ATHEIST violation flag.
    """
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.constants.actions import Command
    from Nethax.nethax.subsystems.conduct import Conduct

    env, state, rng = _fresh_env_state()
    # Make prayer safe (no anger, no trouble).
    state = state.replace(
        prayer=state.prayer.replace(
            alignment_record=jnp.int16(5),
            pray_timeout=jnp.int32(0),
        ),
        player_hp_max=jnp.int32(20),
        player_hp=jnp.int32(20),
    )

    rng, step_rng = jax.random.split(rng)
    new_state, _obs, _reward, _done, _info = env.step(
        state, jnp.int32(int(Command.PRAY)), step_rng
    )
    assert bool(new_state.conduct.violations[int(Conduct.ATHEIST)])
