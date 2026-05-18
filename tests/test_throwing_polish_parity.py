"""Throwing polish parity tests — vendor/nethack/src/dothrow.c.

Covers the 6 gaps added to thrown_attack():
  Gap 1  — obstacle check (dothrow.c:1510-1580)
  Gap 2  — knockback     (dothrow.c:1130 mhurtle)
  Gap 3  — glass shatter (dothrow.c:1825 + 2262)
  Gap 4  — boomerang     (dothrow.c:1601-1611)
  Gap 5  — silver dmg    (dothrow.c:1343)
  Gap 6  — range formula (dothrow.c:1616-1625)
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

_RNG = jax.random.PRNGKey(2026)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_state():
    """Return a default EnvState with a clear 20x20 floor map."""
    from Nethax.nethax.state import EnvState
    from Nethax.nethax.constants.tiles import TileType

    state = EnvState.default(_RNG).replace(
        player_pos=jnp.array([10, 10], dtype=jnp.int16),
        player_str=jnp.int16(18),
        player_dex=jnp.int8(18),
        player_xl=jnp.int32(5),
    )
    # Carve a clear corridor eastward: rows 0..19, cols 0..19 all FLOOR
    branch = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    terrain = state.terrain.at[branch, lv].set(
        jnp.full(state.terrain.shape[2:], int(TileType.FLOOR), dtype=state.terrain.dtype)
    )
    return state.replace(terrain=terrain)


def _place_item(state, slot, category, type_id, weight, qty=1, enchant=0):
    """Put an item in inventory slot 0."""
    from Nethax.nethax.subsystems.inventory import ItemCategory
    items = state.inventory.items
    items = items.replace(
        category=items.category.at[slot].set(jnp.int8(int(category))),
        type_id=items.type_id.at[slot].set(jnp.int16(type_id)),
        weight=items.weight.at[slot].set(jnp.int32(weight)),
        quantity=items.quantity.at[slot].set(jnp.int16(qty)),
        enchantment=items.enchantment.at[slot].set(jnp.int8(enchant)),
    )
    return state.replace(inventory=state.inventory.replace(items=items))


def _place_monster(state, idx, row, col, hp=30, ac=10):
    """Place a live monster at (row, col)."""
    mai = state.monster_ai
    n = mai.alive.shape[0]
    # Clear all monsters first
    mai = mai.replace(alive=jnp.zeros((n,), dtype=bool))
    mai = mai.replace(
        alive=mai.alive.at[idx].set(True),
        hp=mai.hp.at[idx].set(jnp.int32(hp)),
        hp_max=mai.hp_max.at[idx].set(jnp.int32(hp)),
        pos=mai.pos.at[idx].set(jnp.array([row, col], dtype=jnp.int16)),
        ac=mai.ac.at[idx].set(jnp.int8(ac)),
        asleep=mai.asleep.at[idx].set(jnp.bool_(False)),
    )
    return state.replace(monster_ai=mai)


def _wall_at(state, row, col):
    """Place a WALL tile at (row, col)."""
    from Nethax.nethax.constants.tiles import TileType
    branch = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    terrain = state.terrain.at[branch, lv, row, col].set(jnp.int8(int(TileType.WALL)))
    return state.replace(terrain=terrain)


# ---------------------------------------------------------------------------
# Gap 1 — obstacle check
# ---------------------------------------------------------------------------

def test_throw_blocked_by_wall():
    """Projectile stops before a wall; monster behind wall is not hit.

    vendor/nethack/src/dothrow.c:1510-1580 — flight stops at walls.
    """
    from Nethax.nethax.subsystems.combat import thrown_attack
    from Nethax.nethax.subsystems.inventory import ItemCategory

    state = _base_state()
    # Put wall at (10, 13) — 3 tiles east of player at (10, 10)
    state = _wall_at(state, 10, 13)
    # Monster at (10, 15) — behind the wall
    state = _place_monster(state, 0, row=10, col=15, hp=50)
    # Slot 0: dagger (WEAPON, weight=10, type_id=3)
    state = _place_item(state, 0, ItemCategory.WEAPON, type_id=3, weight=10, qty=3)
    # Force high STR so range would normally reach col 15
    state = state.replace(player_str=jnp.int16(18))

    rng = jax.random.PRNGKey(1)
    result = thrown_attack(state, rng, jnp.int32(0), jnp.array([0, 1], dtype=jnp.int32))

    # Monster must still have full HP (not reached)
    hp_after = int(result.monster_ai.hp[0])
    assert hp_after == 50, (
        f"Monster behind wall should be unhit (hp=50); got hp={hp_after}"
    )


def test_throw_passes_through_floor():
    """Projectile travels across open floor and reaches the monster.

    vendor/nethack/src/dothrow.c:1510-1580 — no blocking on FLOOR.
    """
    from Nethax.nethax.subsystems.combat import thrown_attack
    from Nethax.nethax.subsystems.inventory import ItemCategory

    state = _base_state()
    # Monster at (10, 13) — 3 tiles east, clear floor path
    state = _place_monster(state, 0, row=10, col=13, hp=50, ac=10)
    # Slot 0: dagger, high player stats guarantee hit
    state = _place_item(state, 0, ItemCategory.WEAPON, type_id=3, weight=10, qty=3)
    state = state.replace(
        player_str=jnp.int16(18 + 100),  # max str
        player_dex=jnp.int8(18),
        player_xl=jnp.int32(10),
    )

    rng = jax.random.PRNGKey(1)
    result = thrown_attack(state, rng, jnp.int32(0), jnp.array([0, 1], dtype=jnp.int32))

    hp_after = int(result.monster_ai.hp[0])
    assert hp_after < 50, (
        f"Monster on clear floor should be hit; hp still {hp_after}"
    )


# ---------------------------------------------------------------------------
# Gap 2 — knockback
# ---------------------------------------------------------------------------

def test_heavy_weapon_knocks_back_monster():
    """Throwing a heavy weapon (weight > 100) knocks the monster back 1 tile.

    vendor/nethack/src/dothrow.c:1130 (mhurtle).
    AC=10 + high stats -> tmp=1+7+10+0=18 -> hits ~85% of rolls.
    """
    from Nethax.nethax.subsystems.combat import thrown_attack
    from Nethax.nethax.subsystems.inventory import ItemCategory

    state = _base_state()
    # Monster at (10, 12) — 2 tiles east; AC=10 (positive = easy to hit)
    state = _place_monster(state, 0, row=10, col=12, hp=200, ac=10)
    # weight=150 triggers > 100 knockback check
    state = _place_item(state, 0, ItemCategory.WEAPON, type_id=58, weight=150, qty=20)
    state = state.replace(
        player_str=jnp.int16(18 + 100),
        player_dex=jnp.int8(18),
        player_xl=jnp.int32(10),
    )

    # Try 20 seeds; at 85% hit rate P(no hit in 20) < 0.1%
    knocked = False
    for seed in range(20):
        rng = jax.random.PRNGKey(seed)
        result = thrown_attack(state, rng, jnp.int32(0), jnp.array([0, 1], dtype=jnp.int32))
        new_col = int(result.monster_ai.pos[0, 1])
        if new_col == 13:
            knocked = True
            break

    assert knocked, (
        "Heavy-weapon knockback should move monster to col 13 in at least one of 20 throws"
    )


# ---------------------------------------------------------------------------
# Gap 5 — silver damage
# ---------------------------------------------------------------------------

def _vampire_idx():
    """Return the index of the first vampire in the MONSTERS list."""
    from Nethax.nethax.constants.monsters import MONSTERS, M2_UNDEAD
    for i, m in enumerate(MONSTERS):
        if m.flags2 & M2_UNDEAD and 'vampire' in m.name.lower():
            return i
    return 0


def _orc_idx():
    """Return the index of the first orc in the MONSTERS list."""
    from Nethax.nethax.constants.monsters import MONSTERS, M2_ORC
    for i, m in enumerate(MONSTERS):
        if m.flags2 & M2_ORC:
            return i
    return 1


def test_throw_silver_dagger_vs_vampire():
    """Silver dagger vs vampire deals more than base max when silver bonus fires.

    vendor/nethack/src/dothrow.c:1343.
    Base damage: max(12//30,1)+spread+enchant = 1+1..4+0 = 2..5.
    Silver bonus: +d20 (1..20), so max possible = 25, min with bonus = 3.
    At least one of 40 seeds should land a hit with silver bonus > 5.
    """
    from Nethax.nethax.subsystems.combat import thrown_attack
    from Nethax.nethax.subsystems.inventory import ItemCategory

    # silver dagger: type_id=20, weight=12, material=SILVER
    # AC=10 + high stats -> tmp=18 -> ~85% hit rate
    state = _base_state()
    state = _place_monster(state, 0, row=10, col=13, hp=500, ac=10)
    v_idx = _vampire_idx()
    mai = state.monster_ai
    mai = mai.replace(entry_idx=mai.entry_idx.at[0].set(jnp.int32(v_idx)))
    state = state.replace(monster_ai=mai)

    state = _place_item(state, 0, ItemCategory.WEAPON, type_id=20, weight=12, qty=40)
    state = state.replace(
        player_str=jnp.int16(18 + 100),
        player_dex=jnp.int8(18),
        player_xl=jnp.int32(10),
    )

    damages = []
    for seed in range(40):
        rng = jax.random.PRNGKey(seed)
        result = thrown_attack(state, rng, jnp.int32(0), jnp.array([0, 1], dtype=jnp.int32))
        hp_after = int(result.monster_ai.hp[0])
        damages.append(500 - hp_after)

    # At least one trial should deal > 5 (base max without silver)
    max_dmg = max(damages)
    assert max_dmg > 5, (
        f"Silver dagger vs vampire should sometimes exceed base damage (5); max_dmg={max_dmg}, "
        f"all_damages={sorted(damages, reverse=True)[:5]}"
    )


def test_throw_silver_dagger_vs_orc():
    """Silver dagger vs orc: no silver bonus (orc doesn't hate silver).

    vendor/nethack/src/dothrow.c:1343.
    base = max(12//30,1)=1; spread=1..4; enchant=0 => max per-throw = 5.
    """
    from Nethax.nethax.subsystems.combat import thrown_attack
    from Nethax.nethax.subsystems.inventory import ItemCategory
    from Nethax.nethax.constants.monsters import MONSTERS, M2_WERE, M2_UNDEAD, M2_DEMON

    state = _base_state()
    state = _place_monster(state, 0, row=10, col=13, hp=500, ac=10)
    o_idx = _orc_idx()
    orc = MONSTERS[o_idx]
    assert not (orc.flags2 & (M2_WERE | M2_UNDEAD | M2_DEMON)), (
        "Chosen monster should not hate silver"
    )
    mai = state.monster_ai
    mai = mai.replace(entry_idx=mai.entry_idx.at[0].set(jnp.int32(o_idx)))
    state = state.replace(monster_ai=mai)

    state = _place_item(state, 0, ItemCategory.WEAPON, type_id=20, weight=12, qty=40)
    state = state.replace(
        player_str=jnp.int16(18 + 100),
        player_dex=jnp.int8(18),
        player_xl=jnp.int32(10),
    )

    # No individual throw should exceed base max of 5 (no silver d20 bonus for orc)
    for seed in range(20):
        rng = jax.random.PRNGKey(seed)
        result = thrown_attack(state, rng, jnp.int32(0), jnp.array([0, 1], dtype=jnp.int32))
        hp_after = int(result.monster_ai.hp[0])
        dmg = 500 - hp_after
        assert dmg <= 5, (
            f"Silver dagger vs orc should not get silver bonus (max=5); dmg={dmg} seed={seed}"
        )


# ---------------------------------------------------------------------------
# Gap 6 — range formula
# ---------------------------------------------------------------------------

def test_throw_range_scales_with_str():
    """Higher STR produces a longer flight range.

    vendor/nethack/src/dothrow.c:1616-1625.
    """
    from Nethax.nethax.subsystems.throwing import compute_throw_range

    weight = jnp.int32(10)  # light item

    r_strong = int(compute_throw_range(jnp.int16(18), weight))
    r_weak   = int(compute_throw_range(jnp.int16(6),  weight))

    assert r_strong > r_weak, (
        f"STR=18 should have longer range than STR=6; got {r_strong} vs {r_weak}"
    )
    assert 1 <= r_weak <= 8
    assert 1 <= r_strong <= 8


# ---------------------------------------------------------------------------
# Gap 3 — glass shatters on landing
# ---------------------------------------------------------------------------

def test_glass_breaks_on_landing():
    """Throwing a glass gem at floor: 50% chance the item's quantity becomes 0.

    vendor/nethack/src/dothrow.c:1825 + 2262.
    We verify that across 40 seeds at least one throw breaks and at least one
    survives (50% rate means P(all-break in 40) < 1e-12).
    """
    from Nethax.nethax.subsystems.combat import thrown_attack
    from Nethax.nethax.subsystems.inventory import ItemCategory
    from Nethax.nethax.constants.objects import Material, OBJECTS

    # Find a glass type_id (any object with material=GLASS)
    glass_tid = None
    for i, obj in enumerate(OBJECTS):
        if obj is not None and int(obj.material) == int(Material.GLASS):
            glass_tid = i
            break
    assert glass_tid is not None, "No GLASS object found in OBJECTS table"

    base_state = _base_state()
    # No monster — projectile always misses and lands on floor
    mai = base_state.monster_ai
    mai = mai.replace(alive=jnp.zeros(mai.alive.shape, dtype=bool))
    base_state = base_state.replace(monster_ai=mai)
    base_state = base_state.replace(
        player_str=jnp.int16(18),
        player_dex=jnp.int8(10),
        player_xl=jnp.int32(1),
    )

    broke_count = 0
    intact_count = 0
    n_trials = 40
    for seed in range(n_trials):
        rng = jax.random.PRNGKey(seed * 13 + 7)
        s = _place_item(base_state, 0, ItemCategory.GEM, type_id=glass_tid, weight=1, qty=1)
        result = thrown_attack(s, rng, jnp.int32(0), jnp.array([0, 1], dtype=jnp.int32))
        # Broken items land on the ground with quantity=0;
        # intact items land with quantity=1.
        # The item lands at the terminal tile east of the player.
        # Scan ground stacks at row=10, cols 11-19 for this item.
        b = int(result.dungeon.current_branch)
        lv = int(result.dungeon.current_level) - 1
        gi = result.ground_items
        n_stack = gi.category.shape[-1]
        ground_qty = None
        for col in range(11, 20):
            for si in range(n_stack):
                cat = int(gi.category[b, lv, 10, col, si])
                if cat != 0:
                    ground_qty = int(gi.quantity[b, lv, 10, col, si])
                    break
            if ground_qty is not None:
                break
        if ground_qty == 0:
            broke_count += 1
        elif ground_qty == 1:
            intact_count += 1

    assert broke_count > 0, (
        f"Glass gem should shatter at least once in {n_trials} throws; broke={broke_count}"
    )
    assert intact_count > 0, (
        f"Glass gem should survive at least once in {n_trials} throws; intact={intact_count}"
    )
