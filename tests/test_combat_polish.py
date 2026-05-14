"""Wave 5 Phase 1 combat-polish tests.

Covers:
  - Per-slot armor AC bonus (worn_armor_ac_bonus cache).
  - Two-weapon combat toggle + dual to-hit per turn.
  - Thrown/ranged combat (item to monster; miss drops projectile on floor).
  - Polymorph combat (form attack dice, form AC overrides armor).

Canonical references:
  vendor/nethack/src/do_wear.c::find_ac      — per-slot armor AC sum.
  vendor/nethack/src/uhitm.c::hitum          — two-weapon attack loop.
  vendor/nethack/src/dothrow.c::throwit      — throw/fire mechanics.
  vendor/nethack/src/polyself.c::find_uac    — polymorphed AC.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp


_RNG = jax.random.PRNGKey(2026)


def _fresh_state():
    from Nethax.nethax.state import EnvState
    return EnvState.default(_RNG)


# ---------------------------------------------------------------------------
# 1. Per-slot armor AC
# ---------------------------------------------------------------------------

def test_helmet_grants_ac_bonus():
    """Setting worn_armor_ac_bonus[HELM] reduces computed player_ac."""
    from Nethax.nethax.subsystems.combat import compute_ac, PLAYER_BASE_AC
    from Nethax.nethax.subsystems.inventory import ArmorSlot

    state = _fresh_state()
    base_ac = int(compute_ac(state))
    assert base_ac == PLAYER_BASE_AC

    new_bonus = state.inventory.worn_armor_ac_bonus.at[int(ArmorSlot.HELM)].set(jnp.int8(2))
    state = state.replace(
        inventory=state.inventory.replace(worn_armor_ac_bonus=new_bonus),
    )
    helmed_ac = int(compute_ac(state))
    assert helmed_ac == base_ac - 2, (
        f"expected AC reduced by helmet bonus 2; got {helmed_ac}"
    )


def test_remove_helmet_restores_ac():
    """Clearing the cached bonus restores base AC."""
    from Nethax.nethax.subsystems.combat import compute_ac, PLAYER_BASE_AC
    from Nethax.nethax.subsystems.inventory import ArmorSlot

    state = _fresh_state()
    new_bonus = state.inventory.worn_armor_ac_bonus.at[int(ArmorSlot.HELM)].set(jnp.int8(2))
    state = state.replace(
        inventory=state.inventory.replace(worn_armor_ac_bonus=new_bonus),
    )
    assert int(compute_ac(state)) == PLAYER_BASE_AC - 2

    cleared = state.inventory.worn_armor_ac_bonus.at[int(ArmorSlot.HELM)].set(jnp.int8(0))
    state = state.replace(
        inventory=state.inventory.replace(worn_armor_ac_bonus=cleared),
    )
    assert int(compute_ac(state)) == PLAYER_BASE_AC


def test_worn_armor_total_ac_from_multiple_slots():
    """Per-slot AC bonuses sum across helmet + body + boots."""
    from Nethax.nethax.subsystems.combat import compute_ac, PLAYER_BASE_AC
    from Nethax.nethax.subsystems.inventory import ArmorSlot

    state = _fresh_state()
    bonus = state.inventory.worn_armor_ac_bonus
    bonus = bonus.at[int(ArmorSlot.HELM)].set(jnp.int8(1))
    bonus = bonus.at[int(ArmorSlot.BODY)].set(jnp.int8(3))
    bonus = bonus.at[int(ArmorSlot.BOOTS)].set(jnp.int8(1))
    state = state.replace(
        inventory=state.inventory.replace(worn_armor_ac_bonus=bonus),
    )
    expected = PLAYER_BASE_AC - (1 + 3 + 1)
    assert int(compute_ac(state)) == expected


# ---------------------------------------------------------------------------
# 2. Two-weapon combat
# ---------------------------------------------------------------------------

def test_two_weapon_toggle_via_command():
    """env.step(Command.TWOWEAPON) flips state.combat.two_weapon."""
    from Nethax.nethax.env import NethaxEnv
    from Nethax.nethax.constants.actions import Command

    env = NethaxEnv()
    rng = jax.random.PRNGKey(7)
    state, _obs = env.reset(rng)
    assert bool(state.combat.two_weapon) is False

    rng, sub = jax.random.split(rng)
    state, _obs, _r, _d, _info = env.step(state, jnp.int32(int(Command.TWOWEAPON)), sub)
    assert bool(state.combat.two_weapon) is True, (
        "TWOWEAPON command should toggle two_weapon flag to True"
    )

    rng, sub = jax.random.split(rng)
    state, _obs, _r, _d, _info = env.step(state, jnp.int32(int(Command.TWOWEAPON)), sub)
    assert bool(state.combat.two_weapon) is False, (
        "TWOWEAPON should toggle back to False"
    )


def test_two_weapon_attacks_twice_per_turn():
    """With two_weapon=True, melee_attack performs two strikes; expected
    damage roughly doubles vs the single-strike baseline (averaged)."""
    from Nethax.nethax.subsystems.combat import melee_attack

    def _setup(two_weapon: bool):
        state = _fresh_state().replace(
            player_pos=jnp.array([5, 5], dtype=jnp.int16),
            player_str=jnp.int16(18 + 100),
            player_dex=jnp.int8(18),
            player_xl=jnp.int32(5),
        )
        mai = state.monster_ai
        mai = mai.replace(
            alive=mai.alive.at[0].set(True),
            hp=mai.hp.at[0].set(jnp.int32(10_000)),
            hp_max=mai.hp_max.at[0].set(jnp.int32(10_000)),
            pos=mai.pos.at[0].set(jnp.array([5, 6], dtype=jnp.int16)),
            ac=mai.ac.at[0].set(jnp.int8(10)),
        )
        state = state.replace(monster_ai=mai)
        state = state.replace(
            combat=state.combat.replace(two_weapon=jnp.bool_(two_weapon)),
        )
        return state

    def _avg_damage(state, n=40, seed=11):
        rng = jax.random.PRNGKey(seed)
        total = 0
        for _ in range(n):
            rng, sub = jax.random.split(rng)
            _, dmg, _ = melee_attack(state, sub, jnp.int32(0))
            total += int(dmg)
        return total / n

    avg_single = _avg_damage(_setup(False), n=40, seed=11)
    avg_double = _avg_damage(_setup(True), n=40, seed=11)

    assert avg_double > avg_single, (
        f"two-weapon avg dmg should exceed single-weapon; "
        f"single={avg_single:.2f}, double={avg_double:.2f}"
    )


# ---------------------------------------------------------------------------
# 3. Ranged / thrown combat
# ---------------------------------------------------------------------------

def _place_thrown_setup(thrown_weight=20, dist=3):
    """Helper: place player + monster ``dist`` tiles east; populate inv slot 0
    with a throwable weapon (quantity=5)."""
    from Nethax.nethax.subsystems.inventory import (
        ItemCategory,
    )
    state = _fresh_state().replace(
        player_pos=jnp.array([5, 5], dtype=jnp.int16),
        player_str=jnp.int16(18 + 100),
        player_dex=jnp.int8(18),
        player_xl=jnp.int32(5),
    )

    # Inv slot 0 = throwable weapon (dart-like).
    items = state.inventory.items
    items = items.replace(
        category=items.category.at[0].set(jnp.int8(ItemCategory.WEAPON)),
        type_id=items.type_id.at[0].set(jnp.int16(8)),
        quantity=items.quantity.at[0].set(jnp.int16(5)),
        weight=items.weight.at[0].set(jnp.int32(thrown_weight)),
        enchantment=items.enchantment.at[0].set(jnp.int8(2)),
    )
    state = state.replace(
        inventory=state.inventory.replace(items=items),
    )

    # Monster ``dist`` tiles east.
    mai = state.monster_ai
    mai = mai.replace(
        alive=mai.alive.at[0].set(True),
        hp=mai.hp.at[0].set(jnp.int32(100)),
        hp_max=mai.hp_max.at[0].set(jnp.int32(100)),
        pos=mai.pos.at[0].set(jnp.array([5, 5 + dist], dtype=jnp.int16)),
        ac=mai.ac.at[0].set(jnp.int8(10)),
    )
    state = state.replace(monster_ai=mai)
    return state


def test_thrown_dart_hits_distant_monster():
    """Throwing a heavy projectile east at a monster 3 tiles away damages it."""
    from Nethax.nethax.subsystems.combat import thrown_attack

    state = _place_thrown_setup(thrown_weight=120, dist=3)
    hp_before = int(state.monster_ai.hp[0])

    # Several throws so RNG variance doesn't dominate.
    rng = jax.random.PRNGKey(33)
    cur = state
    for _ in range(8):
        rng, sub = jax.random.split(rng)
        cur = thrown_attack(
            cur, sub, jnp.int32(0), jnp.array([0, 1], dtype=jnp.int32),
        )

    hp_after = int(cur.monster_ai.hp[0])
    assert hp_after < hp_before, (
        f"thrown attacks should reduce monster HP; before={hp_before}, after={hp_after}"
    )


def test_thrown_projectile_drops_on_floor_if_miss():
    """When the trajectory hits no monster, the projectile lands on the
    ground at the terminal tile."""
    from Nethax.nethax.subsystems.combat import thrown_attack, THROW_MAX_RANGE

    state = _fresh_state().replace(
        player_pos=jnp.array([5, 5], dtype=jnp.int16),
    )

    # Inv slot 0: throwable item, qty 1 (so post-throw the slot empties).
    from Nethax.nethax.subsystems.inventory import ItemCategory
    items = state.inventory.items
    items = items.replace(
        category=items.category.at[0].set(jnp.int8(ItemCategory.WEAPON)),
        type_id=items.type_id.at[0].set(jnp.int16(9)),
        quantity=items.quantity.at[0].set(jnp.int16(1)),
        weight=items.weight.at[0].set(jnp.int32(10)),
    )
    state = state.replace(inventory=state.inventory.replace(items=items))

    # No monster on the path — all alive flags false (default).  Verify.
    assert not bool(jnp.any(state.monster_ai.alive))

    rng = jax.random.PRNGKey(99)
    cur = thrown_attack(state, rng, jnp.int32(0),
                        jnp.array([0, 1], dtype=jnp.int32))

    branch = int(cur.dungeon.current_branch)
    level = int(cur.dungeon.current_level) - 1
    # Terminal tile = start + THROW_MAX_RANGE * (0, 1)
    p_row = int(state.player_pos[0])
    p_col = int(state.player_pos[1])
    end_col = min(p_col + THROW_MAX_RANGE, cur.ground_items.category.shape[3] - 1)

    cat = int(cur.ground_items.category[branch, level, p_row, end_col, 0])
    assert cat == int(ItemCategory.WEAPON), (
        f"expected dropped weapon at ({p_row},{end_col}); got category={cat}"
    )


# ---------------------------------------------------------------------------
# 4. Polymorph combat integration
# ---------------------------------------------------------------------------

def _polymorph_state_with_high_attack():
    """Synthesize a polymorphed state with strong form attacks (e.g. 6d8)."""
    from Nethax.nethax.subsystems.polymorph import PolymorphState, NATTK
    from Nethax.nethax.constants.objects import ObjectClass  # noqa: F401

    state = _fresh_state()
    poly = state.polymorph

    # Build attack arrays: first slot has 6d8 (dragon-like).
    n_dice = jnp.zeros((NATTK,), dtype=jnp.uint8).at[0].set(jnp.uint8(6))
    n_sides = jnp.zeros((NATTK,), dtype=jnp.uint8).at[0].set(jnp.uint8(8))

    poly = poly.replace(
        is_polymorphed=jnp.bool_(True),
        current_form_idx=jnp.int16(0),
        poly_timer=jnp.int16(500),
        attack_n_dice=n_dice,
        attack_n_sides=n_sides,
        orig_ac=jnp.int32(10),
    )
    return state.replace(polymorph=poly)


def test_polymorph_combat_uses_form_attacks():
    """Polymorphed player damage uses form attack dice (6d8) — should
    exceed unarmed (1d4 + STR) baseline by a large margin."""
    from Nethax.nethax.subsystems.combat import melee_attack

    def _setup(polymorphed: bool):
        if polymorphed:
            state = _polymorph_state_with_high_attack()
        else:
            state = _fresh_state()
        state = state.replace(
            player_pos=jnp.array([5, 5], dtype=jnp.int16),
            player_str=jnp.int16(18),
            player_dex=jnp.int8(14),
            player_xl=jnp.int32(3),
        )
        mai = state.monster_ai
        mai = mai.replace(
            alive=mai.alive.at[0].set(True),
            hp=mai.hp.at[0].set(jnp.int32(10_000)),
            hp_max=mai.hp_max.at[0].set(jnp.int32(10_000)),
            pos=mai.pos.at[0].set(jnp.array([5, 6], dtype=jnp.int16)),
            ac=mai.ac.at[0].set(jnp.int8(10)),
        )
        return state.replace(monster_ai=mai)

    def _avg(state, n=40, seed=21):
        rng = jax.random.PRNGKey(seed)
        total = 0
        for _ in range(n):
            rng, sub = jax.random.split(rng)
            _, dmg, _ = melee_attack(state, sub, jnp.int32(0))
            total += int(dmg)
        return total / n

    base = _avg(_setup(False), n=40, seed=21)
    poly = _avg(_setup(True), n=40, seed=21)

    assert poly > base * 1.5, (
        f"polymorph (6d8) avg dmg should clearly exceed unarmed avg; "
        f"base={base:.2f}, poly={poly:.2f}"
    )


def test_polymorph_ac_replaces_armor():
    """When polymorphed, compute_ac returns the form AC (state.player_ac),
    overriding any worn-armor AC bonus contribution."""
    from Nethax.nethax.subsystems.combat import compute_ac
    from Nethax.nethax.subsystems.inventory import ArmorSlot

    state = _fresh_state()
    # Add a hefty armor bonus (would normally take AC to 10 - 5 = 5).
    bonus = state.inventory.worn_armor_ac_bonus.at[int(ArmorSlot.BODY)].set(jnp.int8(5))
    state = state.replace(
        inventory=state.inventory.replace(worn_armor_ac_bonus=bonus),
    )
    ac_armored = int(compute_ac(state))
    assert ac_armored == 5

    # Now polymorph: set state.player_ac to 2 (dragon-like form AC) and flag.
    from Nethax.nethax.subsystems.polymorph import PolymorphState
    poly = state.polymorph.replace(is_polymorphed=jnp.bool_(True))
    state = state.replace(polymorph=poly, player_ac=jnp.int32(2))

    ac_poly = int(compute_ac(state))
    assert ac_poly == 2, (
        f"polymorphed AC should override armor; expected 2, got {ac_poly}"
    )
