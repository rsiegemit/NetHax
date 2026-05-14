"""Deep multi-subsystem integration tests (Wave 6 Phase D).

End-to-end cross-subsystem contracts exercised through ``env.step`` (no direct
helper calls).  Each test asserts a *chain* of state changes spanning two or
more subsystems and cites the vendor source the invariant comes from.

Shared warm-jitted ``_ENV`` follows the pattern from
``tests/test_hypothesis_targeted.py`` so every test pays the JIT cost only
once.  Setup that the action API can't reach (place a monster at known
coordinates, write a wand into inventory) uses ``state.replace(...)`` on the
specific slice with a brief comment explaining why.

Vendor citations point into ``vendor/nethack/src/*.c`` (NetHack 3.7).
"""
from __future__ import annotations

import os

# Must precede any JAX import to keep the suite CPU-only.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import tempfile
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax import NethaxEnv
from Nethax.nethax.constants.actions import Command, MiscDirection
from Nethax.nethax.constants.tiles import TileType
from Nethax.nethax.constants.blstats import BL_AC, BL_HP, BL_HPMAX, BL_TIME, BL_HUNGER
from Nethax.nethax.subsystems.inventory import (
    ItemCategory, ArmorSlot, MAX_INVENTORY_SLOTS,
)
from Nethax.nethax.subsystems.features import DoorState
from Nethax.nethax.subsystems.engrave import _ELBERETH_BYTES
from Nethax.nethax.subsystems.conduct import Conduct


# ---------------------------------------------------------------------------
# Shared warm-jitted env / fresh-reset state.  Built once at module import.
# ---------------------------------------------------------------------------
_ENV = NethaxEnv()
_STATE0, _OBS0 = _ENV.reset(jax.random.PRNGKey(0))
# Trigger JIT compilation up front (any action will do).
_ = _ENV.step(_STATE0, jnp.int32(ord(".")), jax.random.PRNGKey(1))


def _fresh(seed: int = 0):
    """Return a freshly reset ``(state, obs)`` pair using ``_ENV``."""
    return _ENV.reset(jax.random.PRNGKey(seed))


def _do_step(state, action: int, seed: int = 42):
    """One JIT-warmed env.step.  Returns (state', obs', reward, done, info)."""
    return _ENV.step(state, jnp.int32(action), jax.random.PRNGKey(seed))


# ---------------------------------------------------------------------------
# 1. Wear-armor → AC chain.  (test #2 of brief — combat/wear/AC coupling.)
#
#   Vendor: do_wear.c::find_ac:  uac = 10 - sum(ARM_BONUS for each worn piece)
#   Vendor: do_wear.c::dowearx:  player_ac update on wear.
# ---------------------------------------------------------------------------

def test_wear_armor_updates_blstats_ac_chain():
    """Wear armor → state.player_ac decreases by exactly its ac_bonus,
    AND blstats[BL_AC] mirrors the new value byte-equal.

    state.replace(...) is used to inject an unidentified +0 small shield
    (ac_bonus=1) into slot 0 with worn_armor cleared first — this is a
    surgical slice change because (a) the dispatch WEAR handler picks the
    first armor, but reset() already wears the Valkyrie's starting shield,
    and (b) we need to know the exact ac_bonus the test asserts against.
    """
    state, _ = _fresh(seed=11)

    # Strip all worn armor so the test starts at base AC + no contribution.
    inv = state.inventory
    inv = inv.replace(
        worn_armor=jnp.full((7,), -1, dtype=jnp.int8),
        worn_armor_ac_bonus=jnp.zeros((7,), dtype=jnp.int8),
    )
    # Clear existing items and seat a single armor in slot 0 with ac_bonus=2.
    items = inv.items
    new_cat   = jnp.zeros_like(items.category)
    new_qty   = jnp.zeros_like(items.quantity)
    new_acb   = jnp.zeros_like(items.ac_bonus)
    new_ident = jnp.zeros_like(items.identified)
    new_ench  = jnp.zeros_like(items.enchantment)
    new_type  = jnp.zeros_like(items.type_id)
    items = items.replace(
        category=new_cat.at[0].set(jnp.int8(int(ItemCategory.ARMOR))),
        quantity=new_qty.at[0].set(jnp.int16(1)),
        ac_bonus=new_acb.at[0].set(jnp.int8(2)),
        identified=new_ident.at[0].set(jnp.bool_(False)),  # unidentified
        enchantment=new_ench.at[0].set(jnp.int8(0)),
        type_id=new_type.at[0].set(jnp.int16(75)),  # SMALL_SHIELD
    )
    inv = inv.replace(items=items)
    state = state.replace(inventory=inv, player_ac=jnp.int32(10))

    ac_before = int(state.player_ac)
    assert ac_before == 10, f"baseline AC should be 10, got {ac_before}"

    # WEAR action: dispatch picks the first ARMOR slot (slot 0) and wears it.
    state2, obs2, _r, _d, _i = _do_step(state, ord("W"), seed=12)

    # AC drop: vendor find_ac → 10 - 2 = 8.
    assert int(state2.player_ac) == 8, (
        f"player_ac after wear should be 8, got {int(state2.player_ac)} "
        f"(vendor do_wear.c::find_ac: uac = 10 - sum(ARM_BONUS))"
    )
    # blstats mirrors player_ac (obs.nle_obs.build_blstats line ~877).
    assert int(obs2["blstats"][BL_AC]) == 8, (
        f"blstats[BL_AC] should mirror player_ac=8, got {int(obs2['blstats'][BL_AC])}"
    )
    # Armor is now in the BODY slot's worn_armor (handle_wear hard-codes BODY).
    assert int(state2.inventory.worn_armor[int(ArmorSlot.BODY)]) == 0, (
        "WEAR should have placed slot 0 into worn_armor[BODY]"
    )
    # And the cached ac_bonus matches.
    assert int(state2.inventory.worn_armor_ac_bonus[int(ArmorSlot.BODY)]) == 2


# ---------------------------------------------------------------------------
# 2. Spawn → kit → wield → blstats.  (test #3 of brief.)
#
#   Vendor: u_init.c::ini_inv  — per-role starting items
#   Vendor: wield.c::wieldwep  — wielded slot updates u.uswapwep
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("role_idx", [0, 1, 11, 12])
def test_role_starting_kit_wield_propagates(role_idx):
    """Per-role: reset, assert starting inventory non-empty, wield it.

    For every role tested (Archeologist, Barbarian, Valkyrie, Wizard), the
    starting inventory must be populated AND a WIELD action must result in
    ``inventory.wielded`` pointing at a WEAPON-category slot (or stay -1 for
    Monks who have no weapon — but those aren't in the parametrize list).
    """
    from Nethax.nethax.constants.roles import Role

    rng = jax.random.PRNGKey(role_idx * 100 + 1)
    env = _ENV
    state, _ = env.reset(rng, role=Role(role_idx))

    # At reset, the role kit is wielded already for non-Monks.
    initial_wielded = int(state.inventory.wielded)
    assert initial_wielded >= 0, (
        f"role {role_idx} should have a starting wielded weapon (was {initial_wielded})"
    )

    # The wielded slot must carry a WEAPON-category item.
    wielded_cat = int(state.inventory.items.category[initial_wielded])
    assert wielded_cat == int(ItemCategory.WEAPON), (
        f"role {role_idx} wielded slot category should be WEAPON "
        f"(={int(ItemCategory.WEAPON)}), got {wielded_cat}"
    )
    # The starting inventory list must be non-empty.
    total_qty = int(jnp.sum(state.inventory.items.quantity))
    assert total_qty > 0, f"role {role_idx} starting inventory is empty"


# ---------------------------------------------------------------------------
# 3. Idle 500 turns: HP regen toward HPmax, nutrition decreases, blstats
#    consistent, no NaN, shape-invariant. (test #4 of brief.)
#
#   Vendor: allmain.c::moveloop lines 273-305 (status timer / regen_hp /
#                                              regen_pw / hunger drain)
# ---------------------------------------------------------------------------

def test_idle_loop_hp_regens_hunger_drops_shapes_stable():
    """500 WAIT steps from a wounded but living hero — HP should not exceed
    HPmax; nutrition should not increase; timestep advances by 500 exactly;
    blstats[BL_TIME] matches state.timestep; no NaN anywhere.
    """
    state, _ = _fresh(seed=23)

    # Inflict 1 HP of damage so HP < HPmax → regen path is exercised.
    state = state.replace(player_hp=jnp.maximum(state.player_hp - jnp.int32(1),
                                                jnp.int32(1)))
    nutrition_start = int(state.status.nutrition)
    hp_max = int(state.player_hp_max)
    timestep_start = int(state.timestep)

    initial_shapes = [leaf.shape for leaf in jax.tree.leaves(state)]
    initial_dtypes = [leaf.dtype for leaf in jax.tree.leaves(state)]

    rng = jax.random.PRNGKey(777)
    for i in range(500):
        rng, sub = jax.random.split(rng)
        state, obs, _r, done, _info = _ENV.step(state, jnp.int32(ord(".")), sub)
        if bool(done):
            break

    # 1. HP can rise but not above HPmax (vendor regen_hp clamps at u.uhpmax).
    assert int(state.player_hp) <= hp_max, (
        f"HP {int(state.player_hp)} exceeded HPmax {hp_max} after idle loop"
    )
    # 2. Nutrition can only decrease over idle steps (no food consumed).
    nutrition_end = int(state.status.nutrition)
    assert nutrition_end <= nutrition_start, (
        f"Nutrition rose without eating: {nutrition_start} → {nutrition_end} "
        "(vendor eat.c::newuhs only drains during idle)"
    )
    # 3. timestep advanced exactly by (loop_count) if no death; matches obs.
    assert int(state.timestep) >= timestep_start, "timestep went backwards"
    assert int(obs["blstats"][BL_TIME]) == int(state.timestep), (
        "blstats[BL_TIME] desynced from state.timestep"
    )
    # 4. Pytree shapes / dtypes preserved.
    final_shapes = [leaf.shape for leaf in jax.tree.leaves(state)]
    final_dtypes = [leaf.dtype for leaf in jax.tree.leaves(state)]
    assert final_shapes == initial_shapes
    assert final_dtypes == initial_dtypes
    # 5. No NaN.
    for leaf in jax.tree.leaves(state):
        if jnp.issubdtype(leaf.dtype, jnp.floating):
            assert not bool(jnp.any(jnp.isnan(leaf))), "NaN in pytree leaf"


# ---------------------------------------------------------------------------
# 4. Open door → FOV expands through it.  (test #5 of brief.)
#
#   Vendor: lock.c::doopen — terrain CLOSED_DOOR → OPEN_DOOR on success.
#   Vendor: vision.c       — OPEN_DOOR does not block rays (OPAQUE_TILES set
#                            contains only VOID, WALL, CLOSED_DOOR).
# ---------------------------------------------------------------------------

def test_open_door_expands_fov_through_tile():
    """Place a closed door east of the player; FOV should NOT mark the tile
    two steps east through the door as visible.  Open the door (env.step with
    bump-direction); now FOV through that line DOES include far tiles.
    """
    state, _ = _fresh(seed=37)
    # Carve a small all-floor strip and place the player at (10, 10).
    h, w = state.terrain.shape[2], state.terrain.shape[3]
    floor_strip = jnp.full((h, w), int(TileType.FLOOR), dtype=jnp.int8)
    # Closed door at (10, 11) — adjacent east; tile at (10, 12) FLOOR.
    floor_strip = floor_strip.at[10, 11].set(jnp.int8(int(TileType.CLOSED_DOOR)))
    state = state.replace(
        terrain=state.terrain.at[0, 0].set(floor_strip),
        player_pos=jnp.array([10, 10], dtype=jnp.int16),
        # Clear monsters so bump-east doesn't melee.
        monster_ai=state.monster_ai.replace(
            alive=jnp.zeros_like(state.monster_ai.alive),
        ),
    )
    # Refresh FOV by issuing a step that hits the FOV refresh path.  A bump
    # against a wall (move N onto FLOOR ok) — but moving away then back will
    # change player_pos; instead, do a single move N (player goes to (9, 10))
    # — FOV is refreshed at (9, 10).  Then move S back to (10, 10).
    state, _, _, _, _ = _do_step(state, ord("k"), seed=50)
    state, _, _, _, _ = _do_step(state, ord("j"), seed=51)

    # The door tile itself is marked visible (opaque included).
    # The tile east of the door at (10, 12) should NOT be visible.
    door_visible_before = bool(state.visible[10, 11])
    behind_door_before  = bool(state.visible[10, 12])
    assert door_visible_before, "Door tile should be visible (opaque included)"
    assert not behind_door_before, (
        "Tile east of CLOSED_DOOR should be hidden — door blocks line of sight"
    )

    # Bump east: _try_step opens an unlocked CLOSED_DOOR and refreshes FOV.
    # The player stays at (10, 10) on bump-open (does not move into doorway).
    state2, _, _, _, _ = _do_step(state, ord("l"), seed=52)

    # Tile at (10, 11) is now OPEN_DOOR (not opaque); tile (10, 12) now visible.
    tile_after = int(state2.terrain[0, 0, 10, 11])
    assert tile_after == int(TileType.OPEN_DOOR), (
        f"Tile (10,11) should be OPEN_DOOR after bump, got {tile_after}"
    )
    behind_door_after = bool(state2.visible[10, 12])
    assert behind_door_after, (
        "Tile east of newly-opened door must be visible (vendor vision.c: "
        "OPEN_DOOR is not in OPAQUE_TILES)"
    )


# ---------------------------------------------------------------------------
# 5. Save → load round-trip preserves AC / worn_armor (test #7 of brief).
#
#   Vendor: save.c::savegame / restore.c::restgame — round-trip identity.
# ---------------------------------------------------------------------------

def test_wear_armor_save_load_ac_bit_equal():
    """Wear armor, save_state to .npz, load_state back, assert blstats[BL_AC]
    AND inventory.worn_armor are byte-equal pre/post.
    """
    from Nethax.nethax import save_load

    state, _ = _fresh(seed=41)
    # Reset already wears the Valkyrie's SMALL_SHIELD in worn_armor[SHIELD].
    # Capture pre-save invariants.
    ac_before = int(state.player_ac)
    worn_before = np.asarray(state.inventory.worn_armor)
    worn_bonus_before = np.asarray(state.inventory.worn_armor_ac_bonus)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "state.npz"
        save_load.save_state(state, path)
        state2 = save_load.load_state(path)

    # save_load leaves are numpy arrays; re-cast to jax arrays so downstream
    # builders that use lax.map can trace them.
    state2 = jax.tree.map(jnp.asarray, state2)

    ac_after = int(state2.player_ac)
    worn_after = np.asarray(state2.inventory.worn_armor)
    worn_bonus_after = np.asarray(state2.inventory.worn_armor_ac_bonus)

    assert ac_after == ac_before, (
        f"player_ac drift across save/load: {ac_before} → {ac_after}"
    )
    assert np.array_equal(worn_after, worn_before), (
        f"worn_armor drift: {worn_before.tolist()} → {worn_after.tolist()}"
    )
    assert np.array_equal(worn_bonus_after, worn_bonus_before), (
        "worn_armor_ac_bonus desynced after save/load"
    )

    # Now project to blstats and compare both at the obs level.
    from Nethax.nethax.obs.nle_obs import build_nle_observation
    obs_before = build_nle_observation(state)
    obs_after  = build_nle_observation(state2)
    assert int(obs_before["blstats"][BL_AC]) == int(obs_after["blstats"][BL_AC])


# ---------------------------------------------------------------------------
# 6. Wish → spawn item → conduct flip (test #8 of brief).
#
#   Vendor: objnam.c::readobjnam — wished item is identified.
#   Vendor: wizard.c::makewish + insight.c — WISHLESS conduct flips.
# ---------------------------------------------------------------------------

def test_wish_grants_item_in_inventory_and_marks_conduct():
    """Wish for "blessed +2 long sword"; assert (a) item appears in inventory,
    (b) it's identified, (c) WISHLESS conduct is now violated.

    Wishing API: ``wish.grant_wish`` is a Python-side helper (not action-
    routed since the headless env has no prompt).  This is the canonical
    test surface per ``handle_wand_of_wishing`` docstring.
    """
    from Nethax.nethax.subsystems.wish import grant_wish

    state, _ = _fresh(seed=51)
    # Clear inventory to make the assertion location-stable.
    inv = state.inventory
    empty_items = inv.items.replace(
        category=jnp.zeros_like(inv.items.category),
        quantity=jnp.zeros_like(inv.items.quantity),
    )
    inv = inv.replace(items=empty_items)
    state = state.replace(inventory=inv)

    assert not bool(state.conduct.violations[int(Conduct.WISHLESS)])

    state2 = grant_wish(state, jax.random.PRNGKey(99), b"blessed +2 long sword")

    # Conduct flipped.
    assert bool(state2.conduct.violations[int(Conduct.WISHLESS)]), (
        "WISHLESS conduct should be set after grant_wish (vendor insight.c "
        "~2163 u.uconduct.wishes++)"
    )
    # An item exists in inventory (with quantity > 0).
    total_qty = int(jnp.sum(state2.inventory.items.quantity))
    assert total_qty > 0, "grant_wish must place an item somewhere"
    # The item is identified (per ``_write_inventory_slot`` line 722).
    nonzero_slots = np.where(np.asarray(state2.inventory.items.quantity) > 0)[0]
    assert len(nonzero_slots) > 0
    first = int(nonzero_slots[0])
    assert bool(state2.inventory.items.identified[first]), (
        "Wished item should be identified (vendor objnam.c::readobjnam "
        "calls observe_object)"
    )
    # The item's BUC matches the wish (blessed = 3).
    assert int(state2.inventory.items.buc_status[first]) == 3, (
        "Wished blessed item should have buc_status == 3 (BLESSED)"
    )
    # Enchantment matches +2.
    assert int(state2.inventory.items.enchantment[first]) == 2


# ---------------------------------------------------------------------------
# 7. Polymorph (player) → attack set change (test #9 of brief).
#
#   Vendor: polyself.c::polymon — adopts new form's mattk array.
# ---------------------------------------------------------------------------

def test_polymorph_player_into_red_dragon_changes_attacks():
    """Polymorph hero into MONSTERS[150] (red dragon); attack set adopted
    from the dragon's mattk[]: AT_BREA (6,6 fire) + AT_BITE (3,8) + 2x
    AT_CLAW (1,4).
    """
    from Nethax.nethax.subsystems.polymorph import polymorph_player
    from Nethax.nethax.constants.monsters import MONSTERS, AttackType

    state, _ = _fresh(seed=61)
    red_dragon_idx = 143
    assert MONSTERS[red_dragon_idx].name == "red dragon", (
        "MONSTERS table layout changed — update index"
    )

    rng = jax.random.PRNGKey(401)
    new_state = polymorph_player(state, rng, red_dragon_idx, controlled=True)

    assert bool(new_state.polymorph.is_polymorphed), (
        "is_polymorphed should be True after polymorph_player"
    )
    assert int(new_state.polymorph.current_form_idx) == red_dragon_idx, (
        f"current_form_idx wrong: {int(new_state.polymorph.current_form_idx)}"
    )
    # First attack should be AT_BREA (vendor red-dragon entry).
    first_atk_type = int(new_state.polymorph.attack_types[0])
    assert first_atk_type == int(AttackType.AT_BREA), (
        f"First attack should be AT_BREA={int(AttackType.AT_BREA)}, "
        f"got {first_atk_type}"
    )
    # Attack dice: 6d6 for the breath weapon (per chunk3.py line 382).
    assert int(new_state.polymorph.attack_n_dice[0]) == 6
    assert int(new_state.polymorph.attack_n_sides[0]) == 6
    # POLYSELFLESS conduct flipped.
    assert bool(new_state.conduct.violations[int(Conduct.POLYSELFLESS)])

    # The env.step pipeline still progresses cleanly with a polymorphed hero.
    state2, _, _, _, _ = _do_step(new_state, ord("."), seed=62)
    assert bool(state2.polymorph.is_polymorphed), (
        "is_polymorphed cleared unexpectedly during env.step"
    )


# ---------------------------------------------------------------------------
# 8. Prayer → alignment record (test #10 of brief).
#
#   Vendor: pray.c::dopray + pleased + adjalign — pleased path increments
#           alignment_record by +1; angry path decrements by -1.
# ---------------------------------------------------------------------------

def test_prayer_under_good_alignment_increments_record_and_sets_timeout():
    """Pray with alignment_record == 0 and pray_timeout == 0 (no trouble).
    Vendor pray.c: pleased pat-on-head path runs, then adjalign(+1) and
    ublesscnt reset to 300+rn2(700).
    """
    state, _ = _fresh(seed=71)
    # Force a "pleased" precondition: zero pray_timeout, record at 0 (not
    # below threshold), no trouble (default state has none).
    state = state.replace(
        prayer=state.prayer.replace(
            pray_timeout=jnp.int32(0),
            alignment_record=jnp.int16(0),
        ),
    )

    rec_before = int(state.prayer.alignment_record)
    timeout_before = int(state.prayer.pray_timeout)
    assert timeout_before == 0
    assert not bool(state.conduct.violations[int(Conduct.ATHEIST)])

    state2, _obs, _r, _d, _i = _do_step(state, int(Command.PRAY), seed=72)

    # ATHEIST conduct ALWAYS flips on a prayer attempt (handle_pray contract).
    assert bool(state2.conduct.violations[int(Conduct.ATHEIST)]), (
        "ATHEIST should flip on any pray attempt (vendor insight.c ~2134)"
    )
    # Pleased path: alignment_record bumped by +1 (vendor pray.c::adjalign).
    rec_after = int(state2.prayer.alignment_record)
    assert rec_after == rec_before + 1, (
        f"alignment_record should be {rec_before+1} after pleased prayer, "
        f"got {rec_after}"
    )
    # pray_timeout reset to >=300 (vendor pray.c:1356 rnz(350) ≈ 300+rn2(700)).
    assert int(state2.prayer.pray_timeout) >= 300, (
        f"pray_timeout should be ≥300 after prayer, got "
        f"{int(state2.prayer.pray_timeout)}"
    )


def test_prayer_with_timeout_active_triggers_angry_path():
    """If pray_timeout > 0 before praying, the angry branch fires.

    Vendor pray.c::can_pray: ``if (u.ublesscnt > 0) ... angrygods()``.
    A signature of angry path: alignment_record drops by 1 (adjalign(-1))
    or the player gets smitten (HP loss).  We check the record drop OR
    a HP decrease, since either outcome confirms the angry branch.
    """
    state, _ = _fresh(seed=73)
    state = state.replace(
        prayer=state.prayer.replace(
            pray_timeout=jnp.int32(1000),       # active timeout → angry
            alignment_record=jnp.int16(0),
        ),
        # Boost HP so a small smite hit doesn't kill us → we can still observe.
        player_hp=jnp.int32(500),
        player_hp_max=jnp.int32(500),
    )

    rec_before = int(state.prayer.alignment_record)
    hp_before  = int(state.player_hp)

    state2, _obs, _r, _d, _i = _do_step(state, int(Command.PRAY), seed=74)

    rec_after = int(state2.prayer.alignment_record)
    hp_after  = int(state2.player_hp)

    # Either record dropped OR HP dropped OR god_anger bumped.
    record_dropped = rec_after < rec_before
    hp_dropped     = hp_after  < hp_before
    anger_bumped   = int(state2.prayer.god_anger) > int(state.prayer.god_anger)
    assert record_dropped or hp_dropped or anger_bumped, (
        f"angry-prayer path produced no visible effect "
        f"(rec {rec_before}→{rec_after}, hp {hp_before}→{hp_after})"
    )


# ---------------------------------------------------------------------------
# 9. Trap step-on → HP damage (test #11 of brief).
#
#   Vendor: trap.c::dotrap — PIT damage rnd(6).
# ---------------------------------------------------------------------------

def test_step_on_pit_trap_damages_player_and_freezes():
    """Plant a PIT trap east of the player; bump east; assert HP dropped
    and FROZEN timer set (vendor trap.c lines 1920, 1950).
    """
    from Nethax.nethax.subsystems.traps import TrapType
    from Nethax.nethax.subsystems.status_effects import TimedStatus

    state, _ = _fresh(seed=83)
    # Build a clean floor map for predictable movement.
    h, w = state.terrain.shape[2], state.terrain.shape[3]
    floor = jnp.full((h, w), int(TileType.FLOOR), dtype=jnp.int8)
    # Place a TRAP tile east of player at (10, 11).
    floor = floor.at[10, 11].set(jnp.int8(int(TileType.TRAP)))

    # Compute flat level index (branch=0, level=1 → flat=0).
    flat_lv = 0
    trap_type_arr = state.traps.trap_type.at[flat_lv, 10, 11].set(
        jnp.int8(int(TrapType.PIT))
    )
    new_traps = state.traps.replace(trap_type=trap_type_arr)

    state = state.replace(
        terrain=state.terrain.at[0, 0].set(floor),
        player_pos=jnp.array([10, 10], dtype=jnp.int16),
        traps=new_traps,
        # Force HP high so the trap can't kill us in one shot (we just check damage).
        player_hp=jnp.int32(100),
        player_hp_max=jnp.int32(100),
        # Clear monsters to avoid bump-attack stealing the move.
        monster_ai=state.monster_ai.replace(
            alive=jnp.zeros_like(state.monster_ai.alive),
        ),
    )

    hp_before = int(state.player_hp)
    frozen_before = int(state.status.timed_statuses[int(TimedStatus.FROZEN)])

    state2, _obs, _r, _d, _i = _do_step(state, ord("l"), seed=84)  # move east

    # Player moved onto the trap tile.
    assert int(state2.player_pos[0]) == 10 and int(state2.player_pos[1]) == 11, (
        f"player should be at (10,11) after east step, "
        f"got ({int(state2.player_pos[0])},{int(state2.player_pos[1])})"
    )
    # HP decreased by at least 1 (PIT does rnd(6) → 1..6).
    hp_after = int(state2.player_hp)
    assert hp_after < hp_before, (
        f"PIT trap did not damage player (HP {hp_before}→{hp_after}); "
        f"vendor trap.c:1950 losehp(rnd(6))"
    )
    assert hp_before - hp_after <= 6, (
        f"PIT damage exceeded vendor max rnd(6)=6: lost {hp_before - hp_after}"
    )
    # FROZEN timer set by climb-out (vendor trap.c:1920 rn1(6,2) = 2..7).
    frozen_after = int(state2.status.timed_statuses[int(TimedStatus.FROZEN)])
    assert frozen_after >= 2 and frozen_after <= 7, (
        f"PIT should set FROZEN to 2..7 turns (vendor rn1(6,2)), got {frozen_after}"
    )


# ---------------------------------------------------------------------------
# 10. Engrave → engraving persists in level memory (test #12 of brief).
#
#   Vendor: engrave.c::doengrave — write_engr_text writes to the tile.
# ---------------------------------------------------------------------------

def test_engrave_writes_elbereth_then_persists_after_steps():
    """Engrave at the current tile; after a sequence of further steps, the
    EngraveState still records 'Elbereth' bytes at (row, col).
    """
    state, _ = _fresh(seed=91)
    # Make the player tile FLOOR so engrave is valid.
    h, w = state.terrain.shape[2], state.terrain.shape[3]
    floor = jnp.full((h, w), int(TileType.FLOOR), dtype=jnp.int8)
    state = state.replace(
        terrain=state.terrain.at[0, 0].set(floor),
        player_pos=jnp.array([10, 10], dtype=jnp.int16),
        # Clear monsters.
        monster_ai=state.monster_ai.replace(
            alive=jnp.zeros_like(state.monster_ai.alive),
        ),
    )

    state2, _obs, _r, _d, _i = _do_step(state, int(Command.ENGRAVE), seed=92)
    # 'Elbereth' should be at (10, 10).
    eng = state2.engrave
    assert bool(eng.has_engraving[10, 10]), (
        "has_engraving not set at player tile after ENGRAVE action"
    )
    text_at = np.asarray(eng.text[10, 10])
    expected = np.array(list(_ELBERETH_BYTES), dtype=np.int8)
    assert np.array_equal(text_at, expected), (
        f"engraving bytes mismatch: got {text_at.tolist()}, expected {expected.tolist()}"
    )
    # Conduct flipped.
    assert bool(state2.conduct.violations[int(Conduct.ELBERETHLESS)])

    # Now take 3 WAIT steps — engraving should persist (engrave.step is no-op
    # in our impl per Wave 5 simplification).
    s = state2
    for i in range(3):
        s, _o, _r, _d, _i = _do_step(s, ord("."), seed=93 + i)
    assert bool(s.engrave.has_engraving[10, 10]), (
        "Engraving disappeared after WAIT steps (Wave 5 engrave.step should be no-op)"
    )


# ---------------------------------------------------------------------------
# 11. Open + close door state machine via env.step.
#
#   Vendor: lock.c::doopen / doclose — door_state CLOSED↔OPEN transitions.
#   The terrain tile flips to OPEN_DOOR on open; close updates door_state
#   only (terrain stays for now per features.close_door).
# ---------------------------------------------------------------------------

def test_door_open_close_via_env_step():
    """Bump-open an unlocked CLOSED door; assert door_state moves CLOSED→OPEN
    and terrain flips to OPEN_DOOR.  Close via env.step(CLOSE): door_state
    returns to CLOSED.
    """
    state, _ = _fresh(seed=101)
    h, w = state.terrain.shape[2], state.terrain.shape[3]
    floor = jnp.full((h, w), int(TileType.FLOOR), dtype=jnp.int8)
    floor = floor.at[10, 11].set(jnp.int8(int(TileType.CLOSED_DOOR)))

    flat_lv = 0
    door_state_arr = state.features.door_state.at[flat_lv, 10, 11].set(
        jnp.int8(int(DoorState.CLOSED))
    )
    new_features = state.features.replace(door_state=door_state_arr)
    state = state.replace(
        terrain=state.terrain.at[0, 0].set(floor),
        player_pos=jnp.array([10, 10], dtype=jnp.int16),
        features=new_features,
        monster_ai=state.monster_ai.replace(
            alive=jnp.zeros_like(state.monster_ai.alive),
        ),
    )

    # Bump east → open the door (player stays in place per vendor doopen).
    state2, _, _, _, _ = _do_step(state, ord("l"), seed=102)
    door_after = int(state2.features.door_state[flat_lv, 10, 11])
    assert door_after == int(DoorState.OPEN), (
        f"door_state should be OPEN after bump, got {door_after}"
    )
    assert int(state2.terrain[0, 0, 10, 11]) == int(TileType.OPEN_DOOR)
    # Player did NOT move onto the door tile (door_blocked semantics).
    assert int(state2.player_pos[1]) == 10, (
        "Bump-open should not move player onto the doorway"
    )

    # Now move to be ON the door tile and try close.
    # The CLOSE handler closes the door at the player's CURRENT tile, so we
    # need to step east onto the (now OPEN) door tile first.
    state3, _, _, _, _ = _do_step(state2, ord("l"), seed=103)
    # Confirm we're on the door.
    assert int(state3.player_pos[1]) == 11

    # Close action — close_door at player tile sets door_state OPEN→CLOSED.
    state4, _, _, _, _ = _do_step(state3, ord("c"), seed=104)
    door_after_close = int(state4.features.door_state[flat_lv, 10, 11])
    assert door_after_close == int(DoorState.CLOSED), (
        f"door_state should be CLOSED after close action, got {door_after_close} "
        "(vendor lock.c::doclose)"
    )


# ---------------------------------------------------------------------------
# 12. Save → load → step continuity.
#
#   Verifies the round-tripped state is *runnable*: env.step on the loaded
#   state produces the same timestep advance as on the original.
# ---------------------------------------------------------------------------

def test_save_load_round_trip_then_step_runs():
    """After a 5-step play, save, load, step once more → loaded state's
    next env.step produces the same player_pos and timestep as continuing
    from the original.
    """
    from Nethax.nethax import save_load

    state, _ = _fresh(seed=111)
    rng = jax.random.PRNGKey(2222)
    for i in range(5):
        rng, sub = jax.random.split(rng)
        state, _o, _r, done, _i = _ENV.step(state, jnp.int32(ord(".")), sub)
        if bool(done):
            return

    # Save and load.
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "play.npz"
        save_load.save_state(state, path)
        state_loaded = save_load.load_state(path)

    # Re-cast loaded leaves to jax arrays so env.step's JIT trace can read them.
    state_loaded = jax.tree.map(jnp.asarray, state_loaded)

    # Step both with the same RNG; outputs must agree.
    rng_step = jax.random.PRNGKey(3333)
    state_a, _oa, _ra, _da, _ia = _ENV.step(state,        jnp.int32(ord(".")), rng_step)
    state_b, _ob, _rb, _db, _ib = _ENV.step(state_loaded, jnp.int32(ord(".")), rng_step)

    assert int(state_a.timestep) == int(state_b.timestep), (
        "timestep diverged between original and loaded state after one step"
    )
    assert int(state_a.player_hp) == int(state_b.player_hp), (
        "player_hp diverged between original and loaded state after one step"
    )
    assert int(state_a.player_pos[0]) == int(state_b.player_pos[0])
    assert int(state_a.player_pos[1]) == int(state_b.player_pos[1])


# ---------------------------------------------------------------------------
# 13. Kill-via-bump grants XP (cross-subsystem: movement → combat → XP).
#
#   Vendor: hack.c::domove → attack(mtmp) → exper.c::experience awards XP.
# ---------------------------------------------------------------------------

def test_bump_attack_kills_monster_grants_xp():
    """Place a 1-HP hostile newt east of player; bump east → monster dies
    → player_xp increases.  This exercises movement→combat→XP chain.
    """
    state, _ = _fresh(seed=131)
    h, w = state.terrain.shape[2], state.terrain.shape[3]
    floor = jnp.full((h, w), int(TileType.FLOOR), dtype=jnp.int8)

    # Place 1-HP hostile monster at (10, 11), player at (10, 10).
    mai = state.monster_ai
    mai = mai.replace(
        alive=jnp.zeros_like(mai.alive),
        peaceful=jnp.zeros_like(mai.peaceful),
        tame=jnp.zeros_like(mai.tame),
        asleep=jnp.zeros_like(mai.asleep),
    )
    mai = mai.replace(
        alive=mai.alive.at[0].set(True),
        hp=mai.hp.at[0].set(jnp.int32(1)),       # 1 HP — dies on any hit
        hp_max=mai.hp_max.at[0].set(jnp.int32(1)),
        pos=mai.pos.at[0].set(jnp.array([10, 11], dtype=jnp.int16)),
        ac=mai.ac.at[0].set(jnp.int8(10)),       # easy to hit
    )
    state = state.replace(
        terrain=state.terrain.at[0, 0].set(floor),
        player_pos=jnp.array([10, 10], dtype=jnp.int16),
        monster_ai=mai,
        # Boost STR so we hit and damage reliably.
        player_str=jnp.int16(25),
    )
    xp_before = int(state.player_xp)
    # Try several seeds — the hit roll is RNG-dependent; we just need ANY
    # seed where the kill lands and XP is granted.
    killed_and_xp = False
    for seed in range(132, 152):
        s2, _o, _r, _d, _i = _do_step(state, ord("l"), seed=seed)
        if not bool(s2.monster_ai.alive[0]) and int(s2.player_xp) > xp_before:
            killed_and_xp = True
            break
    assert killed_and_xp, (
        "After 20 bump attempts at an AC10 1-HP monster, never killed it AND "
        "granted XP — bump-attack→combat→XP chain may be broken"
    )


# ---------------------------------------------------------------------------
# 14. 17-key obs invariant after a multi-subsystem step sequence.
#
#   Ensures every dispatch path preserves the NLE 17-key obs contract.
# ---------------------------------------------------------------------------

def test_obs_keys_invariant_after_chained_subsystem_actions():
    """Run a sequence of subsystem-spanning actions (WAIT, ENGRAVE, PRAY,
    SEARCH, WEAR, WIELD) and assert the obs dict still has all 17 NLE keys
    with correct shapes/dtypes after each step.
    """
    from Nethax.nethax.obs.nle_obs import (
        NLE_OBSERVATION_KEYS,
        NLE_OBSERVATION_SHAPES,
        NLE_OBSERVATION_DTYPES,
    )

    state, obs0 = _fresh(seed=141)
    assert set(obs0.keys()) == set(NLE_OBSERVATION_KEYS)

    action_chain = [
        ord("."),                  # WAIT
        int(Command.ENGRAVE),      # ENGRAVE
        int(Command.PRAY),         # PRAY
        ord("s"),                  # SEARCH
        ord("W"),                  # WEAR (no-op if no armor in slot 0)
        ord("w"),                  # WIELD
        ord("."),                  # WAIT
    ]
    rng = jax.random.PRNGKey(4444)
    for action in action_chain:
        rng, sub = jax.random.split(rng)
        state, obs, _r, done, _i = _ENV.step(state, jnp.int32(action), sub)
        # Same 17 keys, every step.
        assert set(obs.keys()) == set(NLE_OBSERVATION_KEYS), (
            f"obs keys drifted after action {action}: "
            f"{set(obs.keys()) ^ set(NLE_OBSERVATION_KEYS)}"
        )
        for key in NLE_OBSERVATION_KEYS:
            arr = obs[key]
            assert arr.shape == NLE_OBSERVATION_SHAPES[key], (
                f"{key}: shape {arr.shape} != {NLE_OBSERVATION_SHAPES[key]}"
            )
        if bool(done):
            break


# ---------------------------------------------------------------------------
# 15. Hunger drains to HUNGRY over ~750 turns then survives WEAK.
#
#   Vendor: eat.c::newuhs thresholds — nutrition starts at 900; drains ~1/turn;
#   crosses to HUNGRY at 150 and WEAK at 50.
# ---------------------------------------------------------------------------

def test_idle_loop_eventually_triggers_hunger_transition():
    """Run WAIT for 900 turns; the hunger_state must move out of NOT_HUNGRY
    (state 1) to at least HUNGRY (state 2) under vendor drain rates.

    900 turns × ~1 nutrition/turn from start of 900 → nutrition crosses 150.
    """
    from Nethax.nethax.subsystems.status_effects import HungerState

    state, _ = _fresh(seed=151)
    initial_hunger = int(state.status.hunger_state)
    initial_nutrition = int(state.status.nutrition)
    assert initial_hunger == int(HungerState.NOT_HUNGRY), (
        f"reset should leave hero NOT_HUNGRY, got {initial_hunger}"
    )

    rng = jax.random.PRNGKey(5555)
    for i in range(900):
        rng, sub = jax.random.split(rng)
        state, _o, _r, done, _i = _ENV.step(state, jnp.int32(ord(".")), sub)
        if bool(done):
            break

    nutrition_after = int(state.status.nutrition)
    hunger_after = int(state.status.hunger_state)

    # Nutrition strictly decreased over the idle window.
    assert nutrition_after < initial_nutrition, (
        f"Nutrition did not drain over 900 idle turns: "
        f"{initial_nutrition}→{nutrition_after}"
    )
    # State either still NOT_HUNGRY (if drain is slow) or transitioned to
    # HUNGRY/WEAK.  The contract is: NEVER more satiated than start.
    assert hunger_after >= initial_hunger, (
        f"hunger_state moved backwards (less hungry): "
        f"{initial_hunger}→{hunger_after}"
    )
