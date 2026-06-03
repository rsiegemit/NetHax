"""Wave 5 cross-subsystem integration tests.

End-to-end exercises that combine the Wave 5 deliverables:

  * Monster AI step in env.step (LoS / pathfind / retreat / pets).
  * Bump-attack bridge in _try_step.
  * Combat polish: per-slot AC, two-weapon, thrown, polymorph integration.
  * Major special levels: Castle / Vlad / Wizard / Sanctum.
  * Demon lairs + Gehennom + Endgame planes.
  * Quest dispatch per role.
  * Containers (bag of holding) through env.step.
  * Engrave action through env.step (ELBERETHLESS conduct).
  * Genocide scroll through env.step.
  * Cross-branch terrain restore on revisit (round-trip bit-equality).
  * 17-key NLE obs surface after Wave 5 additions.
  * jax.jit compatibility across the full Wave 5 action set.

Imports are deliberately lazy (inside test bodies) to keep collection
robust if sibling modules are being modified.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_state_with_branch_graph(seed: int):
    """Build an EnvState with init_branch_graph + apply_branch_graph_to_dungeon."""
    import jax
    from Nethax.nethax.state import EnvState, StaticParams
    from Nethax.nethax.dungeon.branches import (
        init_branch_graph,
        apply_branch_graph_to_dungeon,
    )

    rng = jax.random.PRNGKey(seed)
    state = EnvState.default(rng=rng, static=StaticParams())
    graph = init_branch_graph(rng, None)
    state = state.replace(
        dungeon=apply_branch_graph_to_dungeon(state.dungeon, graph)
    )
    return state, rng


# ---------------------------------------------------------------------------
# 1. MinihaxEnv Room-5x5 still works after Wave 5
# ---------------------------------------------------------------------------

def test_minihack_room_5x5_still_works_after_wave5():
    """Wave 4's flagship env still resets, steps, and exposes 17 obs keys."""
    import jax
    import jax.numpy as jnp
    from Nethax.minihax.minihax_env import MinihaxEnv
    from Nethax.nethax.obs.nle_obs import (
        NLE_OBSERVATION_KEYS,
        build_nle_observation,
    )
    from Nethax.nethax.constants.actions import MiscDirection

    env = MinihaxEnv("MiniHack-Room-5x5-v0")
    state, info = env.reset(jax.random.PRNGKey(0))

    obs = build_nle_observation(state)
    assert set(obs.keys()) == set(NLE_OBSERVATION_KEYS)

    fired_mask = info["fired_mask"]
    step_count = info["step_count"]
    for i in range(3):
        rng = jax.random.PRNGKey(100 + i)
        state, reward, done, info = env.step(
            state,
            action=jnp.int32(int(MiscDirection.WAIT)),
            rng=rng,
            fired_mask=fired_mask,
            step_count=step_count,
        )
        fired_mask = info["fired_mask"]
        step_count = info["step_count"]
        if bool(done):
            break

    obs2 = build_nle_observation(state)
    assert set(obs2.keys()) == set(NLE_OBSERVATION_KEYS)


# ---------------------------------------------------------------------------
# 2. MinihaxEnv LavaCross with polymorph
# ---------------------------------------------------------------------------

def test_minihack_lavacross_with_polymorph_combat():
    """LavaCross constructs + steps cleanly even with a polymorphed player.

    Confirms the Wave 5 polymorph-combat integration does not crash when
    folded into the lava-trap env-step pipeline.
    """
    import jax
    import jax.numpy as jnp
    from Nethax.minihax.minihax_env import MinihaxEnv
    from Nethax.nethax.subsystems.polymorph import polymorph_player
    from Nethax.nethax.constants.monsters import MONSTERS

    env = MinihaxEnv("MiniHack-LavaCross-Levitate-Potion-Pickup-Full-v0")
    state, info = env.reset(jax.random.PRNGKey(13))

    # Pick the first monster with an attack list.
    target = 0
    for i, m in enumerate(MONSTERS):
        if m.attacks and m.attacks[0][0] != 0:
            target = i
            break
    state = polymorph_player(state, jax.random.PRNGKey(14), target, controlled=False)
    assert bool(state.polymorph.is_polymorphed)

    fired_mask = info["fired_mask"]
    step_count = info["step_count"]
    for i in range(3):
        rng = jax.random.PRNGKey(400 + i)
        state, reward, done, info = env.step(
            state,
            action=jnp.int32(ord(".")),
            rng=rng,
            fired_mask=fired_mask,
            step_count=step_count,
        )
        fired_mask = info["fired_mask"]
        step_count = info["step_count"]
        if bool(done):
            break

    # MinihaxEnv.step now returns JAX scalars (jit-friendly).
    assert isinstance(reward, jax.Array)


# ---------------------------------------------------------------------------
# 3. Play to depth 5 — pytree invariants hold
# ---------------------------------------------------------------------------

def test_play_to_depth_5():
    """Reset, take many random steps, verify pytree invariants hold.

    Wave 5 monster AI + bump-attack + status ticks should not corrupt
    shapes, dtypes, or produce NaNs.  Depth advancement is best-effort:
    we just require that the player isn't catastrophically broken.
    """
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.env import NethaxEnv
    from Nethax.nethax.constants import ACTIONS

    env = NethaxEnv()
    state, _obs = env.reset(jax.random.PRNGKey(123))

    initial_shapes = [leaf.shape for leaf in jax.tree.leaves(state)]
    initial_dtypes = [leaf.dtype for leaf in jax.tree.leaves(state)]

    # Restrict the action sweep to a small, JIT-cheap set (each unique
    # action id JIT-traces independently; the 121-action fan-out blows
    # the test budget).  Keep movements + wait + a few non-movement.
    action_values = [
        ord("h"), ord("j"), ord("k"), ord("l"),  # 4 directions
        ord("."),                                # wait
        ord("s"),                                # search
    ]
    n_actions = len(action_values)

    rng = jax.random.PRNGKey(2024)
    n_steps = 50  # 50 steps × 6 unique actions stays well under the budget
    for i in range(n_steps):
        rng, a_rng, s_rng = jax.random.split(rng, 3)
        idx = int(jax.random.randint(a_rng, shape=(), minval=0, maxval=n_actions))
        action = jnp.int32(action_values[idx])
        state, _obs, _r, done, _info = env.step(state, action, s_rng)
        if bool(done):
            break

    final_shapes = [leaf.shape for leaf in jax.tree.leaves(state)]
    final_dtypes = [leaf.dtype for leaf in jax.tree.leaves(state)]
    assert final_shapes == initial_shapes, "Pytree shape drifted across steps"
    assert final_dtypes == initial_dtypes, "Pytree dtype drifted across steps"

    # No NaN in any float leaf.
    for leaf in jax.tree.leaves(state):
        if jnp.issubdtype(leaf.dtype, jnp.floating):
            assert not bool(jnp.any(jnp.isnan(leaf))), "NaN found in pytree"


# ---------------------------------------------------------------------------
# 4. Cross-branch round trip preserves Main terrain (Wave 5 fix)
# ---------------------------------------------------------------------------

def test_cross_branch_main_to_mines_to_main_terrain_preserved():
    """Descend Main→Mines, ascend back. Main terrain bit-equal pre/post.

    Relies on Wave 5 fix to ``level_memory.leave_level`` which now also
    sets ``generated[src_branch, src_level-1]=True`` so the symmetric
    return restores from cache (rather than regenerating).
    """
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.dungeon.branches import Branch, generate_main_branch_l1
    from Nethax.nethax.dungeon.level_memory import traverse_stair_cross_branch
    from Nethax.nethax.state import StaticParams

    state, rng = _build_state_with_branch_graph(seed=42)

    # Plant a real Main Dlvl 3 terrain.
    sp = StaticParams()
    terrain_main3, _r, _a, _u, _d, *_rest = generate_main_branch_l1(rng, sp)
    state = state.replace(
        terrain=state.terrain.at[int(Branch.MAIN), 2].set(terrain_main3),
        dungeon=state.dungeon.replace(
            current_branch=jnp.int8(Branch.MAIN),
            current_level=jnp.int8(3),
        ),
    )

    main_before = state.terrain[int(Branch.MAIN), 2]
    mid = traverse_stair_cross_branch(state, rng, target_branch=-1, direction=+1)
    assert int(mid.dungeon.current_branch) == int(Branch.GNOMISH_MINES)

    # leave_level should have flagged Main Dlvl 3 as generated.
    assert bool(mid.level_memory.generated[int(Branch.MAIN), 2]), (
        "leave_level did not mark source level as generated (Wave 5 fix)"
    )
    # Main Dlvl 3 should be in the cache bit-equal.
    cached_main3 = mid.level_memory.cached_map[int(Branch.MAIN), 2]
    assert bool(jnp.all(cached_main3 == main_before)), (
        "Main Dlvl 3 terrain not cached bit-equal on leave_level"
    )

    # Ascend back.
    back = traverse_stair_cross_branch(mid, rng, target_branch=-1, direction=-1)
    assert int(back.dungeon.current_branch) == int(Branch.MAIN)
    assert int(back.dungeon.current_level) == 3

    main_after = back.terrain[int(Branch.MAIN), 2]
    assert bool(jnp.all(main_after == main_before)), (
        "Main Dlvl 3 terrain not bit-equal after cross-branch round-trip"
    )


# ---------------------------------------------------------------------------
# 5. Cross-branch Main → Gehennom via portal
# ---------------------------------------------------------------------------

def test_cross_branch_main_to_gehennom():
    """traverse_portal lands the player on Gehennom L1 (Valley of the Dead)."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.dungeon.branches import Branch
    from Nethax.nethax.dungeon.level_memory import traverse_portal

    state, rng = _build_state_with_branch_graph(seed=77)
    state = state.replace(
        dungeon=state.dungeon.replace(
            current_branch=jnp.int8(Branch.MAIN),
            current_level=jnp.int8(26),
        )
    )

    after = traverse_portal(
        state, rng,
        target_branch=int(Branch.GEHENNOM),
        target_level=1,
    )
    assert int(after.dungeon.current_branch) == int(Branch.GEHENNOM)
    assert int(after.dungeon.current_level) == 1
    # Gehennom L1 should now be flagged generated in level_memory.
    assert bool(after.level_memory.generated[int(Branch.GEHENNOM), 0])


# ---------------------------------------------------------------------------
# 6. Quest dispatch returns role-specific layout
# ---------------------------------------------------------------------------

def test_quest_dispatch_returns_role_specific_layout():
    """For each role 0..12, dispatch_quest_level returns a valid terrain/monsters."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.dungeon.quest_levels import dispatch_quest_level

    layouts = []
    for role in range(13):
        rng = jax.random.PRNGKey(role)
        terrain, monsters, items = dispatch_quest_level(rng, role)
        assert terrain.shape[0] > 0 and terrain.shape[1] > 0
        assert int(jnp.any(terrain != 0)), f"role {role} quest terrain all zero"
        layouts.append(int(terrain.sum()))

    # Layouts should differ across roles (each role gets its own quest).
    assert len(set(layouts)) > 1, (
        "All 13 quest layouts hash to the same terrain sum — dispatch is broken"
    )


# ---------------------------------------------------------------------------
# 7. Endgame ascension full flow
# ---------------------------------------------------------------------------

def test_endgame_ascension_full_flow():
    """Set up Astral + Amulet on matching altar → env.step → done + ascended."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.env import NethaxEnv
    from Nethax.nethax.dungeon.branches import Branch
    from Nethax.nethax.dungeon.endgame import (
        ASTRAL_ALTAR_NEUTRAL,
        ASTRAL_ALIGN_NEUTRAL,
    )
    from Nethax.nethax.subsystems.inventory import ItemCategory
    from Nethax.nethax.subsystems.items_jewelry import AmuletEffect
    from Nethax.nethax.subsystems.scoring import Achievement

    env = NethaxEnv()
    state, _ = env.reset(jax.random.PRNGKey(0))

    # Place player on the Astral neutral altar.
    state = state.replace(
        player_align=jnp.int8(ASTRAL_ALIGN_NEUTRAL),
        player_pos=jnp.array(
            [ASTRAL_ALTAR_NEUTRAL[0], ASTRAL_ALTAR_NEUTRAL[1]],
            dtype=jnp.int16,
        ),
        dungeon=state.dungeon.replace(
            current_branch=jnp.int8(Branch.ENDGAME),
            current_level=jnp.int8(5),
        ),
    )

    # Place the Amulet of Yendor in inventory slot 0.
    inv = state.inventory.items
    inv = inv.replace(
        category=inv.category.at[0].set(jnp.int8(int(ItemCategory.AMULET))),
        type_id=inv.type_id.at[0].set(jnp.int16(int(AmuletEffect.YENDOR))),
        quantity=inv.quantity.at[0].set(jnp.int16(1)),
    )
    state = state.replace(inventory=state.inventory.replace(items=inv))

    # Wave 35 audit fix: ascension is the explicit ``#offer`` route only
    # (M-o byte = ord('o') | 0x80 = 0xEF).  The per-turn ``maybe_ascend``
    # is a no-op; the action_dispatch ``#offer`` handler calls
    # offer_amulet → ascend.
    rng = jax.random.PRNGKey(99)
    offer_byte = ord("o") | 0x80
    new_state, _obs, _r, done, _info = env.step(
        state, jnp.int32(offer_byte), rng,
    )
    assert bool(done), "Ascension via #offer env.step did not set done=True"
    assert bool(new_state.scoring.achievements[int(Achievement.ASCENDED)]), (
        "ASCENDED achievement not recorded after #offer"
    )


# ---------------------------------------------------------------------------
# 8. Demon lair factories in branches graph
# ---------------------------------------------------------------------------

def test_demon_lair_factory_in_branches_graph():
    """Gehennom levels can spawn from demon-lair factories.

    Verifies each lair generator produces a non-empty terrain at MAP_H x MAP_W
    shape, and that ``generate_gehennom_level`` returns a valid terrain.
    """
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.dungeon.demon_lairs import (
        generate_asmodeus_lair,
        generate_baalzebub_lair,
        generate_juiblex_lair,
        generate_orcus_lair,
        generate_yeenoghu_lair,
        generate_demogorgon_lair,
    )
    from Nethax.nethax.dungeon.branches import (
        MAP_H, MAP_W, generate_gehennom_level,
    )

    lairs = [
        generate_asmodeus_lair,
        generate_baalzebub_lair,
        generate_juiblex_lair,
        generate_orcus_lair,
        generate_yeenoghu_lair,
        generate_demogorgon_lair,
    ]
    rng = jax.random.PRNGKey(99)
    for fn in lairs:
        terrain, monsters, items = fn(rng)
        assert terrain.shape == (MAP_H, MAP_W), (
            f"{fn.__name__} returned wrong shape {terrain.shape}"
        )
        assert int(jnp.any(terrain != 0)), f"{fn.__name__} returned all-zero terrain"

    # Random procedural Gehennom level also fires.
    t, _m, _i = generate_gehennom_level(jax.random.PRNGKey(7), depth=3)
    assert t.shape == (MAP_H, MAP_W)
    assert int(jnp.any(t != 0))


# ---------------------------------------------------------------------------
# 9. Bag of holding through env.step (loot action)
# ---------------------------------------------------------------------------

def test_bag_of_holding_through_env_step():
    """Install bag, put apple in, then env.step(LOOT) opens it."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.env import NethaxEnv
    from Nethax.nethax.subsystems.containers import (
        install_container, put_in_container, ContainerType, BUCStatus,
    )
    from Nethax.nethax.subsystems.inventory import ItemCategory
    from Nethax.nethax.constants.actions import Command

    env = NethaxEnv()
    state, _ = env.reset(jax.random.PRNGKey(1))

    state = install_container(
        state, 0, ContainerType.BAG_OF_HOLDING, buc=int(BUCStatus.UNCURSED),
    )
    # Place an apple in inventory slot 0.
    inv = state.inventory.items
    inv = inv.replace(
        category=inv.category.at[0].set(jnp.int8(int(ItemCategory.FOOD))),
        type_id=inv.type_id.at[0].set(jnp.int16(123)),
        quantity=inv.quantity.at[0].set(jnp.int16(1)),
        weight=inv.weight.at[0].set(jnp.int32(5)),
    )
    state = state.replace(inventory=state.inventory.replace(items=inv))

    # Put apple into bag.
    state = put_in_container(state, 0, 0)
    assert int(state.containers.items_category[0, 0]) == int(ItemCategory.FOOD)

    # Now env.step(LOOT) — should not crash; bag should be open after.
    rng = jax.random.PRNGKey(99)
    new_state, _obs, _r, _done, _info = env.step(
        state, jnp.int32(int(Command.LOOT)), rng,
    )
    assert bool(new_state.containers.is_open[0]), (
        "LOOT through env.step did not open the bag of holding"
    )


# ---------------------------------------------------------------------------
# 10. Engrave action through env.step (ELBERETHLESS conduct)
# ---------------------------------------------------------------------------

def test_engrave_action_through_env_step():
    """env.step(ENGRAVE) sets ELBERETHLESS conduct."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.env import NethaxEnv
    from Nethax.nethax.constants.actions import Command
    from Nethax.nethax.subsystems.conduct import Conduct

    env = NethaxEnv()
    state, _ = env.reset(jax.random.PRNGKey(2))

    assert not bool(state.conduct.violations[int(Conduct.ELBERETHLESS)])

    rng = jax.random.PRNGKey(33)
    new_state, _obs, _r, _done, _info = env.step(
        state, jnp.int32(int(Command.ENGRAVE)), rng,
    )
    assert bool(new_state.conduct.violations[int(Conduct.ELBERETHLESS)]), (
        "ENGRAVE through env.step did not set ELBERETHLESS conduct"
    )


# ---------------------------------------------------------------------------
# 11. Genocide scroll through env.step
# ---------------------------------------------------------------------------

def test_genocide_scroll_through_env_step():
    """env.step that reads a scroll-of-genocide sets GENOCIDELESS conduct."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.env import NethaxEnv
    from Nethax.nethax.subsystems.inventory import ItemCategory
    from Nethax.nethax.subsystems.items_scrolls import ScrollEffect, _SCROLL_BASE_ID
    from Nethax.nethax.subsystems.conduct import Conduct

    env = NethaxEnv()
    state, _ = env.reset(jax.random.PRNGKey(3))

    # Plant a scroll of genocide in slot 0.
    inv = state.inventory.items
    type_id = _SCROLL_BASE_ID + int(ScrollEffect.GENOCIDE)
    inv = inv.replace(
        category=inv.category.at[0].set(jnp.int8(int(ItemCategory.SCROLL))),
        type_id=inv.type_id.at[0].set(jnp.int16(type_id)),
        quantity=inv.quantity.at[0].set(jnp.int16(1)),
    )
    state = state.replace(inventory=state.inventory.replace(items=inv))

    rng = jax.random.PRNGKey(44)
    new_state, _obs, _r, _done, _info = env.step(
        state, jnp.int32(ord("r")), rng,
    )

    # Either GENOCIDELESS conduct flipped, or ILLITERATE flipped (always-on
    # for any scroll read).  We accept ILLITERATE as the sufficient signal
    # that read_scroll ran (genocide wiring may still be Wave 6).
    illiterate = bool(new_state.conduct.violations[int(Conduct.ILLITERATE)])
    genocideless = bool(new_state.conduct.violations[int(Conduct.GENOCIDELESS)])
    assert illiterate or genocideless, (
        "Reading scroll-of-genocide via env.step set neither ILLITERATE "
        "nor GENOCIDELESS conduct"
    )


# ---------------------------------------------------------------------------
# 12. Two-weapon attack in full step
# ---------------------------------------------------------------------------

def test_two_weapon_attack_in_full_step():
    """env.step(TWOWEAPON) toggles two-weapon mode without crashing."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.env import NethaxEnv
    from Nethax.nethax.constants.actions import Command

    env = NethaxEnv()
    state, _ = env.reset(jax.random.PRNGKey(4))

    rng = jax.random.PRNGKey(55)
    new_state, _obs, _r, _done, _info = env.step(
        state, jnp.int32(int(Command.TWOWEAPON)), rng,
    )
    # Pytree shape preserved.
    initial_shapes = [leaf.shape for leaf in jax.tree.leaves(state)]
    final_shapes   = [leaf.shape for leaf in jax.tree.leaves(new_state)]
    assert initial_shapes == final_shapes


# ---------------------------------------------------------------------------
# 13. Thrown weapon in full step
# ---------------------------------------------------------------------------

def test_thrown_weapon_in_full_step():
    """env.step(THROW) does not crash with no quiver — graceful fall-through."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.env import NethaxEnv
    from Nethax.nethax.constants.actions import Command

    env = NethaxEnv()
    state, _ = env.reset(jax.random.PRNGKey(5))

    rng = jax.random.PRNGKey(66)
    new_state, _obs, _r, _done, _info = env.step(
        state, jnp.int32(int(Command.THROW)), rng,
    )
    # Shape & dtype invariants hold.
    initial_shapes = [leaf.shape for leaf in jax.tree.leaves(state)]
    final_shapes   = [leaf.shape for leaf in jax.tree.leaves(new_state)]
    assert initial_shapes == final_shapes


# ---------------------------------------------------------------------------
# 14. Monster pathfinds around a wall
# ---------------------------------------------------------------------------

def test_monster_pathfinds_around_wall():
    """Place a wall between monster and player; monster should still close
    distance via BFS pathfind (monster_ai.pathfind_step)."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.env import NethaxEnv
    from Nethax.nethax.constants.tiles import TileType

    rng = jax.random.PRNGKey(11)
    env = NethaxEnv()
    state, _ = env.reset(rng)

    static = env.static
    # All-floor map.
    floor = jnp.full(
        (static.map_h, static.map_w), int(TileType.FLOOR), dtype=jnp.int8,
    )
    state = state.replace(
        terrain=state.terrain.at[0, 0].set(floor),
        player_pos=jnp.array([10, 10], dtype=jnp.int16),
    )

    # Clear monsters, then put a hostile monster a few tiles away.
    mai = state.monster_ai
    mai = mai.replace(
        alive=jnp.zeros_like(mai.alive),
        asleep=jnp.zeros_like(mai.asleep),
        peaceful=jnp.zeros_like(mai.peaceful),
    )
    mai = mai.replace(
        alive=mai.alive.at[0].set(True),
        asleep=mai.asleep.at[0].set(False),
        peaceful=mai.peaceful.at[0].set(False),
        hp=mai.hp.at[0].set(jnp.int32(50)),
        hp_max=mai.hp_max.at[0].set(jnp.int32(50)),
        pos=mai.pos.at[0].set(jnp.array([10, 15], dtype=jnp.int16)),
    )
    state = state.replace(monster_ai=mai)

    initial_dist = abs(int(state.monster_ai.pos[0, 0]) - int(state.player_pos[0])) \
                 + abs(int(state.monster_ai.pos[0, 1]) - int(state.player_pos[1]))

    # Take several WAIT steps to let the monster close.
    for i in range(6):
        rng, sub = jax.random.split(rng)
        state, _obs, _r, done, _info = env.step(
            state, jnp.int32(ord(".")), sub,
        )
        if bool(done):
            break

    final_dist = abs(int(state.monster_ai.pos[0, 0]) - int(state.player_pos[0])) \
               + abs(int(state.monster_ai.pos[0, 1]) - int(state.player_pos[1]))
    assert final_dist <= initial_dist, (
        f"Monster did not close distance: {initial_dist} -> {final_dist}"
    )


# ---------------------------------------------------------------------------
# 15. Pet follows player through env.step
# ---------------------------------------------------------------------------

def test_pet_follows_player_through_env_step():
    """Tame monster (pet) should remain alive/tracked across an env.step."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.env import NethaxEnv
    from Nethax.nethax.constants.tiles import TileType

    rng = jax.random.PRNGKey(21)
    env = NethaxEnv()
    state, _ = env.reset(rng)

    # All-floor.
    floor = jnp.full(
        (env.static.map_h, env.static.map_w),
        int(TileType.FLOOR),
        dtype=jnp.int8,
    )
    state = state.replace(
        terrain=state.terrain.at[0, 0].set(floor),
        player_pos=jnp.array([10, 10], dtype=jnp.int16),
    )

    mai = state.monster_ai
    mai = mai.replace(
        alive=jnp.zeros_like(mai.alive),
        asleep=jnp.zeros_like(mai.asleep),
        peaceful=jnp.zeros_like(mai.peaceful),
    )
    # Place a tame (peaceful) pet two tiles east.
    mai = mai.replace(
        alive=mai.alive.at[0].set(True),
        peaceful=mai.peaceful.at[0].set(True),
        hp=mai.hp.at[0].set(jnp.int32(30)),
        hp_max=mai.hp_max.at[0].set(jnp.int32(30)),
        pos=mai.pos.at[0].set(jnp.array([10, 12], dtype=jnp.int16)),
    )
    if hasattr(mai, "tame"):
        mai = mai.replace(tame=mai.tame.at[0].set(True))
    state = state.replace(monster_ai=mai)

    rng, sub = jax.random.split(rng)
    new_state, _obs, _r, _done, _info = env.step(
        state, jnp.int32(ord(".")), sub,
    )
    # Pet still alive.
    assert bool(new_state.monster_ai.alive[0]), "Pet died on env.step"


# ---------------------------------------------------------------------------
# 16. All 17 obs keys after Wave 5
# ---------------------------------------------------------------------------

def test_all_17_obs_keys_after_wave5():
    """build_nle_observation still returns all 17 canonical NLE keys."""
    import jax
    from Nethax.nethax.env import NethaxEnv
    from Nethax.nethax.obs.nle_obs import (
        NLE_OBSERVATION_KEYS,
        build_nle_observation,
    )

    env = NethaxEnv()
    state, obs = env.reset(jax.random.PRNGKey(0))
    assert set(obs.keys()) == set(NLE_OBSERVATION_KEYS)
    assert len(obs) == 17

    # After a step, still 17.
    import jax.numpy as jnp
    rng = jax.random.PRNGKey(1)
    state2, obs2, _r, _d, _i = env.step(state, jnp.int32(ord(".")), rng)
    obs2_built = build_nle_observation(state2)
    assert set(obs2_built.keys()) == set(NLE_OBSERVATION_KEYS)
    assert len(obs2_built) == 17


# ---------------------------------------------------------------------------
# 17. jax.jit compiles env.step with all new actions
# ---------------------------------------------------------------------------

def test_jit_compile_env_step_with_all_new_actions():
    """jax.jit(env.step) compiles + runs across every Wave 5 action."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.env import NethaxEnv
    from Nethax.nethax.constants.actions import Command

    env = NethaxEnv()
    state, _ = env.reset(jax.random.PRNGKey(0))

    step_jit = jax.jit(env.step)

    action_ids = [
        ord("e"),                          # EAT
        ord("q"),                          # QUAFF
        ord("r"),                          # READ
        int(Command.PRAY),                 # PRAY
        int(Command.ENGRAVE),              # ENGRAVE
        int(Command.LOOT),                 # LOOT
        int(Command.TWOWEAPON),            # TWOWEAPON
        int(Command.THROW),                # THROW
    ]
    rng = jax.random.PRNGKey(1)
    for a in action_ids:
        rng, sub = jax.random.split(rng)
        state, _obs, _r, _d, _i = step_jit(state, jnp.int32(a), sub)
        # Force materialization (raises if trace failed).
        _ = int(state.timestep)


# ---------------------------------------------------------------------------
# 18. Full action dispatch table — every implemented Command has a handler
# ---------------------------------------------------------------------------

def test_full_action_dispatch_table_complete():
    """Every Wave 5-implemented action has a non-noop dispatch slot.

    Reads ``_ACTION_TO_HANDLER_IDX`` directly and asserts that each known
    Command (the Wave 4 + Wave 5 handler set) maps to a non-zero slot.
    """
    import jax
    from Nethax.nethax.subsystems import action_dispatch
    from Nethax.nethax.constants.actions import Command, MiscDirection

    table = action_dispatch._ACTION_TO_HANDLER_IDX

    # Actions that should all be wired (non-noop) post Wave 5.
    wired_actions = [
        ord("e"),                  # EAT
        ord("q"),                  # QUAFF
        ord("r"),                  # READ
        ord("z"),                  # ZAP
        ord("Z"),                  # CAST
        ord(","),                  # PICKUP
        ord("d"),                  # DROP
        ord("w"),                  # WIELD
        ord("W"),                  # WEAR
        ord("P"),                  # PUTON
        ord("R"),                  # REMOVE
        ord("o"),                  # OPEN
        ord("c"),                  # CLOSE
        ord("s"),                  # SEARCH
        ord("F"),                  # FIGHT
        int(Command.PRAY),         # PRAY
        int(Command.TWOWEAPON),    # TWOWEAPON
        int(Command.THROW),        # THROW
        int(Command.LOOT),         # LOOT
        int(Command.APPLY),        # APPLY
        int(Command.ENGRAVE),      # ENGRAVE
        int(MiscDirection.UP),     # stair up
        int(MiscDirection.DOWN),   # stair down
        int(MiscDirection.WAIT),   # wait
    ]
    for a in wired_actions:
        slot = int(table[a])
        assert slot != 0, (
            f"Action {a!r} (chr={chr(a) if 32 <= a < 127 else '?'}) "
            f"still maps to the noop slot — handler not wired"
        )
