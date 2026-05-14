"""Wave 6 closing-audit #88 — vendor-parity tests for character init,
monster spawning distribution, and dungeon branch graph.

Sources of truth:
  vendor/nle/src/u_init.c          — u_init / ini_hpwp
  vendor/nle/src/attrib.c          — init_attr, newhp
  vendor/nle/src/exper.c           — newpw
  vendor/nle/src/role.c            — roles[] / races[] tables
  vendor/nle/src/makemon.c         — rndmonst / pm_gen / newmonhp /
                                     peace_minded
  vendor/nle/src/dungeon.c         — init_dungeons / branch wiring
  vendor/nle/dat/dungeon.def       — canonical branch entry depths
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest


# ===========================================================================
# A. character.py — init logic vs u_init.c::u_init / attrib.c::init_attr
# ===========================================================================


def _reset_as(role, race=None, seed=42):
    """Helper: reset env as ``role`` with ``race`` (defaults to HUMAN)."""
    from Nethax.nethax.env import NethaxEnv
    from Nethax.nethax.constants.races import Race

    if race is None:
        race = Race.HUMAN
    rng = jax.random.PRNGKey(seed)
    env = NethaxEnv()
    state, _ = env.reset(rng, role=role, race=race, alignment=0)
    return state


def test_valkyrie_initial_str_18():
    """Valkyrie + Human's STR should land near 18 after init_attr(75).

    Vendor formula (attrib.c::init_attr): start STR at role.attrbase[0]=10,
    then distribute 27 remaining points weighted by attrdist=(30,6,7,20,30,7)
    — STR has 30% weight so on average it picks up ~8 extra points, landing
    near 18.  We assert STR >= 14 (well above the role floor) and <= the
    Human STR18(100)=118 cap to catch table regressions while tolerating
    randomness.
    """
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.constants.races import Race

    # Use a few seeds to confirm STR routinely lands at or above the
    # canonical 18 vendor anchor.  A bug in init_attr (e.g. skipping the
    # weighted-distribution loop) would peg STR at the role floor of 10.
    high_str_count = 0
    for seed in (1, 2, 3, 4, 5, 6, 7, 8):
        state = _reset_as(Role.VALKYRIE, Race.HUMAN, seed=seed)
        s = int(state.player_str)
        assert s >= 10, f"Valkyrie STR={s} below role floor (10)"
        assert s <= 118, f"Valkyrie STR={s} above HUMAN STR cap (118)"
        if s >= 14:
            high_str_count += 1
    # Across 8 seeds at least half should clear 14 — proves the
    # weighted-distribution step is actually firing.
    assert high_str_count >= 4, (
        f"Valkyrie STR rarely crosses 14 ({high_str_count}/8); "
        "init_attr weighted distribution may not be running."
    )


def test_wizard_initial_int_18():
    """Wizard + Human's INT should land near 18 (attrdist=30 on INT)."""
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.constants.races import Race

    high_int_count = 0
    for seed in (1, 2, 3, 4, 5, 6, 7, 8):
        state = _reset_as(Role.WIZARD, Race.HUMAN, seed=seed)
        i = int(state.player_int)
        assert i >= 10, f"Wizard INT={i} below role floor (10)"
        assert i <= 18, f"Wizard INT={i} above HUMAN INT cap (18)"
        if i >= 14:
            high_int_count += 1
    assert high_int_count >= 4, (
        f"Wizard INT rarely crosses 14 ({high_int_count}/8); "
        "init_attr weighted distribution may not be running."
    )


def test_priest_initial_alignment_record_5():
    """Priest starts with u.ualign.record = role.initrecord = 5.

    Vendor: u_init.c::ini_hpwp at ulevel==0 copies role.initrecord into
    u.ualign.record (vendor/nle/src/attrib.c lines 992-995, vendor 3.7 line
    1094).  PrayerState.alignment_record mirrors this field.
    """
    from Nethax.nethax.constants.roles import Role, get_role

    # Vendor table value.
    assert int(get_role(Role.PRIEST).initrecord) == 5, (
        "vendor role.c Priest initrecord must be 5"
    )

    state = _reset_as(Role.PRIEST)
    rec = int(state.prayer.alignment_record)
    assert rec == 5, (
        f"Priest alignment_record={rec}, expected 5 from role.initrecord"
    )


def test_initial_pw_in_role_range():
    """All roles' starting Pw must equal vendor newpw(ulevel==0) range.

    Vendor (exper.c::newpw lines 51-56):
      pw = role.enadv.infix + race.enadv.infix
           + rnd(role.enadv.inrnd) [if > 0]
           + rnd(race.enadv.inrnd) [if > 0]
      pw = max(pw, 1)
    """
    from Nethax.nethax.constants.roles import Role, get_role
    from Nethax.nethax.constants.races import Race, get_race

    race = Race.HUMAN
    race_entry = get_race(race)
    for role in Role:
        r_entry = get_role(role)
        # vendor rnd(n) returns 1..n; min pw with rnd is infix + 1
        # when inrnd>0 (or just infix when inrnd==0).
        min_pw = r_entry.enadv.infix + race_entry.enadv.infix
        if r_entry.enadv.inrnd > 0:
            min_pw += 1
        if race_entry.enadv.inrnd > 0:
            min_pw += 1
        min_pw = max(min_pw, 1)
        max_pw = (
            r_entry.enadv.infix + race_entry.enadv.infix
            + max(r_entry.enadv.inrnd, 0)
            + max(race_entry.enadv.inrnd, 0)
        )
        max_pw = max(max_pw, 1)

        # Sample two seeds and verify both fall in [min_pw, max_pw].
        for seed in (101, 202):
            state = _reset_as(role, race, seed=seed)
            pw = int(state.player_pw)
            assert min_pw <= pw <= max_pw, (
                f"{role.name}: Pw={pw} outside vendor range "
                f"[{min_pw}, {max_pw}] (enadv={r_entry.enadv})"
            )


# ===========================================================================
# B. spawning.py — distribution vs makemon.c::makemon
# ===========================================================================


def test_spawn_eligibility_respects_diff_lvl_plus_5():
    """All monsters eligible at depth ``D`` have ``diff_lvl <= D + 5``.

    Vendor reference: makemon.c::rndmonst rejects entries with
    ``mons[i].difficulty > zlevel + 4`` (the "+4 -> +5" off-by-one matches
    NetHack's vs our depth window).
    """
    from Nethax.nethax.dungeon.spawning import (
        MONSTR_DIFFICULTIES,
        eligible_monsters_for_depth,
    )

    for depth in (1, 3, 5, 8, 14, 20, 26):
        mask = eligible_monsters_for_depth(depth=depth)
        diffs = MONSTR_DIFFICULTIES[mask]
        if diffs.shape[0] > 0:
            assert bool(jnp.all(diffs <= depth + 5)), (
                f"depth={depth}: some eligible monsters have "
                f"diff_lvl > {depth + 5}: max={int(jnp.max(diffs))}"
            )


def test_spawn_weighted_by_gen_freq():
    """The empirical pick distribution at a fixed depth must correlate with
    gen_freq for eligible monsters.

    Vendor: makemon.c::pm_gen weights by ``mons[i].geno & G_FREQ`` (low byte).
    """
    from Nethax.nethax.dungeon.spawning import (
        pick_monster_for_level,
        eligible_monsters_for_depth,
        _GEN_FREQS,
    )

    depth = 3
    mask = eligible_monsters_for_depth(depth=depth)
    # Eligible entry with the highest gen_freq should dominate the histogram.
    eligible_freqs = jnp.where(mask, _GEN_FREQS, jnp.int32(0))
    top_freq = int(jnp.max(eligible_freqs))
    assert top_freq > 0, "no eligible monster has gen_freq > 0 at depth 3"
    top_idx = int(jnp.argmax(eligible_freqs))

    # Empirical histogram across 500 samples.
    counts = [0] * int(_GEN_FREQS.shape[0])
    rng = jax.random.PRNGKey(0)
    for i in range(500):
        rng, k = jax.random.split(rng)
        tid = int(pick_monster_for_level(k, depth))
        counts[tid] += 1

    top_freq_total = sum(
        counts[i] for i in range(len(counts))
        if int(_GEN_FREQS[i]) == top_freq and bool(mask[i])
    )
    # The bucket of "highest gen_freq" entries should claim notably more
    # than uniform.  Uniform across N eligible entries would give
    # 500 / N_eligible per slot; the top tier should clear that floor x2.
    n_elig = int(jnp.sum(mask.astype(jnp.int32)))
    uniform_per_slot = 500 // max(n_elig, 1)
    # number of top-freq slots
    n_top = int(jnp.sum(((eligible_freqs == top_freq) & mask).astype(jnp.int32)))
    uniform_expected = uniform_per_slot * n_top
    assert top_freq_total >= 2 * uniform_expected, (
        f"top gen_freq bucket only got {top_freq_total} hits "
        f"(expected >= {2 * uniform_expected} = 2x uniform across "
        f"{n_top} top slots in {n_elig} eligible monsters)"
    )


# ===========================================================================
# C. branches.py — graph vs dungeon.c::init_dungeons / dungeon.def
# ===========================================================================


def test_branch_graph_main_to_mines_at_depth_3():
    """Vendor: ``BRANCH: "The Gnomish Mines" @ (2, 3)`` — entry 2..(2+3)=5.

    We pick canonical mid-point 3.
    """
    from Nethax.nethax.dungeon.branches import init_branch_graph, Branch

    graph = init_branch_graph(jax.random.PRNGKey(0), None)
    dst_branch = int(graph.stair_links[Branch.MAIN, 3 - 1, 0])
    dst_level  = int(graph.stair_links[Branch.MAIN, 3 - 1, 1])
    assert dst_branch == int(Branch.GNOMISH_MINES), (
        f"Main Dlvl 3 should descend to Mines; got branch={dst_branch}"
    )
    assert dst_level == 1, f"Mines entry should be Dlvl 1, got {dst_level}"


def test_branch_graph_sokoban_at_depth_8():
    """Vendor: ``CHAINBRANCH "Sokoban" "oracle" + (1, 0) up`` -- Oracle
    range Dlvl 5..10, Sokoban one above → range 6..10, mid = 8."""
    from Nethax.nethax.dungeon.branches import init_branch_graph, Branch

    graph = init_branch_graph(jax.random.PRNGKey(0), None)
    dst_branch = int(graph.stair_links[Branch.MAIN, 8 - 1, 0])
    dst_level  = int(graph.stair_links[Branch.MAIN, 8 - 1, 1])
    assert dst_branch == int(Branch.SOKOBAN), (
        f"Main Dlvl 8 should descend to Sokoban; got branch={dst_branch}"
    )
    assert dst_level == 1


def test_branch_graph_quest_at_depth_14():
    """Vendor: ``CHAINBRANCH "The Quest" "oracle" + (6, 2) portal`` --
    Quest portal Dlvl ~ Oracle + 6 ±2 (12..16), mid 14 (XL14 gate)."""
    from Nethax.nethax.dungeon.branches import init_branch_graph, Branch

    graph = init_branch_graph(jax.random.PRNGKey(0), None)
    dst_branch = int(graph.stair_links[Branch.MAIN, 14 - 1, 0])
    dst_level  = int(graph.stair_links[Branch.MAIN, 14 - 1, 1])
    assert dst_branch == int(Branch.QUEST)
    assert dst_level == 1


def test_gehennom_below_castle():
    """Vendor: Castle = deepest Main level; Gehennom enters from Castle.

    From dungeon.def: ``LEVEL: "castle" "none" @ (-1, 0)`` (last Main level,
    Dlvl 26 in 3.6) and ``CHAINBRANCH: "Gehennom" "castle" + (0, 0) no_down``
    (Gehennom entrance sits at the Castle level).
    """
    from Nethax.nethax.dungeon.branches import init_branch_graph, Branch

    graph = init_branch_graph(jax.random.PRNGKey(0), None)
    # entry_dlvl[GEHENNOM] records the canonical Castle/Main-bottom depth.
    entry = int(graph.entry_dlvl[int(Branch.GEHENNOM)])
    assert entry == 26, (
        f"Gehennom should enter at Main Dlvl 26 (Castle); got {entry}"
    )
    # Main Dlvl 26 must have a down-link to Gehennom Dlvl 1.
    dst_branch = int(graph.stair_links[Branch.MAIN, 26 - 1, 0])
    dst_level  = int(graph.stair_links[Branch.MAIN, 26 - 1, 1])
    assert dst_branch == int(Branch.GEHENNOM), (
        f"Main Dlvl 26 should descend to Gehennom; got {dst_branch}"
    )
    assert dst_level == 1


def test_gehennom_has_16_internal_levels():
    """Vendor dungeon.def Gehennom block lists 16 levels (Valley + 15
    lairs + Sanctum).  Our BRANCH_TABLE encodes num_levels=16.
    """
    from Nethax.nethax.dungeon.branches import BRANCH_TABLE, Branch

    g_info = BRANCH_TABLE[int(Branch.GEHENNOM)]
    assert int(g_info.num_levels) == 16, (
        f"Gehennom should have 16 levels per vendor dungeon.def; "
        f"got {int(g_info.num_levels)}"
    )


def test_endgame_entered_from_sanctum():
    """Vendor: the Sanctum (deepest Gehennom level, L16) has the vibrating
    square portal to the Elemental Planes (Endgame branch).
    """
    from Nethax.nethax.dungeon.branches import (
        init_branch_graph,
        Branch,
    )

    graph = init_branch_graph(jax.random.PRNGKey(0), None)
    # Sanctum is Gehennom L16 (index 15).  It links into Endgame L1.
    dst_branch = int(graph.stair_links[Branch.GEHENNOM, 15, 0])
    dst_level  = int(graph.stair_links[Branch.GEHENNOM, 15, 1])
    assert dst_branch == int(Branch.ENDGAME), (
        f"Gehennom L16 (Sanctum) should portal into Endgame; got "
        f"branch={dst_branch}"
    )
    assert dst_level == 1
    # parent_branch[ENDGAME] should record Gehennom.
    assert int(graph.parent_branch[int(Branch.ENDGAME)]) == int(Branch.GEHENNOM)
