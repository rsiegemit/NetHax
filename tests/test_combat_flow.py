"""Wave 3 integration tests — combat flow across subsystems.

Tests verify that combat interactions (player attacks monster, monster
attacks player, armor damage reduction) produce correct state changes.

All Wave 3 combat logic is implemented by parallel agents.  Each test is
guarded with a skipif when the required feature is still a stub (returns
zero damage / no XP change).

All imports are lazy so collection never fails.
"""

import pytest


def _make_env_with_monster(monster_hp=10, monster_dice_n=1, monster_dice_sides=4,
                            monster_ac=10, adjacent=True, player_hp=10):
    """Helper: reset env and inject a live monster adjacent to the player.

    Returns (env, state, rng, monster_idx=0).
    """
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.env import NethaxEnv
    from Nethax.nethax.constants import TileType

    rng = jax.random.PRNGKey(42)
    env = NethaxEnv()
    state, _ = env.reset(rng)

    # Ensure player HP is set.  Also lift player_hp_max so high ``player_hp``
    # values aren't clamped by the next regen tick
    # (vendor allmain.c::regen_hp clips at u.uhpmax).
    state = state.replace(
        player_hp=jnp.int32(player_hp),
        player_hp_max=jnp.int32(max(int(state.player_hp_max), player_hp)),
    )

    # Place monster at player_pos + (0, 1) — one tile east
    p_row = int(state.player_pos[0])
    p_col = int(state.player_pos[1])
    m_row, m_col = p_row, p_col + 1

    # Carve floor tiles around player so movement/combat can occur
    branch = int(state.dungeon.current_branch)
    level_idx = int(state.dungeon.current_level) - 1
    new_terrain = state.terrain
    for c in range(max(0, p_col - 1), min(state.terrain.shape[3], p_col + 3)):
        new_terrain = new_terrain.at[branch, level_idx, p_row, c].set(
            jnp.int8(TileType.FLOOR)
        )
    state = state.replace(terrain=new_terrain)

    # Inject monster into slot 0 of monster_ai.  Reset all slots to dead
    # first so spawn-generated NPCs don't also attack the player.
    monster_ai = state.monster_ai
    n_slots = monster_ai.alive.shape[0]
    monster_ai = monster_ai.replace(
        alive=jnp.zeros((n_slots,), dtype=monster_ai.alive.dtype),
    )
    monster_ai = monster_ai.replace(
        alive=monster_ai.alive.at[0].set(True),
        hp=monster_ai.hp.at[0].set(jnp.int32(monster_hp)),
        hp_max=monster_ai.hp_max.at[0].set(jnp.int32(monster_hp)),
        pos=monster_ai.pos.at[0].set(
            jnp.array([m_row, m_col], dtype=jnp.int16)
        ),
        ac=monster_ai.ac.at[0].set(jnp.int8(monster_ac)),
        attack_dice_n=monster_ai.attack_dice_n.at[0].set(
            jnp.int8(monster_dice_n)
        ),
        attack_dice_sides=monster_ai.attack_dice_sides.at[0].set(
            jnp.int8(monster_dice_sides)
        ),
    )
    state = state.replace(monster_ai=monster_ai)

    return env, state, rng


def test_kill_monster_grants_xp():
    """Bump-attack a monster until it dies; assert player_xp increased.

    Wave 5: action_dispatch._try_step now routes movement into a
    monster-occupied tile through combat.melee_attack and grants XP on
    kill.
    """
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.constants.actions import CompassCardinalDirection

    env, state, rng = _make_env_with_monster(monster_hp=5)
    xp_before = int(state.player_xp)

    # Move east repeatedly to bump-attack the monster
    action = jnp.int32(int(CompassCardinalDirection.E))
    for _ in range(20):
        rng, step_rng = jax.random.split(rng)
        state, _, _, done, _ = env.step(state, action, step_rng)
        if bool(done):
            break
        # Check if monster is dead
        if not bool(state.monster_ai.alive[0]):
            break

    assert not bool(state.monster_ai.alive[0]), (
        "Monster should be dead after repeated attacks"
    )
    assert int(state.player_xp) > xp_before, (
        f"Expected player_xp > {xp_before}, got {int(state.player_xp)}"
    )


def test_monster_kills_player():
    """Low-HP player adjacent to high-damage monster; step until done=True."""
    import jax
    import jax.numpy as jnp

    # 1 HP player, monster does 1d10 (guaranteed kill)
    env, state, rng = _make_env_with_monster(
        monster_hp=100,
        monster_dice_n=1,
        monster_dice_sides=10,
        player_hp=1,
    )

    action = jnp.int32(ord("."))  # wait — let monster attack
    done = False
    for _ in range(10):
        rng, step_rng = jax.random.split(rng)
        state, _, _, done_arr, _ = env.step(state, action, step_rng)
        done = bool(done_arr)
        if done:
            break

    assert done, "Expected done=True after low-HP player is killed by monster"
    assert int(state.player_hp) <= 0, (
        f"Expected player_hp <= 0, got {int(state.player_hp)}"
    )


@pytest.mark.timeout(300)
def test_armor_reduces_damage():
    """Same monster attacks player with and without armor; compare HP loss.

    Vendor cite: mhitu.c:709-718 — to-hit formula
        tmp = AC_VALUE(u.uac) + 10 + mlev
        hit_iff tmp > rnd(20)
    Lower player AC = lower ``tmp`` = fewer successful hits.

    Wave 6 parity-fix: monster_hp is kept SMALL so the approximate
    ``mlev = hp_max/4`` is small enough that the AC delta drives a visible
    difference in hit-rate.  With monster_hp=100, mlev≈25 makes
    ``tmp = 10+10+25 = 45 > rnd(20)`` always true regardless of armor.
    Many trials are aggregated so a single seed's run doesn't dominate.

    Wave34e perf fix: JIT-compile env.step once outside the trial loop.
    Without JIT, every call retraces the full step graph and 512 retraces
    blow past the 120s pytest timeout.  One compile + many fast calls
    finishes in well under 5 min.
    """
    import jax
    import jax.numpy as jnp

    # Aggregate over multiple RNG seeds so the comparison is statistically
    # meaningful even with the d20 hit gate.
    n_trials = 32
    n_turns = 8
    bare_total = 0
    armor_total = 0
    # Build env once and JIT-compile step — bare/armored states share the
    # same pytree structure so a single jit caches across both branches.
    env_b, _, _ = _make_env_with_monster(
        monster_hp=4, monster_dice_n=1, monster_dice_sides=6, player_hp=200
    )
    jstep = jax.jit(env_b.step)
    action = jnp.int32(ord("."))
    for seed in range(n_trials):
        # Bare
        _, state_b, _ = _make_env_with_monster(
            monster_hp=4, monster_dice_n=1, monster_dice_sides=6, player_hp=200
        )
        rng_b = jax.random.PRNGKey(seed)
        for _ in range(n_turns):
            rng_b, step_rng = jax.random.split(rng_b)
            state_b, _, _, _, _ = jstep(state_b, action, step_rng)
        bare_total += 200 - int(state_b.player_hp)

        # Armored: AC bonus +5 in slot 0 — large enough to dominate noise.
        _, state_a, _ = _make_env_with_monster(
            monster_hp=4, monster_dice_n=1, monster_dice_sides=6, player_hp=200
        )
        inv = state_a.inventory
        new_cache = inv.worn_armor_ac_bonus.at[0].set(jnp.int8(5))
        state_a = state_a.replace(inventory=inv.replace(worn_armor_ac_bonus=new_cache))
        rng_a = jax.random.PRNGKey(seed)
        for _ in range(n_turns):
            rng_a, step_rng = jax.random.split(rng_a)
            state_a, _, _, _, _ = jstep(state_a, action, step_rng)
        armor_total += 200 - int(state_a.player_hp)

    assert armor_total < bare_total, (
        f"Armor should reduce damage in aggregate: "
        f"bare_total={bare_total}, armored_total={armor_total} "
        f"(across {n_trials} seeds, {n_turns} turns each)"
    )
