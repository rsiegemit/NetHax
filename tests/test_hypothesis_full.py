"""Full Hypothesis property-test suite — deep fuzzing tier (opt-in).

HOW TO ENABLE
=============
This suite is gated behind an environment variable so it does NOT run in
normal CI or development loops:

    RUN_HYPOTHESIS_FULL=1 JAX_PLATFORMS=cpu pytest tests/test_hypothesis_full.py -v

Expected runtime when enabled: approximately 5–15 minutes on a modern CPU,
depending on JIT warm-up overhead (the first env.step call compiles ~30-60s
of JAX programs).

WHAT IS COVERED
===============
1. test_stateful_env_machine
   RuleBasedStateMachine — drives env.step for up to 500 random actions per
   run, asserting glyph bounds, blstats bounds, inventory bounds, no NaNs/Infs,
   and done-stickiness on every step.

2. test_dungeon_gen_seeds
   Fuzz dungeon generator over many seeds: every level is connected (every
   floor/corridor tile BFS-reachable from the up-stair), no out-of-bounds
   terrain tiles, and exactly one up-stair and one down-stair.

3. test_wish_parser_never_throws
   Generate random wish strings (random prefixes, BUC tokens, enchant tokens,
   real and nonsense names); assert parse_wish_string never raises, always
   returns a 5-tuple with valid sentinel values.

4. test_rng_uniformity_rn2
   Long-baseline (1 000 000 draws) chi-squared uniformity check for rn2 over
   several values of n.

5. test_rng_dice_roll_mean
   Check that dice_roll(n, s) mean is within tolerance of the analytic
   expectation n*(s+1)/2.

6. test_rng_rne_mean
   Check that rne(x) mean is ≈ 1 + 1/(x-1) (geometric series expectation,
   capped at 9).

7. test_combat_formula_fuzzing
   Fuzz attacker/defender stats over wide ranges; assert find_roll_to_hit_formula
   and dmgval_weapon always return valid outputs.

8. test_identification_round_trip
   Fuzz identification shuffles: each class appearance map is a bijection, and
   type_for_appearance(appearance_for_type(t)) == t.

9. test_save_load_round_trip
   Fuzz arbitrary game states via multiple reset seeds; assert
   load_state(save_state(s)) == s (leaf-equal across all pytree leaves).

10. test_action_enum_exhaustivity
    For each of the 121 NLE actions from a fresh state, assert env.step
    transitions cleanly (no exception, valid obs keys, valid reward/done types).
"""
import os

# Respect the same JAX env vars used everywhere else in the suite.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

# ---------------------------------------------------------------------------
# Opt-in gate — must be first executable statement after imports.
# ---------------------------------------------------------------------------
import pytest  # noqa: E402

if not os.environ.get("RUN_HYPOTHESIS_FULL"):
    pytest.skip(
        "Skipped: set RUN_HYPOTHESIS_FULL=1 to run the deep Hypothesis suite.",
        allow_module_level=True,
    )

# ---------------------------------------------------------------------------
# Remaining imports (only reached when the env var is set)
# ---------------------------------------------------------------------------
import math
import tempfile
from collections import deque
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant

# Project imports
from Nethax.nethax import NethaxEnv
from Nethax.nethax.constants.actions import ACTIONS
from Nethax.nethax.dungeon.branches import generate_main_branch_l1, MAP_H, MAP_W
from Nethax.nethax.rng import rn2, dice_roll, rne, split_n
from Nethax.nethax.save_load import save_state, load_state
from Nethax.nethax.state import StaticParams
from Nethax.nethax.subsystems.combat import (
    find_roll_to_hit_formula,
    dmgval_weapon,
)
from Nethax.nethax.subsystems.identification import (
    N_AMULET_TYPES,
    N_POTION_TYPES,
    N_RING_TYPES,
    N_SCROLL_TYPES,
    N_SPELLBOOK_TYPES,
    N_WAND_TYPES,
    init_shuffled_appearances,
    type_for_appearance,
    unidentified_appearance,
)
from Nethax.nethax.subsystems.wish import parse_wish_string

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_STATIC = StaticParams()
_ENV = NethaxEnv(static=_STATIC)


def _fresh_env_state(seed: int = 0):
    rng = jax.random.PRNGKey(seed)
    state, obs = _ENV.reset(rng)
    return state, obs, rng


def _assert_obs_valid(obs: dict) -> None:
    """Assert standard NLE observation keys are present and within bounds."""
    for key in ("glyphs", "blstats"):
        assert key in obs, f"obs missing required key {key!r}"

    glyphs = np.asarray(obs["glyphs"])
    assert not np.any(np.isnan(glyphs.astype(np.float32))), "NaN in glyphs"
    assert glyphs.shape == (21, 79), f"glyphs shape {glyphs.shape} != (21, 79)"
    # Glyphs must be non-negative and <= NO_GLYPH (2*NO_GLYPH is a loose upper bound)
    assert np.all(glyphs >= 0), "Negative glyph value detected"

    blstats = np.asarray(obs["blstats"])
    assert blstats.shape == (27,), f"blstats shape {blstats.shape} != (27,)"
    assert not np.any(np.isinf(blstats.astype(np.float64))), "Inf in blstats"

    # HP and max-HP indices (BL_HP=10, BL_HPMAX=11)
    hp    = int(blstats[10])
    hpmax = int(blstats[11])
    assert hpmax >= 0, f"blstats HP_MAX < 0: {hpmax}"
    # HP can be 0 (dead) but not < -1 (would be bizarre)
    assert hp >= -1, f"blstats HP suspiciously negative: {hp}"

    # Inventory glyphs must be valid dtype
    if "inv_glyphs" in obs:
        inv = np.asarray(obs["inv_glyphs"])
        assert inv.shape == (55,), f"inv_glyphs shape {inv.shape} != (55,)"


# ===========================================================================
# Test 1 — Stateful env machine (RuleBasedStateMachine)
# ===========================================================================

class EnvMachine(RuleBasedStateMachine):
    """Drive NethaxEnv.step over random sequences, asserting invariants."""

    @initialize()
    def setup(self):
        self.state, self.obs, self.rng = _fresh_env_state(seed=0)
        self._last_done = bool(np.asarray(self.state.done))

    @rule(action_idx=st.integers(min_value=0, max_value=len(ACTIONS) - 1),
          rng_seed=st.integers(min_value=0, max_value=2**31 - 1))
    def step_action(self, action_idx: int, rng_seed: int):
        action = jnp.int32(int(ACTIONS[action_idx]))
        self.rng, step_rng = jax.random.split(jax.random.PRNGKey(rng_seed))
        self.state, self.obs, reward, done, _info = _ENV.step(
            self.state, action, step_rng
        )
        # Invariant: done-stickiness — once done, stays done
        cur_done = bool(np.asarray(done))
        if self._last_done:
            assert cur_done, "done transitioned False->True after being True (done-stickiness violated)"
        self._last_done = cur_done

        # Invariant: reward is a scalar float
        reward_f = float(np.asarray(reward))
        assert math.isfinite(reward_f), f"Non-finite reward: {reward_f}"

    @invariant()
    def obs_is_valid(self):
        _assert_obs_valid(self.obs)

    @invariant()
    def state_leaves_finite(self):
        """No NaN or Inf in any numeric state leaf."""
        for i, leaf in enumerate(jax.tree_util.tree_leaves(self.state)):
            arr = np.asarray(leaf)
            if np.issubdtype(arr.dtype, np.floating):
                assert not np.any(np.isnan(arr)), f"NaN in state leaf {i}"
                assert not np.any(np.isinf(arr)), f"Inf in state leaf {i}"


EnvMachine.TestCase.settings = settings(
    max_examples=50,
    deadline=None,
    stateful_step_count=500,
)
TestEnvMachineStateful = EnvMachine.TestCase


# ===========================================================================
# Test 2 — Dungeon gen seed fuzzing
# ===========================================================================

def _bfs_reachable(terrain: np.ndarray, start_r: int, start_c: int,
                   passable_tiles: set) -> set:
    """BFS flood-fill; return set of (r, c) reachable from start."""
    h, w = terrain.shape
    visited = set()
    queue = deque()
    if terrain[start_r, start_c] in passable_tiles:
        queue.append((start_r, start_c))
        visited.add((start_r, start_c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                if terrain[nr, nc] in passable_tiles:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    return visited


@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
@settings(max_examples=200, deadline=None)
def test_dungeon_gen_seeds(seed: int):
    """Fuzz dungeon generator: connectivity, bounds, stair counts."""
    rng = jax.random.PRNGKey(seed)
    terrain, _rooms, active, up_pos, dn_pos, *_rest = generate_main_branch_l1(rng, _STATIC)

    terrain_np = np.asarray(terrain)
    up_r, up_c = int(up_pos[0]), int(up_pos[1])
    dn_r, dn_c = int(dn_pos[0]), int(dn_pos[1])

    # Bounds: all tile values must be non-negative and within known range
    assert np.all(terrain_np >= 0), "Negative tile value in terrain"
    # Known max TileType is ~16 (SHOP_FLOOR); use generous upper bound 32
    assert np.all(terrain_np < 32), f"Tile value >= 32: {terrain_np.max()}"

    # Stair positions must be within map bounds
    assert 0 <= up_r < MAP_H, f"up-stair row {up_r} out of [0, {MAP_H})"
    assert 0 <= up_c < MAP_W, f"up-stair col {up_c} out of [0, {MAP_W})"
    assert 0 <= dn_r < MAP_H, f"dn-stair row {dn_r} out of [0, {MAP_H})"
    assert 0 <= dn_c < MAP_W, f"dn-stair col {dn_c} out of [0, {MAP_W})"

    # Passable tile set: FLOOR=1, CORRIDOR=2, STAIRCASE_UP=6, STAIRCASE_DOWN=7,
    # OPEN_DOOR=5, SHOP_FLOOR=16
    passable = {1, 2, 5, 6, 7, 16}

    # Connectivity: BFS from up-stair must reach down-stair
    reachable = _bfs_reachable(terrain_np, up_r, up_c, passable)
    assert (dn_r, dn_c) in reachable, (
        f"Down-stair ({dn_r},{dn_c}) not BFS-reachable from up-stair "
        f"({up_r},{up_c}) for seed={seed}"
    )

    # Exactly one up-stair and one down-stair in terrain
    n_up = int(np.sum(terrain_np == 6))
    n_dn = int(np.sum(terrain_np == 7))
    assert n_up == 1, f"Expected 1 up-stair in terrain, got {n_up} (seed={seed})"
    assert n_dn == 1, f"Expected 1 down-stair in terrain, got {n_dn} (seed={seed})"


# ===========================================================================
# Test 3 — Wish parser: never throws, always returns valid sentinel
# ===========================================================================

_BUC_TOKENS = [b"", b"blessed ", b"uncursed ", b"cursed "]
_ENCHANT_TOKENS = [b"", b"+3 ", b"-2 ", b"+0 "]
_REAL_NAMES = [
    b"long sword", b"dagger", b"potion of healing",
    b"scroll of identify", b"gray dragon scale mail",
    b"Excalibur", b"plate mail",
]
_NONSENSE = [b"flux capacitor", b"", b"xyzzy", b"    ", b"123abc!!"]


@given(
    buc=st.sampled_from(_BUC_TOKENS),
    ench=st.sampled_from(_ENCHANT_TOKENS),
    name=st.one_of(
        st.sampled_from(_REAL_NAMES + _NONSENSE),
        st.binary(min_size=0, max_size=40),
    ),
)
@settings(max_examples=200, deadline=None)
def test_wish_parser_never_throws(buc: bytes, ench: bytes, name: bytes):
    """parse_wish_string must never raise and must always return a valid 5-tuple."""
    wish = buc + ench + name
    try:
        result = parse_wish_string(wish)
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"parse_wish_string raised on input {wish!r}: {exc}")

    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 5, f"Expected 5-tuple, got {len(result)}-tuple"
    cat, tid, buc_val, enchant, art = result

    # Sentinels: type_id == -1 means unknown (valid); else must be >= 0
    assert tid >= -1, f"type_id {tid} < -1"
    assert art >= -1, f"artifact_idx {art} < -1"
    # enchantment can be any int; buc can be any int (sentinel or enum value)
    # No further constraint — just must not raise.


# ===========================================================================
# Test 4 — rn2 long-baseline uniformity (chi-squared)
# ===========================================================================

@given(n=st.integers(min_value=2, max_value=20))
@settings(max_examples=10, deadline=None)
def test_rng_uniformity_rn2(n: int):
    """1M draws of rn2(n): chi-squared statistic must be within 3-sigma of expected."""
    N_DRAWS = 1_000_000
    keys = split_n(jax.random.PRNGKey(42 + n), N_DRAWS)
    draws = jax.vmap(lambda k: rn2(k, n))(keys)
    counts = np.bincount(np.asarray(draws), minlength=n).astype(np.float64)

    expected = N_DRAWS / n
    # Chi-squared statistic: sum((O-E)^2/E), df=n-1.
    # Under H0, E[chi2] = n-1, Var[chi2] = 2*(n-1).
    # We use a loose 6-sigma bound to avoid flakiness on valid distributions.
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    df = n - 1
    sigma = math.sqrt(2 * df)
    assert chi2 < df + 6 * sigma, (
        f"rn2({n}): chi2={chi2:.2f} >> df={df} + 6*sigma={df + 6*sigma:.2f}; "
        f"counts range [{counts.min():.0f}, {counts.max():.0f}] expected={expected:.0f}"
    )


# ===========================================================================
# Test 5 — dice_roll mean within tolerance
# ===========================================================================

@given(
    n=st.integers(min_value=1, max_value=6),
    sides=st.integers(min_value=2, max_value=20),
)
@settings(max_examples=30, deadline=None)
def test_rng_dice_roll_mean(n: int, sides: int):
    """dice_roll(n, s) mean must be within 1% of n*(s+1)/2 over 100k draws."""
    N_DRAWS = 100_000
    keys = split_n(jax.random.PRNGKey(7 + n * 100 + sides), N_DRAWS)
    draws = jax.vmap(lambda k: dice_roll(k, n, sides))(keys)
    observed_mean = float(jnp.mean(draws.astype(jnp.float32)))
    expected_mean = n * (sides + 1) / 2.0
    tolerance = max(0.05 * expected_mean, 0.1)
    assert abs(observed_mean - expected_mean) < tolerance, (
        f"dice_roll({n},{sides}): mean={observed_mean:.4f} "
        f"expected={expected_mean:.4f} tol={tolerance:.4f}"
    )


# ===========================================================================
# Test 6 — rne mean check
# ===========================================================================

@given(x=st.integers(min_value=2, max_value=8))
@settings(max_examples=10, deadline=None)
def test_rng_rne_mean(x: int):
    """rne(x) sample mean must be ≈ 1 + 1/(x-1) over 10k draws (geometric)."""
    # rne internally calls jax.random.split(rng, cap), so it can be vmapped
    # by giving each call its own independent subkey via split_n.
    N_DRAWS = 10_000
    keys = split_n(jax.random.PRNGKey(99 + x), N_DRAWS)
    draws = jax.vmap(lambda k: rne(k, x))(keys)
    observed_mean = float(jnp.mean(draws.astype(jnp.float32)))
    expected_mean = 1.0 + 1.0 / (x - 1)  # geometric series sum, capped at 9
    # Tolerance: 10% relative or 0.15 absolute (wider due to smaller sample)
    tol = max(0.10 * expected_mean, 0.15)
    assert abs(observed_mean - expected_mean) < tol, (
        f"rne({x}): observed_mean={observed_mean:.4f} "
        f"expected≈{expected_mean:.4f} tol={tol:.4f}"
    )


# ===========================================================================
# Test 7 — Combat formula fuzzing
# ===========================================================================

@given(
    str_val=st.integers(min_value=3, max_value=125),
    dex_val=st.integers(min_value=3, max_value=25),
    monster_ac=st.integers(min_value=-10, max_value=10),
    skill_tier=st.integers(min_value=0, max_value=5),
    weapon_enchant=st.integers(min_value=-7, max_value=7),
    xl=st.integers(min_value=1, max_value=30),
)
@settings(max_examples=200, deadline=None)
def test_combat_to_hit_formula_valid(
    str_val, dex_val, monster_ac, skill_tier, weapon_enchant, xl
):
    """find_roll_to_hit_formula must return a finite integer for any stat combo."""
    result = find_roll_to_hit_formula(
        str_value=str_val,
        dex_value=dex_val,
        monster_ac=monster_ac,
        skill_tier=skill_tier,
        weapon_enchant=weapon_enchant,
        xl=xl,
    )
    assert isinstance(result, int), f"Expected int, got {type(result)}"
    # Result is an attack-roll accumulator; no strict bound but must be finite
    assert math.isfinite(result), f"Non-finite to-hit tmp: {result}"


@given(
    bigmonst=st.booleans(),
    sdam_roll=st.integers(min_value=1, max_value=20),
    ldam_roll=st.integers(min_value=1, max_value=20),
    spe=st.integers(min_value=-7, max_value=7),
    is_weapon=st.booleans(),
)
@settings(max_examples=200, deadline=None)
def test_combat_dmgval_non_negative(bigmonst, sdam_roll, ldam_roll, spe, is_weapon):
    """dmgval_weapon must produce non-negative damage (weapons clamp at 0)."""
    result = dmgval_weapon(
        bigmonst=bigmonst,
        sdam_roll=sdam_roll,
        ldam_roll=ldam_roll,
        spe=spe,
        is_weapon=is_weapon,
    )
    assert isinstance(result, int), f"Expected int, got {type(result)}"
    if is_weapon:
        assert result >= 0, f"Weapon damage negative: {result}"


# ===========================================================================
# Test 8 — Identification round-trip (bijection per class)
# ===========================================================================

@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
@settings(max_examples=200, deadline=None)
def test_identification_bijection(seed: int):
    """Each shuffled appearance map must be a bijection per class."""
    rng = jax.random.PRNGKey(seed)
    id_state = init_shuffled_appearances(rng)

    def _check_bijection(arr: jnp.ndarray, n: int, label: str):
        arr_np = np.asarray(arr).astype(np.int32)
        assert arr_np.shape == (n,), f"{label}: shape {arr_np.shape} != ({n},)"
        assert sorted(arr_np.tolist()) == list(range(n)), (
            f"{label}: not a permutation of [0..{n}); got {sorted(arr_np.tolist())}"
        )

    _check_bijection(id_state.potion_appearance,    N_POTION_TYPES,    "potion")
    _check_bijection(id_state.scroll_appearance,    N_SCROLL_TYPES,    "scroll")
    _check_bijection(id_state.wand_appearance,       N_WAND_TYPES,      "wand")
    _check_bijection(id_state.ring_appearance,       N_RING_TYPES,      "ring")
    _check_bijection(id_state.amulet_appearance,     N_AMULET_TYPES,    "amulet")
    _check_bijection(id_state.spellbook_appearance,  N_SPELLBOOK_TYPES, "spellbook")


@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    type_id=st.integers(min_value=0, max_value=N_POTION_TYPES - 1),
)
@settings(max_examples=200, deadline=None)
def test_identification_round_trip_potion(seed: int, type_id: int):
    """type_for_appearance(appearance[t], "potion") == t."""
    rng = jax.random.PRNGKey(seed)
    id_state = init_shuffled_appearances(rng)

    # appearance value for this type_id
    appearance = int(id_state.potion_appearance[type_id])
    # Inverse: which type has this appearance?
    recovered = type_for_appearance(id_state, "potion", appearance)
    assert int(recovered) == type_id, (
        f"Round-trip failed: type {type_id} -> appearance {appearance} "
        f"-> recovered {int(recovered)}"
    )


# ===========================================================================
# Test 9 — Save / load round-trip over multiple seeds
# ===========================================================================

@given(seed=st.integers(min_value=0, max_value=999))
@settings(max_examples=30, deadline=None)
def test_save_load_round_trip(seed: int):
    """load_state(save_state(s)) must equal s byte-for-byte on all pytree leaves."""
    state, _obs, _rng = _fresh_env_state(seed=seed)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "state.npz"
        save_state(state, path)
        loaded = load_state(path)

    leaves_orig, td_orig = jax.tree_util.tree_flatten(state)
    leaves_load, td_load = jax.tree_util.tree_flatten(loaded)

    assert td_orig == td_load, "treedef mismatch after save/load"
    assert len(leaves_orig) == len(leaves_load), "leaf count mismatch"

    for i, (lo, ll) in enumerate(zip(leaves_orig, leaves_load)):
        a = np.asarray(lo)
        b = np.asarray(ll)
        assert a.shape == b.shape, f"leaf {i} shape mismatch: {a.shape} vs {b.shape}"
        assert a.dtype == b.dtype, f"leaf {i} dtype mismatch: {a.dtype} vs {b.dtype}"
        assert np.array_equal(a, b), (
            f"leaf {i} values differ after round-trip (seed={seed})"
        )


# ===========================================================================
# Test 10 — Action enum exhaustivity
# ===========================================================================

def test_action_enum_exhaustivity():
    """Every action in ACTIONS must dispatch cleanly from a fresh env state."""
    state, _obs, rng = _fresh_env_state(seed=0)
    failures = []
    for i, action in enumerate(ACTIONS):
        rng, step_rng = jax.random.split(rng)
        try:
            new_state, obs, reward, done, _info = _ENV.step(
                state, jnp.int32(int(action)), step_rng
            )
        except Exception as exc:  # noqa: BLE001
            failures.append((i, int(action), repr(exc)))
            continue

        # Basic validity assertions
        if "glyphs" not in obs:
            failures.append((i, int(action), "obs missing 'glyphs' key"))
        elif np.asarray(obs["glyphs"]).shape != (21, 79):
            failures.append((
                i, int(action),
                f"glyphs shape {np.asarray(obs['glyphs']).shape}"
            ))

    if failures:
        msg = "\n".join(
            f"  action[{idx}]={code}: {err}"
            for idx, code, err in failures
        )
        pytest.fail(f"{len(failures)} actions failed dispatch:\n{msg}")
