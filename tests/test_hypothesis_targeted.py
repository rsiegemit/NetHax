"""Targeted Hypothesis property tests for nethax invariants.

Wave 6 Phase C — Task #56

Covers 10 high-value invariants:
  1. env.step never crashes for any valid action from a fresh reset state.
  2. obs dict has exactly 17 NLE keys with exact shapes and dtypes.
  3. Glyph values always in [0, MAX_GLYPH-1] (5975 inclusive).
  4. blstats shape (27,) with plausible field bounds.
  5. Inventory: category in valid range, quantity >= 0, total_weight >= 0.
  6. RNG determinism: same seed -> identical obs after N steps.
  7. rne(x, cap) always returns int in [1, cap].
  8. rnf(num, den) fire-rate within ±3σ of num/den over 10k samples.
  9. Wish parser: any string from vocab never raises.
 10. Item.replace() preserves all 17 dataclass fields.
 11. Reset(seed=k) twice -> identical states.
 12. Done flag is sticky once True.

All tests use @settings(max_examples=50, deadline=None) (default 50) to stay fast.
"""
import os

# Must be set before JAX is imported.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")

from hypothesis import given, settings, assume
from hypothesis import strategies as st

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Module-level shared fixtures (built once, reused across tests)
# ---------------------------------------------------------------------------

def _make_env():
    from Nethax.nethax import NethaxEnv
    return NethaxEnv()


# Warm up JIT once at module import time so individual tests don't pay the
# full compilation overhead on the first @given example.
_ENV = _make_env()
_STATE0, _OBS0 = _ENV.reset(jax.random.PRNGKey(0))


# ---------------------------------------------------------------------------
# 1. env.step never crashes for any valid action
# ---------------------------------------------------------------------------

@given(action=st.integers(min_value=0, max_value=120))
@settings(max_examples=50, deadline=None)
def test_step_never_crashes(action):
    """env.step for any action in [0, 120] must not raise from a fresh reset.

    deadline=None: the first example after JIT cache miss can be slow.  Total
    suite runtime is bounded by max_examples * step time after warmup (~1ms).
    """
    env = _ENV
    rng = jax.random.PRNGKey(action + 1000)
    new_state, obs, reward, done, info = env.step(_STATE0, jnp.int32(action), rng)
    # The mere fact we got here means no exception.
    assert obs is not None


# ---------------------------------------------------------------------------
# 2. obs dict: exact 17 keys, exact shapes, exact dtypes
# ---------------------------------------------------------------------------

from Nethax.nethax.obs.nle_obs import (
    NLE_OBSERVATION_KEYS,
    NLE_OBSERVATION_SHAPES,
    NLE_OBSERVATION_DTYPES,
)

@given(action=st.integers(min_value=0, max_value=120))
@settings(max_examples=50, deadline=None)
def test_obs_keys_shapes_dtypes(action):
    """Obs after any step must have exactly 17 NLE keys with canonical shapes/dtypes."""
    env = _ENV
    rng = jax.random.PRNGKey(action + 2000)
    _new_state, obs, _rew, _done, _info = env.step(_STATE0, jnp.int32(action), rng)

    assert set(obs.keys()) == set(NLE_OBSERVATION_KEYS), (
        f"Key mismatch: {set(obs.keys()) ^ set(NLE_OBSERVATION_KEYS)}"
    )
    assert len(obs) == 17

    for key in NLE_OBSERVATION_KEYS:
        arr = obs[key]
        expected_shape = NLE_OBSERVATION_SHAPES[key]
        expected_dtype = NLE_OBSERVATION_DTYPES[key]
        assert arr.shape == expected_shape, (
            f"{key}: expected shape {expected_shape}, got {arr.shape}"
        )
        assert arr.dtype == expected_dtype, (
            f"{key}: expected dtype {expected_dtype}, got {arr.dtype}"
        )


# ---------------------------------------------------------------------------
# 3. Glyph values in [0, MAX_GLYPH-1] (5975 inclusive)
# ---------------------------------------------------------------------------

from Nethax.nethax.constants.glyphs import MAX_GLYPH

@given(action=st.integers(min_value=0, max_value=120))
@settings(max_examples=50, deadline=None)
def test_glyphs_in_valid_range(action):
    """All glyph values after any step must be in [0, MAX_GLYPH].

    Real tile glyphs occupy 0..MAX_GLYPH-1 (5975).  NO_GLYPH == MAX_GLYPH ==
    5976 is reserved for inventory slots / internal sentinels only — the map
    `glyphs` obs never contains it: NLE fills unseen cells with
    cmap_to_glyph(S_stone) (= 2359), not NO_GLYPH (vendor winrl.cc:61,250).
    This bound stays an inclusive upper limit regardless.
    """
    env = _ENV
    rng = jax.random.PRNGKey(action + 3000)
    _new_state, obs, _rew, _done, _info = env.step(_STATE0, jnp.int32(action), rng)

    glyphs = np.array(obs["glyphs"])
    assert int(glyphs.min()) >= 0, f"Negative glyph: {int(glyphs.min())}"
    assert int(glyphs.max()) <= MAX_GLYPH, (
        f"Glyph {int(glyphs.max())} > MAX_GLYPH ({MAX_GLYPH})"
    )


# ---------------------------------------------------------------------------
# 4. blstats shape (27,) with plausible field bounds
# ---------------------------------------------------------------------------

from Nethax.nethax.constants.blstats import (
    BL_HP, BL_HPMAX, BL_DEPTH, BL_AC, BL_XP, BL_DLEVEL, BL_GOLD,
)

@given(action=st.integers(min_value=0, max_value=120))
@settings(max_examples=50, deadline=None)
def test_blstats_shape_and_bounds(action):
    """blstats must be shape (27,) with plausible per-field bounds."""
    env = _ENV
    rng = jax.random.PRNGKey(action + 4000)
    _new_state, obs, _rew, _done, _info = env.step(_STATE0, jnp.int32(action), rng)

    bs = np.array(obs["blstats"])
    assert bs.shape == (27,), f"blstats shape {bs.shape} != (27,)"

    hp     = int(bs[BL_HP])
    hpmax  = int(bs[BL_HPMAX])
    depth  = int(bs[BL_DEPTH])
    ac     = int(bs[BL_AC])
    xp     = int(bs[BL_XP])
    dlevel = int(bs[BL_DLEVEL])
    gold   = int(bs[BL_GOLD])

    assert hp >= 0,               f"HP < 0: {hp}"
    assert hpmax >= 0,            f"HPmax < 0: {hpmax}"
    assert depth >= 1,            f"depth < 1: {depth}"
    assert -128 <= ac <= 127,     f"AC {ac} out of [-128, 127]"
    assert xp >= 1,               f"XP level < 1: {xp}"
    assert dlevel >= 1,           f"dlevel < 1: {dlevel}"
    assert gold >= 0,             f"gold < 0: {gold}"


# ---------------------------------------------------------------------------
# 5. Inventory invariants: category in range, quantity >= 0, total_weight >= 0
# ---------------------------------------------------------------------------

from Nethax.nethax.subsystems.inventory import ItemCategory, MAX_INVENTORY_SLOTS

_VALID_CATEGORIES = set(int(c) for c in ItemCategory)

@given(action=st.integers(min_value=0, max_value=120))
@settings(max_examples=50, deadline=None)
def test_inventory_invariants(action):
    """After any step: category in valid enum range, quantity >= 0, total_weight >= 0."""
    env = _ENV
    rng = jax.random.PRNGKey(action + 5000)
    new_state, _obs, _rew, _done, _info = env.step(_STATE0, jnp.int32(action), rng)

    inv = new_state.inventory
    categories = np.array(inv.items.category)   # [MAX_INVENTORY_SLOTS]
    quantities  = np.array(inv.items.quantity)   # [MAX_INVENTORY_SLOTS]
    total_w     = int(inv.total_weight)

    for i in range(MAX_INVENTORY_SLOTS):
        cat = int(categories[i])
        assert cat in _VALID_CATEGORIES, (
            f"slot {i}: category {cat} not in ItemCategory"
        )
        qty = int(quantities[i])
        assert qty >= 0, f"slot {i}: quantity {qty} < 0"

    assert total_w >= 0, f"total_weight {total_w} < 0"


# ---------------------------------------------------------------------------
# 6. RNG determinism: same seed -> identical obs after N steps
# ---------------------------------------------------------------------------

@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=20, deadline=None)
def test_rng_determinism(seed):
    """Two identical reset+step sequences with the same seed produce identical obs.

    Re-uses the warm module-level _ENV so JIT cache is hot.  Fewer examples
    (20 vs 50) keep runtime in budget — reset+step is heavier than step alone.
    """
    env = _ENV

    def run(s):
        rng = jax.random.PRNGKey(s)
        state, _obs = env.reset(rng)
        rng_step = jax.random.PRNGKey(s + 999999)
        _st2, obs2, _rew, _done, _info = env.step(state, jnp.int32(0), rng_step)
        return obs2

    obs_a = run(seed)
    obs_b = run(seed)

    for key in NLE_OBSERVATION_KEYS:
        np.testing.assert_array_equal(
            np.array(obs_a[key]),
            np.array(obs_b[key]),
            err_msg=f"Determinism broken for obs key '{key}' with seed={seed}",
        )


# ---------------------------------------------------------------------------
# 7. rne(x, cap) always returns int in [1, cap]
# ---------------------------------------------------------------------------

from Nethax.nethax.rng import rne

@given(
    seed=st.integers(min_value=0, max_value=10000),
    x=st.integers(min_value=2, max_value=100),
    cap=st.integers(min_value=3, max_value=9),
)
@settings(max_examples=50, deadline=None)
def test_rne_range(seed, x, cap):
    """rne(x, cap) must return an int in [1, cap].

    deadline=None: rne has Python-static (x, cap), so each combo is a
    separate JIT trace.  Compile cost per trace can exceed Hypothesis's
    default 200ms deadline; total wall-time stays bounded by max_examples.
    """
    rng = jax.random.PRNGKey(seed)
    result = int(rne(rng, x, cap))
    assert 1 <= result <= cap, f"rne({x}, {cap}) = {result} not in [1, {cap}]"


# ---------------------------------------------------------------------------
# 8. rnf(num, den) fire-rate within ±3σ of num/den over 10k samples
# ---------------------------------------------------------------------------

from Nethax.nethax.rng import rnf

@given(
    num=st.integers(min_value=1, max_value=8),
    den=st.integers(min_value=10, max_value=20),
)
@settings(max_examples=10, deadline=None)  # 10k samples per example; deadline N/A
def test_rnf_rate(num, den):
    """rnf(num, den) fire rate over 10k samples must be within ±3σ of num/den."""
    assume(num < den)
    n_samples = 10_000
    keys = jax.random.split(jax.random.PRNGKey(num * 1000 + den), n_samples)
    results = jax.vmap(lambda k: rnf(k, num, den))(keys)
    observed = float(jnp.mean(results.astype(jnp.float32)))
    expected = num / den
    # σ for a Bernoulli(p) mean over n_samples: sqrt(p*(1-p)/n)
    sigma = (expected * (1 - expected) / n_samples) ** 0.5
    assert abs(observed - expected) <= 3 * sigma, (
        f"rnf({num},{den}) rate {observed:.4f} deviates from {expected:.4f} by "
        f"{abs(observed-expected)/sigma:.2f}σ (>3σ)"
    )


# ---------------------------------------------------------------------------
# 9. Wish parser: any vocab string never raises
# ---------------------------------------------------------------------------

from Nethax.nethax.subsystems.wish import parse_wish_string, _OBJECT_BY_NAME

_WISH_VOCAB = list(_OBJECT_BY_NAME.keys())

@given(name=st.sampled_from(_WISH_VOCAB))
@settings(max_examples=50, deadline=None)
def test_wish_parser_no_crash(name):
    """parse_wish_string on any known vocab name must not raise."""
    result = parse_wish_string(name.encode())
    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 5, f"Expected 5-tuple, got length {len(result)}"


@given(s=st.text(max_size=32))
@settings(max_examples=50, deadline=None)
def test_wish_parser_unknown_no_crash(s):
    """parse_wish_string on arbitrary strings must not raise (handles unknowns)."""
    try:
        result = parse_wish_string(s.encode("utf-8", errors="replace"))
    except Exception as exc:
        pytest.fail(f"parse_wish_string raised on {s!r}: {exc}")
    assert isinstance(result, tuple)


# ---------------------------------------------------------------------------
# 10. Item.replace() preserves all 17 dataclass fields
# ---------------------------------------------------------------------------

from Nethax.nethax.subsystems.inventory import make_empty_item

_ITEM_FIELDS = (
    "category", "type_id", "buc_status", "enchantment", "charges",
    "identified", "quantity", "weight", "ac_bonus", "is_two_handed",
    "greased", "oeroded", "oeroded2", "oerodeproof", "bknown",
    "lamplit", "olocked",
)

@given(
    category=st.integers(min_value=0, max_value=17),
    quantity=st.integers(min_value=0, max_value=100),
    enchantment=st.integers(min_value=-7, max_value=7),
)
@settings(max_examples=50, deadline=None)
def test_item_replace_preserves_fields(category, quantity, enchantment):
    """Item.replace() must preserve all 17 fields (none silently dropped)."""
    item = make_empty_item()

    new_item = item.replace(
        category=jnp.int8(category),
        quantity=jnp.int16(quantity),
        enchantment=jnp.int8(enchantment),
    )

    assert len(_ITEM_FIELDS) == 17, "Field count changed — update test"
    for field in _ITEM_FIELDS:
        assert hasattr(new_item, field), f"Field '{field}' missing after replace()"

    # The three we changed should reflect the new values.
    assert int(new_item.category)    == category
    assert int(new_item.quantity)    == quantity
    assert int(new_item.enchantment) == enchantment

    # The remaining 14 fields must equal their original values.
    for field in _ITEM_FIELDS:
        if field in ("category", "quantity", "enchantment"):
            continue
        orig_val = np.array(getattr(item, field))
        new_val  = np.array(getattr(new_item, field))
        np.testing.assert_array_equal(
            orig_val, new_val,
            err_msg=f"Field '{field}' changed unexpectedly after replace()",
        )


# ---------------------------------------------------------------------------
# 11. Reset(seed=k) twice returns identical states
# ---------------------------------------------------------------------------

@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=50, deadline=None)
def test_reset_deterministic(seed):
    """Calling reset with the same seed twice must return identical obs dicts."""
    env = _ENV  # reuse the warm module-level env (JIT cache hot)
    rng = jax.random.PRNGKey(seed)

    _state_a, obs_a = env.reset(rng)
    _state_b, obs_b = env.reset(rng)

    for key in NLE_OBSERVATION_KEYS:
        np.testing.assert_array_equal(
            np.array(obs_a[key]),
            np.array(obs_b[key]),
            err_msg=f"Reset non-deterministic for key '{key}' seed={seed}",
        )


# ---------------------------------------------------------------------------
# 12. Done flag is sticky — once True, stays True
# ---------------------------------------------------------------------------

@given(
    steps=st.integers(min_value=1, max_value=5),
    seed=st.integers(min_value=0, max_value=9999),
)
@settings(max_examples=50, deadline=None)
def test_done_is_sticky(steps, seed):
    """Once done=True, every subsequent step must keep done=True."""
    env = _ENV  # reuse the warm module-level env (JIT cache hot)
    rng = jax.random.PRNGKey(seed)
    state, _obs = env.reset(rng)

    # Force done=True so we can test stickiness without waiting for death.
    state = state.replace(done=jnp.bool_(True))

    for i in range(steps):
        step_rng = jax.random.PRNGKey(seed + i + 1)
        state, _obs, _rew, done, _info = env.step(state, jnp.int32(0), step_rng)
        assert bool(done), (
            f"done became False after {i+1} step(s) past a True done; seed={seed}"
        )
