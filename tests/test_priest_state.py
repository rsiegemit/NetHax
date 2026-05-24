"""Tests for PriestState top-level slice (priest.py / state.py).

Vendor source: vendor/nethack/include/epri.h (struct epri) +
               vendor/nethack/src/priest.c::priest_talk lines 612-680
               (uses EPRI(priest)->cheapskate counter across calls).
"""

import pytest


def _fresh_state():
    import jax
    from Nethax.nethax.state import EnvState
    return EnvState.default(jax.random.PRNGKey(7))


def test_priest_state_present_with_defaults():
    """PriestState must be wired into EnvState with zeroed defaults."""
    state = _fresh_state()
    assert hasattr(state, "priest"), "EnvState missing 'priest' slice"
    assert int(state.priest.cheapskate_count) == 0
    assert int(state.priest.pri_alignment) == 0
    assert int(state.priest.pri_intone_time) == 0
    assert int(state.priest.pri_enter_time) == 0


def test_priest_state_pytree_round_trip():
    """PriestState must be a valid pytree (Flax struct)."""
    import jax
    state = _fresh_state()
    leaves, treedef = jax.tree_util.tree_flatten(state.priest)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert int(rebuilt.cheapskate_count) == 0


def test_priest_talk_persists_cheapskate_through_priest_state():
    """priest_talk should read+write state.priest.cheapskate_count.

    Vendor: priest.c lines 612-680 — EPRI(priest)->cheapskate bumps when
    the caller offers 0 (or below threshold) and persists into the next
    #chat-on-priest call.
    """
    import jax, jax.numpy as jnp
    from Nethax.nethax.subsystems.priest import priest_talk

    state = _fresh_state()
    # Force "offer 0" bucket by zeroing gold — priest_talk's deterministic
    # offer = gold//2 = 0 then.  But has_gold gate is True only if gold>0.
    # To force the bucket-0 path we need gold>0 yet offer rounds to 0; use
    # gold=1 → offer=0, has_gold=True → bucket_zero matches.
    state = state.replace(player_gold=jnp.int32(1))
    s2 = priest_talk(state, jax.random.PRNGKey(2))
    assert int(s2.priest.cheapskate_count) == 1, (
        f"cheapskate_count not bumped (got {int(s2.priest.cheapskate_count)})"
    )
    # Call again — should accumulate.
    s3 = priest_talk(s2.replace(player_gold=jnp.int32(1)), jax.random.PRNGKey(3))
    assert int(s3.priest.cheapskate_count) == 2


def test_priest_talk_no_gold_is_noop_on_cheapskate():
    """When the player has no gold, priest_talk leaves cheapskate alone.

    Vendor: priest.c line ~589 — early return when ``!money_cnt(invent)``
    before the bucket cascade runs.
    """
    import jax, jax.numpy as jnp
    from Nethax.nethax.subsystems.priest import priest_talk

    state = _fresh_state()
    state = state.replace(player_gold=jnp.int32(0))
    s2 = priest_talk(state, jax.random.PRNGKey(4))
    assert int(s2.priest.cheapskate_count) == 0
