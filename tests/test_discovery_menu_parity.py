"""Wave 8c — Discovery menu (``\\`` command) vendor parity.

Cite: vendor/nethack/src/o_init.c::dodiscovered (lines 762-873).
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import pytest

from Nethax.nethax.env import NethaxEnv
from Nethax.nethax.obs.discovery import build_discovery_text


def _decode_rows(rows: np.ndarray) -> list[str]:
    return [bytes(r).rstrip(b"\x00").decode("ascii") for r in rows]


@pytest.fixture(scope="module")
def initial_state():
    env = NethaxEnv()
    state, _ = env.reset(jax.random.PRNGKey(0))
    return state


def test_discovery_empty_on_fresh_state(initial_state):
    """A fresh state has nothing identified -> single 'haven't discovered yet'
    line (mirrors vendor o_init.c line 857)."""
    rows = build_discovery_text(initial_state)
    decoded = _decode_rows(rows)
    assert len(decoded) == 1
    assert "haven't discovered" in decoded[0]


def test_discovery_row_shape(initial_state):
    """Output shape must be (N, 80) uint8 — fixed-width row vectors."""
    rows = build_discovery_text(initial_state)
    assert rows.dtype == np.uint8
    assert rows.shape[1] == 80


def test_discovery_with_one_identified_potion(initial_state):
    """ID one potion -> 'Potions' header + that potion in the menu."""
    import jax.numpy as jnp
    state = initial_state
    # Find a real potion in OBJECTS and mark it identified.
    from Nethax.nethax.constants.objects import OBJECTS, ObjectClass
    potion_id = None
    for i, obj in enumerate(OBJECTS):
        if obj.class_ == ObjectClass.POTION_CLASS and obj.name:
            potion_id = i
            break
    assert potion_id is not None, "no potion found in OBJECTS table"

    new_ident = state.identification.replace(
        identified=state.identification.identified.at[potion_id].set(True)
    )
    state2 = state.replace(identification=new_ident)
    rows = build_discovery_text(state2)
    decoded = _decode_rows(rows)
    assert any("Potions" in L for L in decoded), (
        f"missing 'Potions' header in: {decoded}"
    )
    # The potion's display name should appear somewhere in the rows.
    pname = OBJECTS[potion_id].name
    assert any(pname in L for L in decoded), (
        f"missing potion name {pname!r} in: {decoded}"
    )
