"""Wave 5 Phase 4 genocide-scroll/spell tests.

Covers:
  - apply_genocide: a level seeded with monsters of a single class has them
    all removed (alive=False) when that class is the random pick.
  - apply_genocide always flips the GENOCIDELESS conduct.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.conduct import Conduct
from Nethax.nethax.subsystems.items_scrolls import apply_genocide
from Nethax.nethax.constants.monsters import MONSTERS, MonsterSymbol


_RNG = jax.random.PRNGKey(0)


def _find_entry_idx_with_symbol(symbol: int) -> int:
    """Return the first MONSTERS index whose .symbol matches ``symbol``."""
    for i, m in enumerate(MONSTERS):
        if int(m.symbol) == int(symbol):
            return i
    raise ValueError(f"No monster with symbol {symbol}")


def _seed_monsters_of_class(state: EnvState, symbol: int, n: int) -> EnvState:
    """Place ``n`` live monsters of the given monster-class symbol in slots
    0..n-1 of state.monster_ai."""
    entry_idx = _find_entry_idx_with_symbol(symbol)

    mai = state.monster_ai
    new_alive = mai.alive
    new_entry = mai.entry_idx
    new_hp = mai.hp
    new_pos = mai.pos
    for i in range(n):
        new_alive = new_alive.at[i].set(jnp.bool_(True))
        new_entry = new_entry.at[i].set(jnp.int16(entry_idx))
        new_hp = new_hp.at[i].set(jnp.int32(5))
        new_pos = new_pos.at[i].set(jnp.array([5 + i, 5 + i], dtype=jnp.int16))
    new_mai = mai.replace(alive=new_alive, entry_idx=new_entry,
                          hp=new_hp, pos=new_pos)
    return state.replace(monster_ai=new_mai)


def test_genocide_scroll_removes_class_of_monsters():
    """Spawn 5 gnomes; iterate seeds until apply_genocide picks the gnome
    class; assert all 5 are dead afterwards."""
    state = EnvState.default(_RNG)
    state = _seed_monsters_of_class(state, int(MonsterSymbol.S_GNOME), n=5)

    # The candidate-pool draw is RNG-dependent.  S_GNOME is in the pool;
    # search through several seeds until it is chosen.
    killed_all = False
    for seed in range(40):
        new_state = apply_genocide(state, jax.random.PRNGKey(seed))
        if not bool(new_state.monster_ai.alive[:5].any()):
            killed_all = True
            break
    assert killed_all, (
        "Expected at least one seed to pick S_GNOME and kill all five gnomes"
    )


def test_genocide_violates_genocideless():
    """apply_genocide always marks GENOCIDELESS, regardless of class pick."""
    state = EnvState.default(_RNG)
    assert bool(state.conduct.violations[int(Conduct.GENOCIDELESS)]) is False
    new_state = apply_genocide(state, _RNG)
    assert bool(new_state.conduct.violations[int(Conduct.GENOCIDELESS)]) is True


def test_genocide_via_scroll_path():
    """Reading a genocide scroll fires the same effect as apply_genocide
    (drives the items_scrolls dispatch table → _effect_genocide)."""
    from Nethax.nethax.constants.objects import ObjectClass
    from Nethax.nethax.subsystems.items_scrolls import handle_read, _SCROLL_BASE_ID, ScrollEffect

    state = EnvState.default(_RNG)
    # Put a genocide scroll in slot 0.
    type_id = _SCROLL_BASE_ID + int(ScrollEffect.GENOCIDE)
    items = state.inventory.items
    new_items = items.replace(
        category=items.category.at[0].set(jnp.int8(ObjectClass.SCROLL_CLASS)),
        type_id=items.type_id.at[0].set(jnp.int16(type_id)),
        quantity=items.quantity.at[0].set(jnp.int16(1)),
    )
    state = state.replace(inventory=state.inventory.replace(items=new_items))

    new_state = handle_read(state, _RNG)
    # Reading marks both ILLITERATE (any scroll read) and GENOCIDELESS.
    assert bool(new_state.conduct.violations[int(Conduct.ILLITERATE)]) is True
    assert bool(new_state.conduct.violations[int(Conduct.GENOCIDELESS)]) is True
