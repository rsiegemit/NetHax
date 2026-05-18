"""Prayer polish parity tests — gaps filled per vendor pray.c.

Covers:
  1. gods table / god_name() — pray.c:50
  2. altar conversion after 5 cross-aligned coaligned sacrifices — pray.c dosacrifice
  3. bone sacrifice (MS_BONES) → demonlord summon — pray.c dosacrifice bones branch
  4. good sacrifice bumps player_luck — pray.c dosacrifice change_luck(+1)
  5. dopray luck gate (Luck < -9 → no-op) — pray.c:250
  6. altar_known flag set on step — pray.c:1900 doaltar
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_state():
    """Return a reset EnvState via NethaxEnv."""
    from Nethax.nethax.env import NethaxEnv
    env = NethaxEnv()
    state, _ = env.reset(jax.random.PRNGKey(42))
    return state


def _place_altar(state, align_int: int):
    """Place an altar of the given alignment at the player's current position."""
    max_levels = state.terrain.shape[1]
    b  = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    flat_lv = b * max_levels + lv
    row = int(state.player_pos[0])
    col = int(state.player_pos[1])
    new_aa = state.features.altar_alignment.at[flat_lv, row, col].set(
        jnp.int8(align_int)
    )
    return state.replace(features=state.features.replace(altar_alignment=new_aa))


def _add_corpse(state, slot: int, type_id: int, align_override: int = None):
    """Add a FOOD-category corpse in inventory slot *slot* with the given type_id."""
    items = state.inventory.items
    new_cat = items.category.at[slot].set(jnp.int8(7))   # FOOD
    new_qty = items.quantity.at[slot].set(jnp.int16(1))
    # type_id used to encode corpse kind / alignment
    new_tid = items.type_id.at[slot].set(jnp.int16(type_id))
    new_items = items.replace(category=new_cat, quantity=new_qty, type_id=new_tid)
    return state.replace(inventory=state.inventory.replace(items=new_items))


# ---------------------------------------------------------------------------
# 1. God name table — pray.c:50
# ---------------------------------------------------------------------------

def test_god_name_human_lawful():
    """god_name(HUMAN, LAWFUL, OWN) == 'Anu'  (pray.c:50)."""
    from Nethax.nethax.subsystems.prayer import god_name, Race, Alignment, GOD_OWN
    assert god_name(Race.HUMAN, Alignment.LAWFUL, GOD_OWN) == "Anu"


def test_god_name_elf_chaotic():
    """god_name(ELF, CHAOTIC, OWN) == 'Lolth'  (pray.c:50)."""
    from Nethax.nethax.subsystems.prayer import god_name, Race, Alignment, GOD_OWN
    assert god_name(Race.ELF, Alignment.CHAOTIC, GOD_OWN) == "Lolth"


def test_god_name_neutral_kind():
    """god_name neutral-kind returns the middle entry."""
    from Nethax.nethax.subsystems.prayer import god_name, Race, Alignment, GOD_NEUTRAL
    assert god_name(Race.HUMAN, Alignment.LAWFUL, GOD_NEUTRAL) == "Ishtar"


def test_god_name_opposite_kind():
    """god_name opposite-kind returns the third entry."""
    from Nethax.nethax.subsystems.prayer import god_name, Race, Alignment, GOD_OPPOSITE
    assert god_name(Race.GNOME, Alignment.LAWFUL, GOD_OPPOSITE) == "Urdlen"


# ---------------------------------------------------------------------------
# 5. Luck gate — pray.c:250
# ---------------------------------------------------------------------------

def test_luck_gate_blocks_prayer():
    """pray() is a no-op when player_luck < -9  (pray.c:250)."""
    from Nethax.nethax.subsystems.prayer import pray

    state = _fresh_state()
    # Set luck to -10 (below -9 threshold).
    state = state.replace(player_luck=jnp.int8(-10))
    # Record current alignment_record so we can verify nothing changed.
    before_record = int(state.prayer.alignment_record)
    before_hp = int(state.player_hp)

    rng = jax.random.PRNGKey(0)
    new_state = pray(state, rng)

    assert int(new_state.prayer.alignment_record) == before_record, \
        "alignment_record should not change when luck < -9"
    assert int(new_state.player_hp) == before_hp, \
        "player_hp should not change when luck < -9"


def test_luck_gate_allows_prayer_at_minus_9():
    """pray() executes normally when player_luck == -9  (pray.c:250 boundary)."""
    from Nethax.nethax.subsystems.prayer import pray

    state = _fresh_state()
    state = state.replace(player_luck=jnp.int8(-9))
    # Force a non-angry state so the pleased branch fires.
    state = state.replace(prayer=state.prayer.replace(
        alignment_record=jnp.int16(10),
        pray_timeout=jnp.int32(0),
    ))

    rng = jax.random.PRNGKey(7)
    new_state = pray(state, rng)

    # pray_timeout should have been reset (> 0) meaning the pipeline ran.
    assert int(new_state.prayer.pray_timeout) > 0, \
        "pray_timeout should be reset when luck == -9 (gate should pass)"


# ---------------------------------------------------------------------------
# 4. Good sacrifice bumps luck — pray.c dosacrifice change_luck(+1)
# ---------------------------------------------------------------------------

def test_good_sacrifice_bumps_luck():
    """Successful coaligned sacrifice on coaligned altar increments player_luck."""
    from Nethax.nethax.subsystems.prayer import sacrifice_on_altar, Alignment

    state = _fresh_state()
    # Player is NEUTRAL (Alignment.NEUTRAL == 1).
    state = state.replace(player_align=jnp.int8(int(Alignment.NEUTRAL)))
    # Place a NEUTRAL altar at player position.
    state = _place_altar(state, int(Alignment.NEUTRAL))
    # Add a coaligned corpse (type_id == 1 = NEUTRAL alignment sentinel).
    state = _add_corpse(state, slot=0, type_id=int(Alignment.NEUTRAL))
    state = state.replace(player_luck=jnp.int8(0))

    rng = jax.random.PRNGKey(1)
    new_state = sacrifice_on_altar(state, rng, jnp.int32(0))

    assert int(new_state.player_luck) == 1, \
        f"expected luck 1 after good sacrifice, got {int(new_state.player_luck)}"


# ---------------------------------------------------------------------------
# 2. Altar conversion after 5 sacrifices — pray.c dosacrifice altar-conversion
# ---------------------------------------------------------------------------

def test_altar_conversion_after_5_sacrifices():
    """5 good coaligned sacrifices on a cross-aligned altar convert it."""
    from Nethax.nethax.subsystems.prayer import (
        sacrifice_on_altar, Alignment, ALTAR_CONVERT_THRESHOLD,
    )

    state = _fresh_state()
    # Player is LAWFUL (2), altar is CHAOTIC (0) — cross-aligned.
    state = state.replace(player_align=jnp.int8(int(Alignment.LAWFUL)))
    state = _place_altar(state, int(Alignment.CHAOTIC))

    # Coaligned corpse = type_id matching player alignment (LAWFUL == 2).
    # We need 5 sacrifices; loop using plain Python (host-side accumulation).
    rng = jax.random.PRNGKey(2)
    for i in range(ALTAR_CONVERT_THRESHOLD):
        state = _add_corpse(state, slot=0, type_id=int(Alignment.LAWFUL))
        rng, sub = jax.random.split(rng)
        state = sacrifice_on_altar(state, sub, jnp.int32(0))

    # Read altar alignment at player position.
    max_levels = state.terrain.shape[1]
    b  = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    flat_lv = b * max_levels + lv
    row = int(state.player_pos[0])
    col = int(state.player_pos[1])
    final_align = int(state.features.altar_alignment[flat_lv, row, col])
    assert final_align == int(Alignment.LAWFUL), \
        f"altar should have converted to LAWFUL, got {final_align}"
    # Sacrifice count should have reset to 0.
    final_count = int(state.features.altar_sacrifice_count[flat_lv, row, col])
    assert final_count == 0, f"altar_sacrifice_count should reset to 0, got {final_count}"


# ---------------------------------------------------------------------------
# 3. Bone sacrifice → demonlord summon — pray.c dosacrifice bones branch
# ---------------------------------------------------------------------------

def test_bone_sacrifice_summons_monster():
    """Sacrificing a bones corpse (type_id==19) activates a high-level monster slot."""
    from Nethax.nethax.subsystems.prayer import (
        sacrifice_on_altar, Alignment, BONES_TYPE_ID, DEMONLORD_LEVEL_MIN,
    )
    from Nethax.nethax.constants.monsters import MONSTERS

    state = _fresh_state()
    state = state.replace(player_align=jnp.int8(int(Alignment.NEUTRAL)))
    state = _place_altar(state, int(Alignment.NEUTRAL))
    # Add a bones corpse in slot 0.
    state = _add_corpse(state, slot=0, type_id=BONES_TYPE_ID)

    # Pre-condition: ensure at least one dead slot with a high-level entry exists.
    # Find a MONSTERS entry with level >= DEMONLORD_LEVEL_MIN.
    high_idx = next(
        i for i, m in enumerate(MONSTERS) if int(m.level) >= DEMONLORD_LEVEL_MIN
    )
    # Write that entry_idx into slot 0 and make sure it's dead.
    mai = state.monster_ai
    new_entry = mai.entry_idx.at[0].set(jnp.int16(high_idx))
    new_alive  = mai.alive.at[0].set(jnp.bool_(False))
    state = state.replace(monster_ai=mai.replace(entry_idx=new_entry, alive=new_alive))

    alive_before = int(jnp.sum(state.monster_ai.alive.astype(jnp.int32)))

    rng = jax.random.PRNGKey(3)
    new_state = sacrifice_on_altar(state, rng, jnp.int32(0))

    alive_after = int(jnp.sum(new_state.monster_ai.alive.astype(jnp.int32)))
    assert alive_after > alive_before, \
        f"bone sacrifice should spawn a monster: before={alive_before} after={alive_after}"


# ---------------------------------------------------------------------------
# 6. altar_known flag on step — pray.c:1900 doaltar
# ---------------------------------------------------------------------------

def test_altar_known_after_step():
    """altar_known[r,c] is True after altar_buc_sense is called on an altar tile."""
    from Nethax.nethax.subsystems.features import altar_buc_sense

    state = _fresh_state()
    # Place any altar at player position.
    state = _place_altar(state, 1)  # NEUTRAL

    max_levels = state.terrain.shape[1]
    b  = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    flat_lv = b * max_levels + lv
    row = int(state.player_pos[0])
    col = int(state.player_pos[1])

    assert not bool(state.features.altar_known[flat_lv, row, col]), \
        "altar_known should start False"

    new_state = altar_buc_sense(state)

    assert bool(new_state.features.altar_known[flat_lv, row, col]), \
        "altar_known should be True after stepping on altar"
