"""Deep-parity spell tests — vendor-correct effects for previously-stubbed spells.

Each test invokes the handler directly via ``_EFFECT_DISPATCH`` / ``_StateAdapter``
so the success-roll is bypassed and we observe the pure effect.

Canonical sources:
  vendor/nethack/src/spell.c  — spelleffects dispatch
  vendor/nethack/src/zap.c    — ray/beam damage formulas
  vendor/nethack/src/detect.c — detection effects
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.magic import (
    SpellId,
    _EFFECT_DISPATCH,
    _StateAdapter,
)
from Nethax.nethax.subsystems.status_effects import TimedStatus
from Nethax.nethax.constants.tiles import TileType, VendorTileType
from Nethax.nethax.subsystems.features import DoorState


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _state(**over) -> EnvState:
    """Default EnvState: wizard role (12), pw=200, hp=10/100, INT=18, XL=10."""
    rng = jax.random.PRNGKey(0)
    s = EnvState.default(rng)
    s = s.replace(
        player_pw=jnp.int32(over.pop("player_pw", 200)),
        player_pw_max=jnp.int32(over.pop("player_pw_max", 200)),
        player_hp=jnp.int32(over.pop("player_hp", 10)),
        player_hp_max=jnp.int32(over.pop("player_hp_max", 100)),
        player_int=jnp.int8(over.pop("player_int", 18)),
        player_wis=jnp.int8(over.pop("player_wis", 18)),
        player_xl=jnp.int32(over.pop("player_xl", 10)),
        player_role=jnp.int8(over.pop("player_role", 12)),
    )
    for k, v in over.items():
        s = s.replace(**{k: v})
    return s


def _with_monster(state: EnvState, hp: int = 100, slot: int = 0,
                  entry_idx: int = 0) -> EnvState:
    """Place an alive monster in ``slot`` with ``hp`` HP and given entry_idx."""
    mai = state.monster_ai
    new_hp     = mai.hp.at[slot].set(jnp.int32(hp))
    new_hp_max = mai.hp_max.at[slot].set(jnp.int32(max(hp, 100)))
    new_alive  = mai.alive.at[slot].set(True)
    new_entry  = mai.entry_idx.at[slot].set(jnp.int16(entry_idx))
    return state.replace(monster_ai=mai.replace(
        hp=new_hp, hp_max=new_hp_max, alive=new_alive, entry_idx=new_entry,
    ))


def _run(spell_id: SpellId, state: EnvState, seed: int = 0) -> EnvState:
    """Run handler directly; return resulting EnvState."""
    handler = _EFFECT_DISPATCH[spell_id]
    adapter = _StateAdapter(state)
    rng = jax.random.PRNGKey(seed)
    result = handler(adapter, rng)
    if isinstance(result, dict):
        for k, v in result.items():
            adapter[k] = v
    return adapter.build()


# ---------------------------------------------------------------------------
# test_cause_fear_makes_monsters_flee
# Cite: spell.c::spelleffects CAUSE_FEAR → monflee(); flee_until_turn set.
# ---------------------------------------------------------------------------

def test_cause_fear_makes_monsters_flee():
    """CAUSE_FEAR sets flee_until_turn > timestep for alive monsters.

    Cite: vendor/nethack/src/spell.c::spelleffects (CAUSE_FEAR branch) →
    monflee() which sets the flee flag and flee-duration counter.
    """
    state = _with_monster(_state(), hp=50, slot=0)
    ts = int(state.timestep)
    new_state = _run(SpellId.CAUSE_FEAR, state, seed=0)
    flee_turn = int(new_state.monster_ai.flee_until_turn[0])
    assert flee_turn > ts, (
        f"CAUSE_FEAR: flee_until_turn={flee_turn} must be > timestep={ts}"
    )


# ---------------------------------------------------------------------------
# test_charm_monster_peaceful
# Cite: spell.c::spelleffects CHARM_MONSTER → seffects/taming → peaceful=True
# ---------------------------------------------------------------------------

def test_charm_monster_peaceful():
    """CHARM_MONSTER makes monster slot 0 peaceful.

    Cite: vendor/nethack/src/spell.c::spelleffects CHARM_MONSTER routes
    through seffects (read.c::SCR_TAMING) which marks monsters peaceful.
    """
    state = _with_monster(_state(), hp=50, slot=0)
    assert not bool(state.monster_ai.peaceful[0]), "monster should start hostile"
    new_state = _run(SpellId.CHARM_MONSTER, state, seed=0)
    assert bool(new_state.monster_ai.peaceful[0]), "CHARM_MONSTER must make monster peaceful"


# ---------------------------------------------------------------------------
# test_create_familiar_spawns_pet
# Cite: spell.c → makemon.c::makedog — spawns a tame dog/kitten
# ---------------------------------------------------------------------------

def test_create_familiar_spawns_pet():
    """CREATE_FAMILIAR increases the count of tame alive monsters by 1.

    Cite: vendor/nethack/src/spell.c::spelleffects CREATE_FAMILIAR →
    vendor/nethack/src/dog.c::makedog → makemon.c::makemon with tame flag.
    """
    state = _state()
    tame_before = int(jnp.sum(state.monster_ai.alive & state.monster_ai.tame))
    new_state = _run(SpellId.CREATE_FAMILIAR, state, seed=0)
    tame_after = int(jnp.sum(new_state.monster_ai.alive & new_state.monster_ai.tame))
    assert tame_after == tame_before + 1, (
        f"CREATE_FAMILIAR must add 1 tame monster; before={tame_before} after={tame_after}"
    )


# ---------------------------------------------------------------------------
# test_turn_undead_damages_undead
# Cite: zap.c::bhitm SPE_TURN_UNDEAD — undead take rnd(8), non-undead unharmed
# ---------------------------------------------------------------------------

def test_turn_undead_damages_undead():
    """TURN_UNDEAD deals 1..8 damage to undead; non-undead are unharmed.

    Cite: vendor/nethack/src/zap.c::bhitm SPE_TURN_UNDEAD branch:
    undead/vampires take ``rnd(8)`` damage; non-undead only flee.
    entry_idx=196 = orc mummy (M2_UNDEAD); entry_idx=0 = non-undead.
    """
    # Undead: orc mummy is array index 186 (chunk4[0]; vendor PM_ORC_MUMMY=196).
    # chunk1=64 + chunk2=62 + chunk3=60 = 186; chunk4 starts at index 186.
    # orc mummy flags2 = M2_UNDEAD | M2_HOSTILE | M2_ORC | M2_GREEDY | M2_JEWELS.
    UNDEAD_ENTRY = 186
    state_undead = _with_monster(_state(), hp=100, slot=0, entry_idx=UNDEAD_ENTRY)
    damages = []
    for seed in range(20):
        new_undead = _run(SpellId.TURN_UNDEAD, state_undead, seed=seed)
        damages.append(100 - int(new_undead.monster_ai.hp[0]))
    assert all(1 <= d <= 8 for d in damages), (
        f"TURN_UNDEAD deals rnd(8)=1..8 to undead; got damages={damages}"
    )

    # Non-undead (entry_idx=0, first monster in chunk1, not undead) should be unharmed
    state_normal = _with_monster(_state(), hp=100, slot=0, entry_idx=0)
    new_normal = _run(SpellId.TURN_UNDEAD, state_normal, seed=0)
    hp_normal = int(new_normal.monster_ai.hp[0])
    assert hp_normal == 100, (
        f"TURN_UNDEAD must not damage non-undead (entry_idx=0); hp={hp_normal}"
    )


# ---------------------------------------------------------------------------
# test_knock_opens_door
# Cite: lock.c::do_oclose (KNOCK) — opens a CLOSED_DOOR adjacent to player
# ---------------------------------------------------------------------------

def test_knock_opens_door():
    """KNOCK opens a CLOSED_DOOR adjacent to the player.

    Cite: vendor/nethack/src/lock.c::do_oclose (KNOCK spell path).
    """
    state = _state()
    pos = state.player_pos
    pr = int(pos[0])
    pc = int(pos[1])
    br = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    feat = state.features
    max_levels = feat.door_state.shape[0]

    # Compute flat level index (same logic as _effect_knock)
    flat_lv = br * (state.dungeon.stair_links.shape[1]) + lv

    CLOSED = int(DoorState.CLOSED)
    # Place a CLOSED door directly east of player
    new_ds = feat.door_state.at[flat_lv, pr, pc + 1].set(jnp.int8(CLOSED))
    state = state.replace(features=feat.replace(door_state=new_ds))

    new_state = _run(SpellId.KNOCK, state, seed=0)
    door_val = int(new_state.features.door_state[flat_lv, pr, pc + 1])
    assert door_val == int(DoorState.OPEN), (
        f"KNOCK must open CLOSED door; got DoorState={door_val}"
    )


# ---------------------------------------------------------------------------
# test_detect_unseen_reveals_sdoor
# Cite: detect.c SPE_DETECT_UNSEEN — SDOOR -> CLOSED_DOOR on terrain
# ---------------------------------------------------------------------------

def test_detect_unseen_reveals_sdoor():
    """DETECT_UNSEEN converts SDOOR terrain tiles to CLOSED_DOOR.

    Cite: vendor/nethack/src/detect.c (SPE_DETECT_UNSEEN branch, ~line 1340):
    reveals secret doors (SDOOR) as closed doors on the terrain map.
    """
    state = _state()
    br = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1

    SDOOR = int(VendorTileType.SDOOR)
    CLOSED_DOOR = int(TileType.CLOSED_DOOR)

    # Place an SDOOR tile somewhere visible on the level
    new_terrain = state.terrain.at[br, lv, 5, 5].set(jnp.int8(SDOOR))
    state = state.replace(terrain=new_terrain)
    assert int(state.terrain[br, lv, 5, 5]) == SDOOR

    new_state = _run(SpellId.DETECT_UNSEEN, state, seed=0)
    tile_after = int(new_state.terrain[br, lv, 5, 5])
    assert tile_after == CLOSED_DOOR, (
        f"DETECT_UNSEEN must convert SDOOR({SDOOR}) to CLOSED_DOOR({CLOSED_DOOR}); got {tile_after}"
    )


# ---------------------------------------------------------------------------
# test_stone_to_flesh_cures_stoned
# Cite: spell.c STONE_TO_FLESH → cures STONED timed status
# ---------------------------------------------------------------------------

def test_stone_to_flesh_cures_stoned():
    """STONE_TO_FLESH clears the STONED timed status.

    Cite: vendor/nethack/src/spell.c::spelleffects STONE_TO_FLESH — cures
    the petrification (STONED) status effect on the caster.
    """
    state = _state()
    # Set STONED timer to non-zero
    new_ts = state.status.timed_statuses.at[TimedStatus.STONED].set(jnp.int32(5))
    state = state.replace(status=state.status.replace(timed_statuses=new_ts))
    assert int(state.status.timed_statuses[TimedStatus.STONED]) == 5

    new_state = _run(SpellId.STONE_TO_FLESH, state, seed=0)
    stoned_after = int(new_state.status.timed_statuses[TimedStatus.STONED])
    assert stoned_after == 0, (
        f"STONE_TO_FLESH must clear STONED; timer={stoned_after}"
    )
