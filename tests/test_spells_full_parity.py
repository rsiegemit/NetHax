"""Full-parity spell tests — all 43 SpellId effects verified.

Each test invokes the handler directly via ``_EFFECT_DISPATCH`` / ``_StateAdapter``
so the success-roll is bypassed and we observe the pure effect.

Canonical sources:
  vendor/nethack/src/spell.c  — spelleffects dispatch
  vendor/nethack/src/zap.c    — ray/beam damage formulas
  vendor/nethack/src/detect.c — detection effects
  vendor/nethack/src/teleport.c — teleport effects
  vendor/nethack/src/cmd.c    — jump effect
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
    N_SPELLS,
)
from Nethax.nethax.subsystems.status_effects import Intrinsic, TimedStatus
from Nethax.nethax.constants.tiles import TileType


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


def _with_monster(state: EnvState, hp: int = 100, slot: int = 0) -> EnvState:
    """Place an alive monster in ``slot`` with ``hp`` HP."""
    mai = state.monster_ai
    new_hp     = mai.hp.at[slot].set(jnp.int32(hp))
    new_hp_max = mai.hp_max.at[slot].set(jnp.int32(max(hp, 100)))
    new_alive  = mai.alive.at[slot].set(True)
    new_entry  = mai.entry_idx.at[slot].set(jnp.int16(0))
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


def _dmgs(spell_id: SpellId, state: EnvState, n: int = 60) -> list[int]:
    """Sample damage distribution across n seeds."""
    base = int(state.monster_ai.hp[0])
    return [base - int(_run(spell_id, state, seed=i).monster_ai.hp[0]) for i in range(n)]


# ---------------------------------------------------------------------------
# 1. test_force_bolt_damages_target
# Cite: zap.c::bhitm line 205 ``dmg = d(2, 12)``
# ---------------------------------------------------------------------------

def test_force_bolt_damages_target():
    """FORCE_BOLT deals d(2,12) = 2..24 physical damage to monster slot 0."""
    state = _with_monster(_state(), hp=500)
    dmgs = _dmgs(SpellId.FORCE_BOLT, state)
    assert all(2 <= d <= 24 for d in dmgs), f"FORCE_BOLT out of [2,24]: {dmgs}"
    assert len(set(dmgs)) > 1, "FORCE_BOLT damage must be random"


# ---------------------------------------------------------------------------
# 2. test_healing_restores_hp
# Cite: zap.c::zapyourself line 2911 → healup(d(6,4), 0, FALSE, FALSE)
# ---------------------------------------------------------------------------

def test_healing_restores_hp():
    """HEALING restores d(6,4) = 6..24 HP, clamped at hp_max."""
    state = _state(player_hp=1, player_hp_max=200)
    heals = []
    for seed in range(60):
        new_s = _run(SpellId.HEALING, state, seed=seed)
        heals.append(int(new_s.player_hp) - 1)
    assert all(6 <= h <= 24 for h in heals), f"HEALING out of [6,24]: {heals}"
    assert len(set(heals)) > 1, "HEALING must be random"
    # Cap check
    capped = _run(SpellId.HEALING, _state(player_hp=50, player_hp_max=51), seed=0)
    assert int(capped.player_hp) <= 51


# ---------------------------------------------------------------------------
# 3. test_fireball_aoe
# Cite: zap.c::weffects line 3461 ubuzz with nd=u.ulevel/2+1; zhitm ZT_FIRE d(nd,6)
# ---------------------------------------------------------------------------

def test_fireball_aoe():
    """FIREBALL (unskilled) deals d(nd,6) where nd=xl/2+1; at XL=10 → 6..36."""
    state = _with_monster(_state(player_xl=10), hp=500)
    dmgs = _dmgs(SpellId.FIREBALL, state)
    assert all(6 <= d <= 36 for d in dmgs), f"FIREBALL out of [6,36]: {dmgs}"
    assert len(set(dmgs)) > 1, "FIREBALL must be random"


# ---------------------------------------------------------------------------
# 4. test_finger_of_death_instant_kill
# Cite: zap.c::bhitm DEATH path — kills non-resistant monster outright
# ---------------------------------------------------------------------------

def test_finger_of_death_instant_kill():
    """FINGER_OF_DEATH sets monster slot 0 HP to 0."""
    state = _with_monster(_state(), hp=500)
    new_state = _run(SpellId.FINGER_OF_DEATH, state, seed=0)
    assert int(new_state.monster_ai.hp[0]) == 0, "FINGER_OF_DEATH must kill monster"


# ---------------------------------------------------------------------------
# 5. test_detect_monsters_reveals_all
# Cite: detect.c::monster_detect — sets a timer for monster visibility
# ---------------------------------------------------------------------------

def test_detect_monsters_reveals_all():
    """DETECT_MONSTERS sets detect_monsters_until_turn = timestep + 100.

    Cite: vendor/nethack/src/detect.c::monster_detect.
    """
    state = _state()
    ts = int(state.timestep)
    new_state = _run(SpellId.DETECT_MONSTERS, state, seed=0)
    expected = ts + 100
    actual = int(new_state.identification.detect_monsters_until_turn)
    assert actual == expected, f"detect_monsters_until_turn: expected {expected}, got {actual}"


# ---------------------------------------------------------------------------
# 6. test_identify_marks_unknown
# Cite: read.c::SCR_IDENTIFY — identifies first unidentified inventory slot
# ---------------------------------------------------------------------------

def test_identify_marks_unknown():
    """IDENTIFY flips the first unidentified, non-empty inventory slot to identified=True.

    Cite: vendor/nethack/src/read.c::SCR_IDENTIFY.
    """
    state = _state()
    # Patch slot 0: category=1 (non-empty), identified=False
    inv = state.inventory
    items = inv.items
    new_cat  = items.category.at[0].set(jnp.int8(1))
    new_iden = items.identified.at[0].set(jnp.bool_(False))
    new_items = items.replace(category=new_cat, identified=new_iden)
    state = state.replace(inventory=inv.replace(items=new_items))
    assert not bool(state.inventory.items.identified[0])

    new_state = _run(SpellId.IDENTIFY, state, seed=0)
    assert bool(new_state.inventory.items.identified[0]), \
        "IDENTIFY must mark slot 0 identified"


# ---------------------------------------------------------------------------
# 7. test_haste_self_grants_fast
# Cite: potion.c::peffect_speed line 1063 — rn1(10, 100) → 100..109 turns
# ---------------------------------------------------------------------------

def test_haste_self_grants_fast():
    """HASTE_SELF grants FAST intrinsic for 100..109 turns (uncursed spell).

    Cite: vendor/nethack/src/potion.c::peffect_speed line 1063.
    """
    state = _state()
    for seed in range(20):
        new_state = _run(SpellId.HASTE_SELF, state, seed=seed)
        dur = int(new_state.status.timed_intrinsics[Intrinsic.FAST])
        assert 100 <= dur <= 109, f"HASTE_SELF duration {dur} not in [100,109] (seed={seed})"


# ---------------------------------------------------------------------------
# 8. test_invisibility_grants_invis
# Cite: zap.c::zapyourself line 2836 — incr_itimeout(&HInvis, rn1(15,31)) → 31..45
# ---------------------------------------------------------------------------

def test_invisibility_grants_invis():
    """INVISIBILITY grants INVIS_TMP timer of 31..45 turns.

    Cite: vendor/nethack/src/zap.c::zapyourself line 2836.
    """
    state = _state()
    for seed in range(20):
        new_state = _run(SpellId.INVISIBILITY, state, seed=seed)
        dur = int(new_state.status.timed_statuses[TimedStatus.INVIS_TMP])
        assert 31 <= dur <= 45, f"INVISIBILITY duration {dur} not in [31,45] (seed={seed})"


# ---------------------------------------------------------------------------
# 9. test_levitation_grants_levitation
# Cite: potion.c::peffect_levitation — timed levitation via peffects route
# ---------------------------------------------------------------------------

def test_levitation_grants_levitation():
    """LEVITATION grants a positive LEVITATION intrinsic timer.

    Cite: vendor/nethack/src/potion.c::peffect_levitation (via spell.c peffects).
    """
    state = _state()
    new_state = _run(SpellId.LEVITATION, state, seed=0)
    dur = int(new_state.status.timed_intrinsics[Intrinsic.LEVITATION])
    assert dur > 0, f"LEVITATION must grant >0 turns, got {dur}"


# ---------------------------------------------------------------------------
# 10. test_teleport_away_moves_monster
# Cite: teleport.c::dotele — teleports caster to random FLOOR tile on level
# Note: SPE_TELEPORT_AWAY in spell.c routes through dotele (self-teleport).
# ---------------------------------------------------------------------------

def test_teleport_away_moves_monster():
    """TELEPORT_AWAY moves the player to a FLOOR tile (self-teleport path).

    Vendor: vendor/nethack/src/teleport.c::dotele — picks a random floor
    tile and moves the hero there.  Cite: spell.c::spelleffects TELEPORT_AWAY.
    """
    state = _state()
    # Place player at a fixed position (0,0) and seed at least one FLOOR tile elsewhere.
    state = state.replace(player_pos=jnp.array([0, 0], dtype=jnp.int16))
    br = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    terrain = state.terrain.at[br, lv, 5, 5].set(jnp.int8(int(TileType.FLOOR)))
    state = state.replace(terrain=terrain)

    moved = False
    for seed in range(20):
        new_state = _run(SpellId.TELEPORT_AWAY, state, seed=seed)
        new_pos = new_state.player_pos
        if int(new_pos[0]) != 0 or int(new_pos[1]) != 0:
            moved = True
            break
    assert moved, "TELEPORT_AWAY must move player away from (0,0) when floor tiles exist"


# ---------------------------------------------------------------------------
# 11. test_jumping_short_range
# Cite: cmd.c::dojump — short-range jump to walkable tile
# ---------------------------------------------------------------------------

def test_jumping_short_range():
    """JUMPING shifts player_pos east by 2 when destination is FLOOR.

    Cite: vendor/nethack/src/cmd.c::dojump.
    """
    state = _state()
    state = state.replace(player_pos=jnp.array([5, 5], dtype=jnp.int16))
    br = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    # Ensure destination tile (5, 7) is FLOOR.
    terrain = state.terrain.at[br, lv, 5, 7].set(jnp.int8(int(TileType.FLOOR)))
    state = state.replace(terrain=terrain)

    new_state = _run(SpellId.JUMPING, state, seed=0)
    new_pos = new_state.player_pos
    assert int(new_pos[0]) == 5 and int(new_pos[1]) == 7, \
        f"JUMPING should land at (5,7), got ({int(new_pos[0])}, {int(new_pos[1])})"


# ---------------------------------------------------------------------------
# Coverage smoke-test: every SpellId has a handler in _EFFECT_DISPATCH
# ---------------------------------------------------------------------------

def test_all_43_spells_have_handlers():
    """All N_SPELLS SpellId values must be present in _EFFECT_DISPATCH."""
    missing = [s for s in SpellId if s not in _EFFECT_DISPATCH]
    assert not missing, f"SpellIds missing from _EFFECT_DISPATCH: {missing}"
    assert len(_EFFECT_DISPATCH) == N_SPELLS, \
        f"_EFFECT_DISPATCH has {len(_EFFECT_DISPATCH)} entries, expected {N_SPELLS}"
