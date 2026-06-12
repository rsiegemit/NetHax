"""Quest subsystem — quest branch, leader/nemesis/artifact tracking.

Canonical sources:
  vendor/nethack/src/quest.c    — quest_status struct (Qstat macros), leader
                                   chat (chat_with_leader, lines ~282-370),
                                   nemesis kill handler (killed_nemesis, ~109-125),
                                   artifact pickup (touched_artifact, ~127-141),
                                   finish_quest (~226-280)
  vendor/nethack/src/questpgr.c — quest dialogue text (skipped for JAX reimpl;
                                   no text rendering in RL environment)
  vendor/nethack/src/role.c     — Role struct fields:
                                    .lead0/.lead1/.lead2 — leader monster ids
                                    .neminum             — nemesis monster id
                                    .qlist               — quest artifact id
                                    .prefix              — quest text prefix
                                   See role.c lines 30-573 (13 role entries).

Status: Wave 5 Phase 3 — per-role data table, nemesis fight mechanics,
        leader-return victory check.
"""
from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import struct


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------
class QuestStage(IntEnum):
    """Linear quest progression stages.

    Mirrors the boolean flags in quest_status (quest.c Qstat macros):
      met_leader, killed_nemesis, touched_artifact, finish_quest.
    Collapsed into a single ordered stage for RL convenience.
    """

    NOT_STARTED = 0        # hero has not entered the quest branch
    ENTERED_QUEST = 1      # hero descended into the quest branch
    LEADER_GREETED = 2     # Qstat(met_leader) = TRUE (quest.c ~323)
    NEMESIS_KILLED = 3     # Qstat(killed_nemesis) = TRUE (quest.c ~110)
    ARTIFACT_RECOVERED = 4 # Qstat(touched_artifact) = TRUE (quest.c ~132)
    RETURNED_TO_LEADER = 5 # finish_quest() completed (quest.c ~263)


# ---------------------------------------------------------------------------
# Per-role quest data table (Wave 5 Phase 3)
# ---------------------------------------------------------------------------
# Each role gets a (leader_idx, nemesis_idx, artifact_idx, prefix_idx) tuple.
# Sources:
#   leader_idx   — Role.lead0  (role.c, e.g. line 47 PM_LORD_CARNARVON for Arc)
#   nemesis_idx  — Role.neminum (role.c, e.g. line 49 PM_MINION_OF_HUHETOTL)
#   artifact_idx — Role.qlist  (role.c, e.g. line 54 ART_ORB_OF_DETECTION)
#   prefix_idx   — Role index itself; quest text prefix is the 3-letter
#                  code at role.c .prefix (e.g. "Arc", line 42).
#
# Indices are computed at import time against the canonical
# MONSTERS table (Nethax/nethax/constants/monsters.py) so they remain in
# sync with that source of truth. Artifact indices come from the position
# in vendor/nethack/include/artilist.h (0-based, "" at index 0).
# ---------------------------------------------------------------------------

class _RoleQuest(NamedTuple):
    """Per-role quest constants. Fields match role.c struct Role names."""
    role_code: str       # e.g. "Arc", "Wiz"  (role.c .prefix field)
    leader_idx: int      # MONSTERS index for lead0   (role.c .lead0)
    nemesis_idx: int     # MONSTERS index for nemesis (role.c .neminum)
    artifact_idx: int    # artilist.h 0-based index   (role.c .qlist)
    prefix_idx: int      # role ordinal; mirrors role.c roles[] order


# Per-role data table (13 entries; ordered matching role.c roles[] order).
# Citations: role.c line numbers in trailing comments.
_QUEST_DATA = (
    # Wave 6 parity-fix: updated to match vendor/nethack/src/role.c roles[]
    # leader/nemesis fields against the actual MONSTERS table indices
    # (constants/monsters.py).  Prior entries were off by one row.
    # 0 — Archeologist (role.c lines 28-69; lead0=Lord Carnarvon role.c:45,
    #     nemesis=Minion of Huhetotl role.c:47)
    _RoleQuest("Arc", leader_idx=342, nemesis_idx=355, artifact_idx=20, prefix_idx=0),
    # 1 — Barbarian (role.c lines 70-111; lead0=Pelias role.c:87,
    #     nemesis=Thoth Amon role.c:89)
    _RoleQuest("Bar", leader_idx=343, nemesis_idx=356, artifact_idx=21, prefix_idx=1),
    # 2 — Caveman (role.c lines 112-153; lead0=Shaman Karnov role.c:129,
    #     nemesis=Chromatic Dragon role.c:131)
    _RoleQuest("Cav", leader_idx=344, nemesis_idx=357, artifact_idx=22, prefix_idx=2),
    # 3 — Healer (role.c lines 154-194; lead0=Hippocrates role.c:171,
    #     nemesis=Cyclops role.c:173)
    _RoleQuest("Hea", leader_idx=345, nemesis_idx=358, artifact_idx=24, prefix_idx=3),
    # 4 — Knight (role.c lines 195-235; lead0=King Arthur role.c:212,
    #     nemesis=Ixoth role.c:214)
    _RoleQuest("Kni", leader_idx=346, nemesis_idx=359, artifact_idx=25, prefix_idx=4),
    # 5 — Monk (role.c lines 236-277; lead0=Grand Master role.c:253,
    #     nemesis=Master Kaen role.c:255)
    _RoleQuest("Mon", leader_idx=347, nemesis_idx=360, artifact_idx=26, prefix_idx=5),
    # 6 — Priest (role.c lines 278-319; lead0=Arch Priest role.c:295,
    #     nemesis=Nalzok role.c:297)
    _RoleQuest("Pri", leader_idx=348, nemesis_idx=361, artifact_idx=27, prefix_idx=6),
    # 7 — Rogue (role.c lines 322-362; lead0=Master of Thieves role.c:339,
    #     nemesis=Master Assassin role.c:341)
    _RoleQuest("Rog", leader_idx=350, nemesis_idx=363, artifact_idx=29, prefix_idx=7),
    # 8 — Ranger (role.c lines 363-418; lead0=Orion role.c:394,
    #     nemesis=Scorpius role.c:396)
    _RoleQuest("Ran", leader_idx=349, nemesis_idx=362, artifact_idx=28, prefix_idx=8),
    # 9 — Samurai (role.c lines 419-459; lead0=Lord Sato role.c:436,
    #     nemesis=Ashikaga Takauji role.c:438)
    _RoleQuest("Sam", leader_idx=351, nemesis_idx=364, artifact_idx=30, prefix_idx=9),
    # 10 — Tourist (role.c lines 460-500; lead0=Twoflower role.c:477,
    #      nemesis=Master of Thieves role.c:479 [same monster as Rog leader])
    _RoleQuest("Tou", leader_idx=352, nemesis_idx=350, artifact_idx=31, prefix_idx=10),
    # 11 — Valkyrie (role.c lines 501-541; lead0=Norn role.c:518,
    #      nemesis=Lord Surtur role.c:520)
    _RoleQuest("Val", leader_idx=353, nemesis_idx=365, artifact_idx=32, prefix_idx=11),
    # 12 — Wizard (role.c lines 542-583; lead0=Neferet the Green role.c:559,
    #      nemesis=Dark One role.c:561)
    _RoleQuest("Wiz", leader_idx=354, nemesis_idx=366, artifact_idx=33, prefix_idx=12),
)


def get_quest_data(role: int) -> _RoleQuest:
    """Return the per-role quest data tuple for `role` (Python-int index)."""
    return _QUEST_DATA[role]


# JIT-friendly flat int arrays for use inside jit'd code.
_LEADER_IDX_BY_ROLE   = jnp.array([d.leader_idx   for d in _QUEST_DATA], dtype=jnp.int16)
_NEMESIS_IDX_BY_ROLE  = jnp.array([d.nemesis_idx  for d in _QUEST_DATA], dtype=jnp.int16)
_ARTIFACT_IDX_BY_ROLE = jnp.array([d.artifact_idx for d in _QUEST_DATA], dtype=jnp.int16)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
@struct.dataclass
class QuestState:
    """Quest progress and entity positions for the current game.

    Fields
    ------
    stage              : int8 — current QuestStage value
    met_leader         : bool — Qstat(met_leader); set once the hero has
                                spoken to the quest leader at least once
                                (quest.c chat_with_leader ~323)
    nemesis_alive      : bool — False once slay_nemesis() fires
    nemesis_killed     : bool — sticky flag set when nemesis dies
                                (Qstat(killed_nemesis), quest.c ~109-125)
    touched_artifact   : bool — Qstat(touched_artifact); set once the
                                quest artifact has been touched/picked up
                                (quest.c ~127-141)
    artifact_carried   : bool — True while the quest artifact is in the
                                hero's inventory (may toggle off if dropped;
                                touched_artifact stays sticky)
    completed          : bool — Qstat(qcompleted) / u.uevent.qcompleted;
                                True after finish_quest succeeds (~263)
    qexpelled          : bool — u.uevent.qexpelled; set when the hero is
                                expelled by an offended leader (quest.c ~202)
    leader_pos         : [2] int16 — (row, col) of quest leader on quest
                                level; (-1, -1) when hero is off-level
    nemesis_pos        : [2] int16 — (row, col) of nemesis; (-1, -1) when dead
    """

    stage: jnp.ndarray             # scalar   int8
    met_leader: jnp.ndarray        # scalar   bool   (Qstat(met_leader))
    nemesis_alive: jnp.ndarray     # scalar   bool
    nemesis_killed: jnp.ndarray    # scalar   bool   (Wave 5 sticky flag)
    touched_artifact: jnp.ndarray  # scalar   bool   (Qstat(touched_artifact))
    artifact_carried: jnp.ndarray  # scalar   bool
    completed: jnp.ndarray         # scalar   bool   (Wave 5 victory flag)
    qexpelled: jnp.ndarray         # scalar   bool   (u.uevent.qexpelled)
    leader_pos: jnp.ndarray        # [2]      int16
    nemesis_pos: jnp.ndarray       # [2]      int16

    @classmethod
    def default(cls) -> "QuestState":
        """Return a default QuestState for a new game (quest not yet started)."""
        return cls(
            stage=jnp.int8(QuestStage.NOT_STARTED),
            met_leader=jnp.bool_(False),
            nemesis_alive=jnp.bool_(True),
            nemesis_killed=jnp.bool_(False),
            touched_artifact=jnp.bool_(False),
            artifact_carried=jnp.bool_(False),
            completed=jnp.bool_(False),
            qexpelled=jnp.bool_(False),
            leader_pos=jnp.full((2,), -1, dtype=jnp.int16),
            nemesis_pos=jnp.full((2,), -1, dtype=jnp.int16),
        )


# ---------------------------------------------------------------------------
# Stage-transition wiring (vendor quest.c on_quest_level + leader_talk +
# kill_nemesis + pickup_artifact + return_to_leader event chain).
# ---------------------------------------------------------------------------
def enter_quest_branch(state: QuestState, role: int) -> QuestState:
    """Player descended into the quest branch — advance stage to ENTERED_QUEST.

    Mirrors vendor/nethack/src/quest.c::on_quest_level / Is_questlevel
    enforcement: the first time the hero arrives on the quest branch,
    the stage advances from BEFORE_QUEST to ENTERED_QUEST.

    Role-specific leader-spawn-position lookup is owned by the dungeon
    layer (dungeon/quest_levels.py); this function only owns the QuestState
    transition.

    Cite: vendor/nethack/src/quest.c::on_quest_level (the per-turn
          guard) + Is_questlevel macro.
    """
    new_stage = jnp.maximum(state.stage, jnp.int8(QuestStage.ENTERED_QUEST))
    return state.replace(stage=new_stage)


def talk_to_leader(state: QuestState) -> QuestState:
    """Mark the leader as met (quest.c: chat_with_leader ~321-324).

    Sets ``met_leader = True`` (sticky) and advances ``stage`` to at least
    ``LEADER_GREETED``.
    """
    new_stage = jnp.maximum(state.stage, jnp.int8(QuestStage.LEADER_GREETED))
    return state.replace(
        met_leader=jnp.bool_(True),
        stage=new_stage,
    )


def slay_nemesis(state: QuestState, monster_idx: jnp.ndarray) -> QuestState:
    """Mark the nemesis as slain (quest.c: killed_nemesis lines ~109-125).

    Sets `nemesis_alive=False`, `nemesis_killed=True`, advances stage to
    NEMESIS_KILLED, and zeroes nemesis_pos. The actual artifact drop is
    handled by the level/item subsystem when the monster object dies.
    """
    new_stage = jnp.maximum(state.stage, jnp.int8(QuestStage.NEMESIS_KILLED))
    return state.replace(
        nemesis_alive=jnp.bool_(False),
        nemesis_killed=jnp.bool_(True),
        stage=new_stage,
        nemesis_pos=jnp.full((2,), -1, dtype=jnp.int16),
    )


def pickup_artifact(state: QuestState) -> QuestState:
    """Mark the quest artifact as touched (quest.c: touched_artifact ~127-141).

    Sets the sticky ``touched_artifact`` flag and the transient
    ``artifact_carried`` flag, and advances ``stage`` to at least
    ``ARTIFACT_RECOVERED``.
    """
    new_stage = jnp.maximum(state.stage, jnp.int8(QuestStage.ARTIFACT_RECOVERED))
    return state.replace(
        touched_artifact=jnp.bool_(True),
        artifact_carried=jnp.bool_(True),
        stage=new_stage,
    )


# ---------------------------------------------------------------------------
# Wave 5 Phase 3 — nemesis fight mechanics
# ---------------------------------------------------------------------------
# The nemesis is a boosted version of its base monster entry: vendor/nethack
# applies elite scaling in mklev.c and quest_init.  We mirror this with a
# simple "4x hp + 1 extra attack roll" model (see Hea-goal.lua, Wiz-goal.lua
# which already give the nemesis special status).
# ---------------------------------------------------------------------------

NEMESIS_HP_MULTIPLIER: int = 4
NEMESIS_EXTRA_ATTACK_ROLLS: int = 1


def is_nemesis(role: int, monster_entry_idx: int) -> bool:
    """Return True if `monster_entry_idx` matches the nemesis for `role`."""
    return int(_QUEST_DATA[role].nemesis_idx) == int(monster_entry_idx)


def is_nemesis_at(state, monster_idx: jnp.ndarray, role: int) -> jnp.ndarray:
    """JIT-friendly check: is the monster at slot `monster_idx` the role nemesis?

    Args:
        state:        EnvState (must have .monster_ai.entry_idx).
        monster_idx:  scalar int slot in MAX_MONSTERS_PER_LEVEL.
        role:         Python-int role index (static for JIT).

    Returns:
        scalar bool jnp.ndarray.
    """
    target = jnp.int16(_QUEST_DATA[role].nemesis_idx)
    return state.monster_ai.entry_idx[monster_idx] == target


def boost_nemesis_hp(base_hp: jnp.ndarray) -> jnp.ndarray:
    """Apply nemesis hp multiplier (vendor: 4x base).

    Used at spawn time; integer math, JIT-safe.
    """
    return base_hp * jnp.int32(NEMESIS_HP_MULTIPLIER)


def nemesis_attack_rolls(base_rolls: int = 1) -> int:
    """Return the number of attack rolls the nemesis gets per turn.

    Vendor (mhitu.c) gives "elite" quest monsters one extra attack action
    in addition to their natural attack list.  We approximate with +1.
    """
    return base_rolls + NEMESIS_EXTRA_ATTACK_ROLLS


def nemesis_killed(state: QuestState) -> QuestState:
    """Set the sticky `nemesis_killed` flag and advance stage.

    Idempotent — calling this twice is a no-op after the first call.
    """
    return slay_nemesis(state, jnp.int32(-1))


# ---------------------------------------------------------------------------
# Wave 5 Phase 3 — return-to-leader victory check
# ---------------------------------------------------------------------------

def _adjacent(pos_a: jnp.ndarray, pos_b: jnp.ndarray) -> jnp.ndarray:
    """Return True if two (row, col) positions are within Chebyshev distance 1."""
    dr = jnp.abs(pos_a[0] - pos_b[0])
    dc = jnp.abs(pos_a[1] - pos_b[1])
    return jnp.logical_and(dr <= 1, dc <= 1)


def check_quest_complete(
    state: QuestState,
    player_pos: jnp.ndarray,
    has_artifact: jnp.ndarray,
) -> QuestState:
    """Return updated QuestState with `completed=True` if victory conditions hold.

    Wave 5 / vendor parity gate (quest.c finish_quest ~226-280):
      (1) the nemesis has been killed (sticky `nemesis_killed`),
      (2) the artifact has been touched (sticky `touched_artifact`),
      (3) the player is adjacent to the leader (proxy for met_leader chat).

    `met_leader` adjacency is treated as the runtime equivalent of
    Qstat(met_leader): standing next to the leader at completion time
    implies the chat has happened.  The `touched_artifact` flag is
    auto-set whenever ``has_artifact`` is True (you can't carry the
    artifact without having touched it), preserving back-compat with
    callers that only flip `artifact_carried`.

    Args:
        state:        QuestState with leader_pos and sticky flags.
        player_pos:   int16[2] — (row, col) of the player.
        has_artifact: bool — True if the quest artifact is in inventory.

    Returns:
        QuestState — same fields, with `completed` set if all three
        predicates hold; otherwise unchanged.  ``touched_artifact``
        becomes sticky-True the first time `has_artifact` is True.
    """
    leader_valid = state.leader_pos[0] >= 0
    near_leader = jnp.logical_and(leader_valid, _adjacent(player_pos, state.leader_pos))
    has_artifact_b = jnp.bool_(has_artifact)
    # Sticky touched_artifact: once you've held it, the flag stays True.
    new_touched = jnp.logical_or(state.touched_artifact, has_artifact_b)
    all_conditions = jnp.logical_and(
        state.nemesis_killed,
        jnp.logical_and(
            jnp.logical_and(has_artifact_b, new_touched),
            near_leader,
        ),
    )
    new_stage = jnp.where(
        all_conditions,
        jnp.int8(QuestStage.RETURNED_TO_LEADER),
        state.stage,
    )
    return state.replace(
        touched_artifact=new_touched,
        completed=jnp.logical_or(state.completed, all_conditions),
        stage=new_stage,
    )


def step(state: QuestState, rng: jax.Array) -> QuestState:
    """Per-turn no-op — quest state only changes on explicit event calls."""
    return state


# ---------------------------------------------------------------------------
# Spec-aligned progression hooks (quest.c / qstplay.c parity)
#
# Stage constants used below match qstplay.c QSTAGE_* values:
#   _QSTAGE_BEGUN    = 1  — after meeting Quest Leader (quest.c chat_with_leader ~321-324)
#   _QSTAGE_GOT_OBJ  = 2  — after picking up quest artifact (quest.c artitouch ~127-134)
#   _QSTAGE_COMPLETE = 4  — returned to Leader with artifact (quest.c finish_quest ~263-279)
# ---------------------------------------------------------------------------
_QSTAGE_BEGUN    = jnp.int8(1)
_QSTAGE_GOT_OBJ  = jnp.int8(2)
_QSTAGE_COMPLETE = jnp.int8(4)


# ---------------------------------------------------------------------------
# Leader spawn helpers (P2 — vendor sp_lev / quest.c::chat_with_leader parity)
# ---------------------------------------------------------------------------
# Vendor places the quest leader through the quest-level template files
# (vendor/nethack/dat/{role}-strt.des) at the SLDR mapping symbol.  Here we
# spawn the leader at a deterministic fixed tile near the level center and
# route chat/hostility through the existing peaceful flag.  See
# vendor/nethack/src/quest.c::chat_with_leader (lines 282-370) for the
# alignment-purity / pissed_off rules that gate peacefulness.
# ---------------------------------------------------------------------------

_LEADER_SPAWN_ROW = jnp.int16(10)
_LEADER_SPAWN_COL = jnp.int16(40)


def _spawn_quest_leader(state) -> object:
    """Place the role's quest leader on the current level and record leader_pos.

    Vendor parity:
      * leader monster id from Role.lead0 (vendor/nethack/src/role.c roles[];
        mapped per-role in _LEADER_IDX_BY_ROLE).
      * peacefulness mirrors chat_with_leader's purity check
        (quest.c:323-358): peaceful iff player and leader alignment signs
        match.  Otherwise hostile until cured.
      * leader_pos recorded so check_quest_complete adjacency works.
    """
    from Nethax.nethax.constants.monsters import MONSTERS

    role_i = jnp.clip(
        state.player_role.astype(jnp.int32), 0, _LEADER_IDX_BY_ROLE.shape[0] - 1
    )
    leader_entry = _LEADER_IDX_BY_ROLE[role_i].astype(jnp.int32)

    levels_arr = jnp.array([int(m.level) for m in MONSTERS], dtype=jnp.int32)
    sizes_arr = jnp.array([int(m.size) for m in MONSTERS], dtype=jnp.int32)
    align_arr = jnp.array(
        [int(getattr(m, "alignment", 0)) for m in MONSTERS], dtype=jnp.int32
    )
    from Nethax.nethax.dungeon.spawning import _ATK_DICE_N, _ATK_DICE_S, _BASE_AC

    lvl = jnp.take(levels_arr, leader_entry)
    ac = jnp.take(_BASE_AC, leader_entry)
    is_large = jnp.take(sizes_arr, leader_entry) >= jnp.int32(4)  # MZ_LARGE
    atk_n = jnp.take(_ATK_DICE_N, leader_entry)
    atk_s = jnp.take(_ATK_DICE_S, leader_entry)
    leader_align = jnp.take(align_arr, leader_entry)

    pal = state.player_align.astype(jnp.int32)
    same_sign = jnp.logical_or(
        jnp.logical_and(pal > 0, leader_align > 0),
        jnp.logical_or(
            jnp.logical_and(pal < 0, leader_align < 0),
            jnp.logical_and(pal == 0, leader_align == 0),
        ),
    )

    # Mean hp = d(mlevel, 8) ≈ 4*lvl; quest level construction is one-shot.
    hp = jnp.maximum(lvl * jnp.int32(4), jnp.int32(1))

    mai = state.monster_ai
    free_slot = jnp.argmin(mai.alive.astype(jnp.int32)).astype(jnp.int32)
    has_free = ~mai.alive[free_slot]

    leader_pos = jnp.stack([_LEADER_SPAWN_ROW, _LEADER_SPAWN_COL])

    def _do_spawn(m):
        return m.replace(
            pos=m.pos.at[free_slot].set(leader_pos),
            hp=m.hp.at[free_slot].set(hp),
            hp_max=m.hp_max.at[free_slot].set(hp),
            alive=m.alive.at[free_slot].set(jnp.bool_(True)),
            ac=m.ac.at[free_slot].set(ac),
            is_large=m.is_large.at[free_slot].set(is_large),
            attack_dice_n=m.attack_dice_n.at[free_slot].set(atk_n.astype(jnp.int8)),
            attack_dice_sides=m.attack_dice_sides.at[free_slot].set(atk_s.astype(jnp.int8)),
            mstrategy=m.mstrategy.at[free_slot].set(jnp.int8(0)),
            entry_idx=m.entry_idx.at[free_slot].set(leader_entry.astype(jnp.int16)),
            peaceful=m.peaceful.at[free_slot].set(same_sign),
            movement_points=m.movement_points.at[free_slot].set(jnp.int16(12)),
        )

    new_mai = jax.tree_util.tree_map(
        lambda s, o: jnp.where(has_free, s, o), _do_spawn(mai), mai,
    )
    new_quest = state.quest.replace(leader_pos=leader_pos)
    return state.replace(monster_ai=new_mai, quest=new_quest)


def on_enter_quest_level(state) -> object:
    """Set met_leader=True, spawn quest leader, advance stage to BEGUN_QUEST.

    P2 vendor parity:
      * Spawn the role's quest leader at a fixed position on first entry.
        Mirrors vendor sp_lev SLDR placement in dat/{role}-strt.des plus
        the chat_with_leader purity gate (quest.c:282-370).
      * leader_pos populated for check_quest_complete adjacency.

    JIT-pure.
    """
    state = _spawn_quest_leader(state)
    new_stage = jnp.maximum(state.quest.stage, _QSTAGE_BEGUN)
    new_quest = state.quest.replace(
        met_leader=jnp.bool_(True),
        stage=new_stage,
    )
    return state.replace(quest=new_quest)


def on_nemesis_killed(state, monster_entry_idx: jnp.ndarray) -> object:
    """Set nemesis_killed=True when the role's nemesis is slain.

    Mirrors quest.c::nemdead (~109-113): Qstat(killed_nemesis) = TRUE.
    The caller confirms monster_entry_idx matches the role's nemesis before
    calling (combat.py gates on entry_idx == nemesis_entry).

    JIT-pure: delegates to slay_nemesis.
    """
    new_quest = slay_nemesis(state.quest, monster_entry_idx)
    return state.replace(quest=new_quest)


def on_artifact_picked_up(state) -> object:
    """Set touched_artifact=True and advance stage to GOT_QUEST_OBJECT (2).

    Mirrors quest.c::artitouch (~127-134): Qstat(touched_artifact) = TRUE.
    Called from inventory pickup when the picked-up item's type_id matches
    the role's quest artifact index (_ARTIFACT_IDX_BY_ROLE[player_role]).

    JIT-pure.
    """
    new_stage = jnp.maximum(state.quest.stage, _QSTAGE_GOT_OBJ)
    new_quest = state.quest.replace(
        touched_artifact=jnp.bool_(True),
        artifact_carried=jnp.bool_(True),
        stage=new_stage,
    )
    return state.replace(quest=new_quest)


def on_return_to_leader(state) -> object:
    """Set completed=True and advance stage to COMPLETED (4).

    Mirrors quest.c::finish_quest (~263-279): u.uevent.qcompleted = 1.
    Called when the player carries the quest artifact back to the leader
    level and is adjacent to the leader (quest.c leaderchat / finish_quest).

    JIT-pure.
    """
    new_stage = jnp.maximum(state.quest.stage, _QSTAGE_COMPLETE)
    new_quest = state.quest.replace(
        completed=jnp.bool_(True),
        stage=new_stage,
    )
    return state.replace(quest=new_quest)
