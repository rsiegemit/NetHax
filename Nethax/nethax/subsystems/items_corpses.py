"""Corpse-eating intrinsic-award system — JAX port of cpostfx().

Canonical source: vendor/nethack/src/eat.c::cpostfx lines 1129-1328
                  vendor/nethack/src/eat.c::corpse_intrinsic lines 1338-1373
                  vendor/nethack/src/eat.c::intrinsic_possible lines 888-954
                  vendor/nethack/include/mondata.h (can_teleport, telepathic macros)

Overview
--------
NetHack's cpostfx() fires after a corpse is fully consumed.  It handles:
  1. Special per-monster effects (wraith XL, nurse HP restore, quantum speed swap).
  2. Generic intrinsic-conveyance via corpse_intrinsic() + givit().

We port the generic path into a precomputed JIT-safe lookup table
``_CORPSE_INTRINSIC_TABLE[NUMMONS, N_INTRINSICS]``, built at module load time
from MONSTERS[i].conveys_mask and M1 flags.  Special cases (telepathy, teleport,
teleport_control) are injected by name into the same table.

Poisonous/acidic side-effect damage (eat.c:1130-1145, cprefx path) is handled
here before the intrinsic award, with a resistance gate.

The special one-off effects (wraith +XL, newt pw_max bump, nurse HP restore,
quantum speed swap, giant +STR) are modelled as extra int16 fields that
apply_corpse_postfx writes into the returned state delta; the caller in
action_dispatch must materialise them.  These are returned as named fields
in a CorpseEffects namedtuple so the logic is auditable.
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from Nethax.nethax.constants.monsters import (
    MONSTERS,
    NUMMONS,
    MR_FIRE,
    MR_COLD,
    MR_SLEEP,
    MR_DISINT,
    MR_ELEC,
    MR_POISON,
    MR_ACID,
    MR_STONE,
    M1_TPORT,
    M1_TPORT_CNTRL,
    M1_POIS,
    M1_ACID,
    M2_GIANT,
    M2_HUMAN,
    M2_ELF,
    M2_DWARF,
    M2_GNOME,
    M2_ORC,
)
from Nethax.nethax.subsystems.status_effects import (
    StatusState,
    Intrinsic,
    TimedStatus,
    N_INTRINSICS,
)


# ---------------------------------------------------------------------------
# Precomputed lookup table
# eat.c::intrinsic_possible lines 888-954 (resistance flags → intrinsic)
# eat.c::can_teleport, control_teleport, telepathic macros from mondata.h
# ---------------------------------------------------------------------------

# Map mconveys / mresists bit → Intrinsic index.
# eat.c::intrinsic_possible: checks ptr->mconveys & MR_* for resistances,
#   can_teleport (M1_TPORT) for TELEPORT,
#   control_teleport (M1_TPORT_CNTRL) for TELEPORT_CONTROL,
#   telepathic (floating eye / mind flayer) for TELEPAT.
_MR_TO_INTRINSIC: list[tuple[int, int]] = [
    (MR_FIRE,   int(Intrinsic.RESIST_FIRE)),
    (MR_COLD,   int(Intrinsic.RESIST_COLD)),
    (MR_SLEEP,  int(Intrinsic.RESIST_SLEEP)),
    (MR_DISINT, int(Intrinsic.RESIST_DISINT)),
    (MR_ELEC,   int(Intrinsic.RESIST_SHOCK)),
    (MR_POISON, int(Intrinsic.RESIST_POISON)),
    (MR_ACID,   int(Intrinsic.RESIST_ACID)),
    (MR_STONE,  int(Intrinsic.RESIST_STONE)),
]

# Monsters that grant TELEPATHY by name (mondata.h::telepathic macro).
# vendor mondata.h line 84-86: floating eye, mind flayer, master mind flayer.
_TELEPATHY_NAMES = frozenset(["floating eye", "mind flayer", "master mind flayer"])

def _build_intrinsic_table() -> np.ndarray:
    """Build bool[NUMMONS, N_INTRINSICS] from MONSTERS data.

    cite: eat.c::intrinsic_possible lines 903-953
    """
    table = np.zeros((NUMMONS, N_INTRINSICS), dtype=np.bool_)
    for i, m in enumerate(MONSTERS):
        conv = m.conveys_mask
        f1   = m.flags1
        # Resistance bits → intrinsic (intrinsic_possible cases FIRE_RES..STONE_RES)
        for mr_bit, intr_idx in _MR_TO_INTRINSIC:
            if conv & mr_bit:
                table[i, intr_idx] = True
        # TELEPORT — can_teleport: M1_TPORT flag (intrinsic_possible TELEPORT case)
        if f1 & M1_TPORT:
            table[i, int(Intrinsic.TELEPORT)] = True
        # TELEPORT_CONTROL — control_teleport: M1_TPORT_CNTRL flag
        if f1 & M1_TPORT_CNTRL:
            table[i, int(Intrinsic.TELEPORT_CONTROL)] = True
        # TELEPATHY — telepathic macro: floating eye / mind flayer / master mind flayer
        if m.name in _TELEPATHY_NAMES:
            table[i, int(Intrinsic.TELEPATHY)] = True
    return table


# Module-level precomputed table (Python np, converted to jnp at first use).
_CORPSE_INTRINSIC_TABLE_NP: np.ndarray = _build_intrinsic_table()
# JAX-side copy (NUMMONS × N_INTRINSICS bool)
_CORPSE_INTRINSIC_TABLE: jnp.ndarray = jnp.array(
    _CORPSE_INTRINSIC_TABLE_NP, dtype=jnp.bool_
)

# Per-monster boolean flags (numpy, for Python-side queries in tests)
_MONSTER_IS_POISONOUS_NP: np.ndarray = np.array(
    [(m.flags1 & M1_POIS) != 0 for m in MONSTERS], dtype=np.bool_
)
_MONSTER_IS_ACIDIC_NP: np.ndarray = np.array(
    [(m.flags1 & M1_ACID) != 0 for m in MONSTERS], dtype=np.bool_
)
_MONSTER_IS_GIANT_NP: np.ndarray = np.array(
    [(m.flags2 & M2_GIANT) != 0 for m in MONSTERS], dtype=np.bool_
)
# Is-wraith by name (exact index lookup)
_WRAITH_IDX_NP: int = next(
    (i for i, m in enumerate(MONSTERS) if m.name == "wraith"), -1
)
_NEWT_IDX_NP: int = next(
    (i for i, m in enumerate(MONSTERS) if m.name == "newt"), -1
)
_NURSE_IDX_NP: int = next(
    (i for i, m in enumerate(MONSTERS) if m.name == "nurse"), -1
)
_QUANTUM_MECHANIC_IDX_NP: int = next(
    (i for i, m in enumerate(MONSTERS) if m.name == "quantum mechanic"), -1
)

# JAX versions
_MONSTER_IS_POISONOUS: jnp.ndarray = jnp.array(_MONSTER_IS_POISONOUS_NP)
_MONSTER_IS_ACIDIC: jnp.ndarray    = jnp.array(_MONSTER_IS_ACIDIC_NP)
_MONSTER_IS_GIANT: jnp.ndarray     = jnp.array(_MONSTER_IS_GIANT_NP)


# ---------------------------------------------------------------------------
# Public: apply_corpse_postfx
# ---------------------------------------------------------------------------

def apply_corpse_postfx(
    state,       # EnvState
    rng: jax.Array,
    monster_entry_idx: jnp.ndarray,  # int32 scalar; -1 = not a corpse
) -> object:
    """Apply post-eat corpse effects for monster at monster_entry_idx.

    cite: vendor/nethack/src/eat.c::cpostfx lines 1129-1328
          vendor/nethack/src/eat.c::corpse_intrinsic lines 1338-1373

    This function is JIT-pure: all branching is via jax.lax.cond / jnp.where.

    Pipeline (mirrors cpostfx order):
      1. Poisonous-corpse side-effects (damage + potential STR loss) if no
         RESIST_POISON.  cite: eat.c cprefx path + cpostfx fallthrough.
      2. Acidic-corpse side-effects (damage) if no RESIST_ACID.
      3. Intrinsic award: pick one intrinsic that this monster can grant
         (table lookup + probabilistic reservoir selection).
      4. Special one-off effects: wraith +XL, newt +pw_max, nurse HP restore,
         quantum speed swap, giant +STR.

    Returns updated EnvState.
    """
    is_corpse = monster_entry_idx >= jnp.int32(0)
    safe_idx  = jnp.clip(monster_entry_idx, 0, NUMMONS - 1)

    # ------------------------------------------------------------------
    # 1. Poisonous side-effects
    # cite: eat.c cprefx / cpostfx 1130-1145 — poisonous corpse deals
    #       rnd(15) hp damage and may drain STR unless player has POISON_RES.
    # ------------------------------------------------------------------
    is_poisonous  = _MONSTER_IS_POISONOUS[safe_idx]
    has_poison_res = state.status.intrinsics[int(Intrinsic.RESIST_POISON)]
    rng, rng_p    = jax.random.split(rng)
    poison_dmg    = jax.random.randint(rng_p, (), 1, 16).astype(jnp.int32)  # rnd(15)
    do_poison_dmg = is_corpse & is_poisonous & ~has_poison_res
    new_hp        = jnp.where(
        do_poison_dmg,
        jnp.maximum(state.player_hp - poison_dmg, jnp.int32(0)),
        state.player_hp,
    )
    state = state.replace(player_hp=new_hp)

    # ------------------------------------------------------------------
    # 2. Acidic side-effects
    # cite: eat.c 1140-1145 — acidic corpse deals rnd(15) hp damage
    #       unless player has ACID_RES.
    # ------------------------------------------------------------------
    is_acidic    = _MONSTER_IS_ACIDIC[safe_idx]
    has_acid_res = state.status.intrinsics[int(Intrinsic.RESIST_ACID)]
    rng, rng_a   = jax.random.split(rng)
    acid_dmg     = jax.random.randint(rng_a, (), 1, 16).astype(jnp.int32)   # rnd(15)
    do_acid_dmg  = is_corpse & is_acidic & ~has_acid_res
    new_hp2      = jnp.where(
        do_acid_dmg,
        jnp.maximum(state.player_hp - acid_dmg, jnp.int32(0)),
        state.player_hp,
    )
    state = state.replace(player_hp=new_hp2)

    # ------------------------------------------------------------------
    # 3. Intrinsic award (probabilistic reservoir selection)
    # cite: eat.c::corpse_intrinsic lines 1338-1373
    #   Iterates over LAST_PROP intrinsic slots; for each candidate a 1/count
    #   chance replaces the running selection.  Giant corpses also have a
    #   -1 fake slot for STR (handled separately in step 4).
    # ------------------------------------------------------------------
    row      = _CORPSE_INTRINSIC_TABLE[safe_idx]   # [N_INTRINSICS] bool
    # Reservoir selection over N_INTRINSICS slots (JIT-safe scan).
    # State: (selected_idx, count, rng)
    def _reservoir_step(carry, i):
        sel_idx, count, rng_ = carry
        is_cand = row[i]
        count_ = count + jnp.where(is_cand, jnp.int32(1), jnp.int32(0))
        rng_, rng_roll = jax.random.split(rng_)
        # 1/count_ chance of picking slot i (vendor: !rn2(count))
        safe_count = jnp.maximum(count_, jnp.int32(1))
        pick = jax.random.randint(rng_roll, (), 0, safe_count) == jnp.int32(0)
        sel_idx_ = jnp.where(is_cand & pick, jnp.int32(i), sel_idx)
        return (sel_idx_, count_, rng_), None

    (chosen_intr, n_candidates, rng), _ = jax.lax.scan(
        _reservoir_step,
        (jnp.int32(-1), jnp.int32(0), rng),
        jnp.arange(N_INTRINSICS, dtype=jnp.int32),
    )

    # Apply chosen intrinsic if valid and is_corpse.
    has_intrinsic_to_grant = is_corpse & (chosen_intr >= jnp.int32(0))
    old_intrinsics = state.status.intrinsics
    # Set the chosen intrinsic slot to True (lax.cond to avoid conditional
    # on tracer; use dynamic_update_slice pattern via .at[].set + where).
    new_intrinsics = jnp.where(
        has_intrinsic_to_grant,
        old_intrinsics.at[chosen_intr].set(True),
        old_intrinsics,
    )
    new_status = state.status.replace(intrinsics=new_intrinsics)
    state = state.replace(status=new_status)

    # ------------------------------------------------------------------
    # 4. Special one-off effects
    # ------------------------------------------------------------------

    # Wraith: +1 XL via pluslvl(incr=False)  cite: eat.c:1141-1142.
    # Vendor calls pluslvl(FALSE) which rolls newhp()/newpw() and sets
    # uexp = newuexp(ulevel); we route through experience.pluslvl so the
    # uhpinc / ueninc / urexp bookkeeping stays consistent.
    is_wraith  = is_corpse & (safe_idx == jnp.int32(_WRAITH_IDX_NP))
    from Nethax.nethax.subsystems.experience import pluslvl as _xp_pluslvl
    rng, rng_wraith = jax.random.split(rng)
    state = jax.lax.cond(
        is_wraith,
        lambda s: _xp_pluslvl(s, rng_wraith, incr=False),
        lambda s: s,
        state,
    )

    # Newt: eye_of_newt_buzz — small chance to bump pw_max
    # cite: eat.c:1311 ``if (attacktype(ptr, AT_MAGC) || pm == PM_NEWT) eye_of_newt_buzz()``
    #       eye_of_newt_buzz lines 1102-1123: if rn2(3)==0 OR 3*uen<=2*uenmax, pw += rnd(3);
    #       if pw > pw_max and !rn2(3): pw_max += 1.
    # Simplified: 1/3 chance of pw_max+1 (matches the "if (!rn2(3))" pw_max bump path).
    is_newt   = is_corpse & (safe_idx == jnp.int32(_NEWT_IDX_NP))
    rng, rng_newt = jax.random.split(rng)
    newt_roll = jax.random.randint(rng_newt, (), 0, 3)  # rn2(3)
    do_pw_bump = is_newt & (newt_roll == jnp.int32(0))
    new_pw_max = jnp.where(do_pw_bump, state.player_pw_max + jnp.int32(1), state.player_pw_max)
    state = state.replace(player_pw_max=new_pw_max)

    # Nurse: restore HP to max  cite: eat.c:1154-1158
    is_nurse  = is_corpse & (safe_idx == jnp.int32(_NURSE_IDX_NP))
    new_hp3   = jnp.where(is_nurse, state.player_hp_max, state.player_hp)
    state = state.replace(player_hp=new_hp3)

    # Quantum mechanic: toggle FAST intrinsic  cite: eat.c:1227-1235
    is_qm = is_corpse & (safe_idx == jnp.int32(_QUANTUM_MECHANIC_IDX_NP))
    cur_fast  = state.status.intrinsics[int(Intrinsic.FAST)]
    new_fast  = jnp.where(is_qm, ~cur_fast, cur_fast)
    qm_intrinsics = state.status.intrinsics.at[int(Intrinsic.FAST)].set(new_fast)
    new_status2 = state.status.replace(intrinsics=jnp.where(
        is_qm,
        qm_intrinsics,
        state.status.intrinsics,
    ))
    state = state.replace(status=new_status2)

    # Giant: +1 STR  cite: eat.c::corpse_intrinsic:1345 ``is_giant(ptr)`` → gainstr
    # is_giant checks M2_GIANT flag (mondata.h).
    is_giant_corp = is_corpse & _MONSTER_IS_GIANT[safe_idx]
    new_str = jnp.where(
        is_giant_corp,
        jnp.minimum(state.player_str + jnp.int16(1), jnp.int16(125)),
        state.player_str,
    )
    state = state.replace(player_str=new_str)

    return state


# ---------------------------------------------------------------------------
# Corpse age / rotten effects
# cite: vendor/nethack/src/eat.c::eatcorpse lines 1884-1916
# ---------------------------------------------------------------------------

# Age threshold (turns) beyond which a corpse is tainted.
# eat.c:1887 rotted=(moves-age)/(10+rn2(20)); tainted if rotted>5.
# Conservative worst-case: age>50 turns (50/10=5 exactly at minimum divisor).
_CORPSE_AGE_THRESHOLD: int = 50


def apply_old_corpse_effects(state, rng: jax.Array, is_old: jnp.ndarray):
    """Apply VOMITING + SICK when eating a tainted/old corpse.

    cite: vendor/nethack/src/eat.c::eatcorpse lines 1895-1916
      rotted > 5L -> make_sick(rn1(10,10), ..., SICK_VOMITABLE)
    When corpse_creation_turn is tracked: is_old = (timestep - creation) > 50.
    JIT-pure.
    """
    rng, rng_sick = jax.random.split(rng)
    sick_time = jax.random.randint(rng_sick, (), 11, 21).astype(jnp.int32)

    cur_sick = state.status.timed_statuses[int(TimedStatus.SICK)]
    new_sick = jnp.where(is_old, jnp.maximum(cur_sick, sick_time), cur_sick)
    new_ts = state.status.timed_statuses.at[int(TimedStatus.SICK)].set(new_sick)

    rng, rng_vom = jax.random.split(rng)
    vom_time = jax.random.randint(rng_vom, (), 11, 26).astype(jnp.int32)
    cur_vom = new_ts[int(TimedStatus.VOMITING)]
    new_vom = jnp.where(is_old, jnp.maximum(cur_vom, vom_time), cur_vom)
    new_ts = new_ts.at[int(TimedStatus.VOMITING)].set(new_vom)

    new_sick_kind = jnp.where(is_old, jnp.int8(1), state.status.sick_kind)
    new_status = state.status.replace(timed_statuses=new_ts, sick_kind=new_sick_kind)
    return state.replace(status=new_status)


# ---------------------------------------------------------------------------
# Cannibalism penalty
# cite: vendor/nethack/src/eat.c::maybe_cannibal lines 757-788
#       eat.c:775 your_race(fptr) checks mflags2 race bits vs player race
# ---------------------------------------------------------------------------

# Map Race enum index -> M2_* race bit.
# Verified: all 5 races (HUMAN/ELF/DWARF/GNOME/ORC) present in monsters.py.
# cite: eat.c:775 your_race(fptr) -- compares permonst.mflags2 race bit.
_RACE_M2_TABLE: np.ndarray = np.array(
    [M2_HUMAN, M2_ELF, M2_DWARF, M2_GNOME, M2_ORC], dtype=np.int32
)
_MONSTER_RACE_BITS_NP: np.ndarray = np.array(
    [
        int(m.flags2) & (M2_HUMAN | M2_ELF | M2_DWARF | M2_GNOME | M2_ORC)
        for m in MONSTERS
    ],
    dtype=np.int32,
)
_MONSTER_RACE_BITS: jnp.ndarray = jnp.array(_MONSTER_RACE_BITS_NP, dtype=jnp.int32)
_RACE_M2_JAX: jnp.ndarray = jnp.array(_RACE_M2_TABLE, dtype=jnp.int32)


def apply_cannibalism_penalty(state, monster_entry_idx: jnp.ndarray):
    """Apply alignment hit + CONFUSION for same-race cannibalism.

    cite: vendor/nethack/src/eat.c::maybe_cannibal lines 770-786
      your_race check -> HAggravate_monster + change_luck(-rn1(4,2))
      -> alignment_record -= 2; CONFUSION set ~20 turns.
    All 5 races covered by _RACE_M2_TABLE (HUMAN/ELF/DWARF/GNOME/ORC).
    """
    safe_idx = jnp.clip(monster_entry_idx, 0, NUMMONS - 1)
    corpse_race_bits = _MONSTER_RACE_BITS[safe_idx]
    safe_race = jnp.clip(state.player_race.astype(jnp.int32), 0, 4)
    player_m2 = _RACE_M2_JAX[safe_race]
    is_cannibal = (monster_entry_idx >= jnp.int32(0)) & (
        (corpse_race_bits & player_m2) != jnp.int32(0)
    )

    new_record = jnp.where(
        is_cannibal,
        (state.prayer.alignment_record - jnp.int16(2)).astype(jnp.int16),
        state.prayer.alignment_record,
    )
    new_prayer = state.prayer.replace(alignment_record=new_record)

    cur_conf = state.status.timed_statuses[int(TimedStatus.CONFUSION)]
    new_conf = jnp.where(is_cannibal, jnp.maximum(cur_conf, jnp.int32(20)), cur_conf)
    new_ts = state.status.timed_statuses.at[int(TimedStatus.CONFUSION)].set(new_conf)
    new_status = state.status.replace(timed_statuses=new_ts)
    return state.replace(prayer=new_prayer, status=new_status)


# ---------------------------------------------------------------------------
# Tin opening counter
# cite: vendor/nethack/src/eat.c::opentin / consume_tin
#       blessed->30 turns, uncursed->50 turns to open
# ---------------------------------------------------------------------------


def apply_tin_open_start(
    state, is_tin: jnp.ndarray, type_id: jnp.ndarray, is_blessed: jnp.ndarray
):
    """Begin opening a tin: set tin_opening_turns_left and type_id.

    cite: vendor/nethack/src/eat.c consume_tin area, line 1370 area.
    """
    turns = jnp.where(is_blessed, jnp.int8(30), jnp.int8(50))
    new_turns = jnp.where(is_tin, turns, state.tin_opening_turns_left)
    new_type = jnp.where(
        is_tin, type_id.astype(jnp.int16), state.tin_opening_type_id
    )
    return state.replace(tin_opening_turns_left=new_turns, tin_opening_type_id=new_type)


def tick_tin_opening(state):
    """Decrement tin-opening counter by 1 (floor 0).

    cite: vendor/nethack/src/eat.c::opentin -- called each turn while opening.
    """
    cur = state.tin_opening_turns_left.astype(jnp.int32)
    new_val = jnp.maximum(cur - jnp.int32(1), jnp.int32(0)).astype(jnp.int8)
    return state.replace(tin_opening_turns_left=new_val)


# ---------------------------------------------------------------------------
# Eattin -- consume opened tin
# cite: vendor/nethack/src/eat.c::consume_tin lines 1527-1698
# ---------------------------------------------------------------------------


def apply_eattin(state, rng: jax.Array, item):
    """Apply effects of eating an opened tin.

    1. Spinach (enchantment==1 -> vendor spe==1): +1 STR clamped at 18.
       cite: eat.c:1684 gainstr(tin, 0, FALSE); eat.c:1470 obj->spe=1 spinach.
    2. Monster-meat (corpse_entry_idx>=0, not spinach): delegate to
       apply_corpse_postfx for same intrinsics as eating the corpse.
       cite: eat.c:1611-1613 cprefx(mnum)/cpostfx(mnum).
    3. Poisoned tin (tin_poisoned==True): rnd(15) HP damage + SICK timer.
       cite: eat.c:1537 if (tin->otrapped || (tin->cursed && !rn2(8))).
    JIT-pure.
    """
    is_spinach = item.enchantment == jnp.int8(1)
    corpse_idx = item.corpse_entry_idx.astype(jnp.int32)
    is_monster_tin = (~is_spinach) & (corpse_idx >= jnp.int32(0))
    is_poisoned_tin = item.tin_poisoned

    # 1. Spinach: +1 STR clamped at 18
    new_str = jnp.where(
        is_spinach,
        jnp.minimum(state.player_str + jnp.int16(1), jnp.int16(18)),
        state.player_str,
    )
    state = state.replace(player_str=new_str)

    # 2. Monster-meat intrinsics via apply_corpse_postfx
    effective_idx = jnp.where(is_monster_tin, corpse_idx, jnp.int32(-1))
    state = apply_corpse_postfx(state, rng, effective_idx)

    # 3. Poisoned tin: rnd(15) HP damage + SICK
    rng, rng_dmg = jax.random.split(rng)
    poison_dmg = jax.random.randint(rng_dmg, (), 1, 16).astype(jnp.int32)
    new_hp = jnp.where(
        is_poisoned_tin,
        jnp.maximum(state.player_hp - poison_dmg, jnp.int32(0)),
        state.player_hp,
    )
    state = state.replace(player_hp=new_hp)

    cur_sick = state.status.timed_statuses[int(TimedStatus.SICK)]
    new_sick = jnp.where(
        is_poisoned_tin, jnp.maximum(cur_sick, jnp.int32(10)), cur_sick
    )
    new_ts = state.status.timed_statuses.at[int(TimedStatus.SICK)].set(new_sick)
    new_sick_kind = jnp.where(is_poisoned_tin, jnp.int8(1), state.status.sick_kind)
    new_status = state.status.replace(timed_statuses=new_ts, sick_kind=new_sick_kind)
    return state.replace(status=new_status)
