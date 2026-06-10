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
# Wave 17f: mind flayer / stalker / displacer beast special-case indices.
# Cite: vendor/nethack/src/eat.c::cpostfx lines 1162-1268.
_MIND_FLAYER_IDX_NP: int = next(
    (i for i, m in enumerate(MONSTERS) if m.name == "mind flayer"), -1
)
_MASTER_MIND_FLAYER_IDX_NP: int = next(
    (i for i, m in enumerate(MONSTERS) if m.name == "master mind flayer"), -1
)
_STALKER_IDX_NP: int = next(
    (i for i, m in enumerate(MONSTERS) if m.name == "stalker"), -1
)
_DISPLACER_BEAST_IDX_NP: int = next(
    (i for i, m in enumerate(MONSTERS) if m.name == "displacer beast"), -1
)
_KILLER_BEE_IDX_NP: int = next(
    (i for i, m in enumerate(MONSTERS) if m.name == "killer bee"), -1
)
_SCORPION_IDX_NP: int = next(
    (i for i, m in enumerate(MONSTERS) if m.name == "scorpion"), -1
)

# Per-monster mlevel (vendor permonst.mlevel) — gate input to ptr->mlevel > rn2(chance).
_MONSTER_MLEVEL_NP: np.ndarray = np.array(
    [int(m.level) for m in MONSTERS], dtype=np.int32
)
_MONSTER_MLEVEL: jnp.ndarray = jnp.array(_MONSTER_MLEVEL_NP)

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
    # 1. Acidic side-effects (vendor checks acid FIRST via else-if chain).
    # cite: eat.c::eatcorpse lines 1923-1927 —
    #     else if (acidic(&mons[mnum]) && !Acid_resistance) {
    #         tp++; losehp(rnd(15), "acidic corpse", KILLED_BY_AN);
    #     }
    # ------------------------------------------------------------------
    is_acidic    = _MONSTER_IS_ACIDIC[safe_idx]
    has_acid_res = state.status.intrinsics[int(Intrinsic.RESIST_ACID)]
    rng, rng_a   = jax.random.split(rng)
    acid_dmg     = jax.random.randint(rng_a, (), 1, 16).astype(jnp.int32)   # rnd(15)
    do_acid_dmg  = is_corpse & is_acidic & ~has_acid_res
    new_hp_acid  = jnp.where(
        do_acid_dmg,
        jnp.maximum(state.player_hp - acid_dmg, jnp.int32(0)),
        state.player_hp,
    )
    state = state.replace(player_hp=new_hp_acid)

    # ------------------------------------------------------------------
    # 2. Poisonous side-effects (vendor: only if NOT already taken acid branch).
    # cite: eat.c::eatcorpse lines 1928-1937 —
    #     else if (poisonous(&mons[mnum]) && rn2(5)) {
    #         tp++;
    #         if (!Poison_resistance)
    #             poison_strdmg(rnd(4), rnd(15), "poisonous corpse", KILLED_BY_AN);
    #     }
    # poison_strdmg deals rnd(4) STR damage AND rnd(15) HP damage.
    # ------------------------------------------------------------------
    is_poisonous   = _MONSTER_IS_POISONOUS[safe_idx]
    has_poison_res = state.status.intrinsics[int(Intrinsic.RESIST_POISON)]
    rng, rng_p_gate, rng_p_hp, rng_p_str = jax.random.split(rng, 4)
    # Vendor's `rn2(5)` is truthy when result is non-zero — 4/5 chance to trigger.
    poison_gate    = jax.random.randint(rng_p_gate, (), 0, 5, dtype=jnp.int32) != jnp.int32(0)
    poison_dmg     = jax.random.randint(rng_p_hp, (), 1, 16).astype(jnp.int32)  # rnd(15)
    poison_str_loss = jax.random.randint(rng_p_str, (), 1, 5).astype(jnp.int16)  # rnd(4)
    # Acid path takes precedence (vendor "else if").
    do_poison_dmg = is_corpse & is_poisonous & poison_gate & ~has_poison_res & ~do_acid_dmg
    new_hp        = jnp.where(
        do_poison_dmg,
        jnp.maximum(state.player_hp - poison_dmg, jnp.int32(0)),
        state.player_hp,
    )
    new_str_pois  = jnp.where(
        do_poison_dmg,
        jnp.maximum(state.player_str - poison_str_loss, jnp.int16(3)),
        state.player_str,
    )
    state = state.replace(player_hp=new_hp, player_str=new_str_pois)

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

    # Vendor mind flayer +INT branch (eat.c:1281-1291) does `break;` (NOT
    # fallthrough) on successful +INT bump — skipping the corpse_intrinsic /
    # givit call entirely.  We pre-compute the +INT roll here so we can
    # suppress the intrinsic award downstream.
    _mf_pre = is_corpse & (
        (safe_idx == jnp.int32(_MIND_FLAYER_IDX_NP))
        | (safe_idx == jnp.int32(_MASTER_MIND_FLAYER_IDX_NP))
    )
    rng, _rng_mf_pre = jax.random.split(rng)
    _mf_pre_roll = jax.random.randint(_rng_mf_pre, (), 0, 2, dtype=jnp.int32)
    _mf_int_lt_cap = state.player_int < jnp.int8(25)
    _mf_eats_brain = _mf_pre & _mf_int_lt_cap & (_mf_pre_roll == jnp.int32(0))

    # Wave 17f — should_givit chance gate (vendor eat.c::should_givit 961-989).
    #   chance = 15  (default)
    #         = 15   (POISON_RES; but 1 for killer-bee / scorpion if !rn2(4))
    #         = 10   (TELEPORT)
    #         = 12   (TELEPORT_CONTROL)
    #         = 1    (TELEPAT)
    # Vendor returns ``ptr->mlevel > rn2(chance)``.
    safe_chosen = jnp.maximum(chosen_intr, jnp.int32(0))
    # Per-intrinsic chance lookup (only differs for the four cases above).
    chance = jnp.where(safe_chosen == jnp.int32(int(Intrinsic.TELEPORT)),
                       jnp.int32(10),
                       jnp.where(safe_chosen == jnp.int32(int(Intrinsic.TELEPORT_CONTROL)),
                                 jnp.int32(12),
                                 jnp.where(safe_chosen == jnp.int32(int(Intrinsic.TELEPATHY)),
                                           jnp.int32(1),
                                           jnp.int32(15))))
    # killer-bee / scorpion POISON_RES fast-track: chance = 1 when !rn2(4).
    rng, rng_pres = jax.random.split(rng)
    is_bee_or_scorp = (safe_idx == jnp.int32(_KILLER_BEE_IDX_NP)) | (
        safe_idx == jnp.int32(_SCORPION_IDX_NP)
    )
    pres_fast = is_bee_or_scorp & (safe_chosen == jnp.int32(int(Intrinsic.RESIST_POISON))) & (
        jax.random.randint(rng_pres, (), 0, 4, dtype=jnp.int32) == jnp.int32(0)
    )
    chance = jnp.where(pres_fast, jnp.int32(1), chance)

    rng, rng_giv = jax.random.split(rng)
    safe_chance = jnp.maximum(chance, jnp.int32(1))
    mlev = _MONSTER_MLEVEL[safe_idx]
    chance_roll = jax.random.randint(rng_giv, (), 0, safe_chance, dtype=jnp.int32)
    pass_gate = mlev > chance_roll  # vendor: ptr->mlevel > rn2(chance)

    # Apply chosen intrinsic if valid AND is_corpse AND chance gate passes
    # AND not suppressed by the mind-flayer +INT branch (vendor break;).
    has_intrinsic_to_grant = (
        is_corpse & (chosen_intr >= jnp.int32(0)) & pass_gate & ~_mf_eats_brain
    )
    old_intrinsics = state.status.intrinsics
    # Set the chosen intrinsic slot to True (lax.cond to avoid conditional
    # on tracer; use dynamic_update_slice pattern via .at[].set + where).
    new_intrinsics = jnp.where(
        has_intrinsic_to_grant,
        old_intrinsics.at[chosen_intr].set(True),
        old_intrinsics,
    )
    # Wave 17f: tag the source as FROMOUTSIDE — vendor "HFire_resistance |= FROMOUTSIDE"
    # at eat.c:1015 (and identical lines for the other corpse-conveyed props).
    # Cite: vendor/nethack/src/eat.c::givit lines 1010-1080.
    from Nethax.nethax.subsystems.status_effects import FROMOUTSIDE as _FROMOUTSIDE
    cur_src = state.status.intrinsic_source
    new_src = jnp.where(
        has_intrinsic_to_grant,
        cur_src.at[chosen_intr].set((cur_src[chosen_intr] | jnp.int8(_FROMOUTSIDE)).astype(jnp.int8)),
        cur_src,
    )
    new_status = state.status.replace(intrinsics=new_intrinsics, intrinsic_source=new_src)
    state = state.replace(status=new_status)

    # Wave 17f — temp_givit (vendor eat.c::temp_givit 992-997 + givit 1007):
    #   givit() at eat.c:1007: `if (!should_givit && !temp_givit) return;`
    #   temp_givit returns TRUE only for STONE_RES (chance 6) or ACID_RES (chance 3).
    #   Crucially, temp_givit fires for the SELECTED intrinsic only — not both.
    # cite: vendor/nethack/src/eat.c::temp_givit lines 991-997.
    rng, rng_t_stone, rng_t_acid, rng_d_stone, rng_d_acid = jax.random.split(rng, 5)
    chose_stone = (chosen_intr == jnp.int32(int(Intrinsic.RESIST_STONE)))
    chose_acid  = (chosen_intr == jnp.int32(int(Intrinsic.RESIST_ACID)))
    # STONE_RES temp gain — gated on chosen intrinsic == RESIST_STONE.
    # Suppressed when mind-flayer +INT branch took the early-break path.
    stone_roll = jax.random.randint(rng_t_stone, (), 0, 6, dtype=jnp.int32)
    do_stone_temp = is_corpse & chose_stone & (mlev > stone_roll) & ~_mf_eats_brain
    # d(3,6) = sum of 3 dice each 1..6  → range [3, 18], triangular dist.
    # Roll 3 independent d6 to match vendor's distribution byte-equal.
    _stone_keys = jax.random.split(rng_d_stone, 3)
    stone_d36 = jnp.sum(jnp.stack([
        jax.random.randint(k, (), 1, 7, dtype=jnp.int32) for k in _stone_keys
    ])).astype(jnp.int32)
    cur_t_stone = state.status.timed_intrinsics[int(Intrinsic.RESIST_STONE)]
    new_t_stone = jnp.where(
        do_stone_temp,
        cur_t_stone + stone_d36,
        cur_t_stone,
    )
    # ACID_RES temp gain — gated on chosen intrinsic == RESIST_ACID.
    acid_roll = jax.random.randint(rng_t_acid, (), 0, 3, dtype=jnp.int32)
    do_acid_temp = is_corpse & chose_acid & (mlev > acid_roll) & ~_mf_eats_brain
    _acid_keys = jax.random.split(rng_d_acid, 3)
    acid_d36 = jnp.sum(jnp.stack([
        jax.random.randint(k, (), 1, 7, dtype=jnp.int32) for k in _acid_keys
    ])).astype(jnp.int32)
    cur_t_acid = state.status.timed_intrinsics[int(Intrinsic.RESIST_ACID)]
    new_t_acid = jnp.where(
        do_acid_temp,
        cur_t_acid + acid_d36,
        cur_t_acid,
    )
    new_timed = state.status.timed_intrinsics.at[int(Intrinsic.RESIST_STONE)].set(new_t_stone)
    new_timed = new_timed.at[int(Intrinsic.RESIST_ACID)].set(new_t_acid)
    state = state.replace(status=state.status.replace(timed_intrinsics=new_timed))

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

    # Nurse: restore HP to max + cure blindness.
    # cite: vendor/nethack/src/eat.c::cpostfx lines 1153-1160
    #   if (Upolyd) u.mh = u.mhmax; else u.uhp = u.uhpmax;
    #   make_blinded(0L, !u.ucreamed);
    is_nurse  = is_corpse & (safe_idx == jnp.int32(_NURSE_IDX_NP))
    new_hp3   = jnp.where(is_nurse, state.player_hp_max, state.player_hp)
    state = state.replace(player_hp=new_hp3)
    cur_blind = state.status.timed_statuses[int(TimedStatus.BLIND)]
    new_blind = jnp.where(is_nurse, jnp.int32(0), cur_blind)
    new_ts_nurse = state.status.timed_statuses.at[int(TimedStatus.BLIND)].set(new_blind)
    state = state.replace(status=state.status.replace(timed_statuses=new_ts_nurse))

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
    #
    # Vendor 50% gate when STR is the ONLY candidate
    # cite: vendor/nethack/src/eat.c::corpse_intrinsic lines 1368-1370 —
    #     /* if strength is the only candidate, give it 50% chance */
    #     if (conveys_STR && count == 1 && !rn2(2))
    #         prop = 0;
    # Vendor's `count` is (1 if conveys_STR) + n_intrinsic_table_candidates.
    # `count == 1` therefore means STR is the only candidate, i.e. our
    # n_candidates (the table-only count) is zero.  Per audit-E: apply +STR
    # when (n_candidates > 0) OR (rn2(2) == 0); equivalently, suppress STR
    # only when n_candidates == 0 AND rn2(2) != 0.
    is_giant_corp = is_corpse & _MONSTER_IS_GIANT[safe_idx]
    rng, rng_str_gate = jax.random.split(rng)
    str_50_roll = jax.random.randint(rng_str_gate, (), 0, 2, dtype=jnp.int32)
    str_only = n_candidates == jnp.int32(0)
    apply_str = is_giant_corp & ((~str_only) | (str_50_roll == jnp.int32(0)))
    new_str = jnp.where(
        apply_str,
        jnp.minimum(state.player_str + jnp.int16(1), jnp.int16(125)),
        state.player_str,
    )
    state = state.replace(player_str=new_str)

    # ------------------------------------------------------------------
    # Wave 17f — Mind flayer / master mind flayer
    # cite: vendor/nethack/src/eat.c::cpostfx lines 1281-1297:
    #     if (ABASE(A_INT) < ATTRMAX(A_INT)) {
    #         if (!rn2(2)) {
    #             (void) adjattrib(A_INT, 1, FALSE);
    #             break;
    #         }
    #     }
    #     /* falls through to default → may grant telepathy via corpse_intrinsic */
    # 50% chance of +1 INT (up to attrmax) when not yet capped, else passes
    # through to the generic intrinsic-award path above (which the table
    # already includes TELEPAT for mind flayers).
    # ------------------------------------------------------------------
    # Re-use the pre-computed roll from above so the +INT branch and the
    # intrinsic-suppression decision stay in lock-step.
    do_int_bump = _mf_eats_brain
    new_int = jnp.where(
        do_int_bump,
        jnp.minimum(state.player_int + jnp.int8(1), jnp.int8(25)),
        state.player_int,
    )
    state = state.replace(player_int=new_int)

    # ------------------------------------------------------------------
    # Wave 17f — Stalker (PM_STALKER).
    # cite: vendor/nethack/src/eat.c::cpostfx lines 1162-1172:
    #     if (!Invis) {
    #         set_itimeout(&HInvis, rn1(100, 50));
    #     } else {
    #         HInvis |= FROMOUTSIDE;
    #         HSee_invisible |= FROMOUTSIDE;
    #     }
    # Approximation: grant temporary INVIS (1..149 turns) and SEE_INVIS
    # intrinsic (FROMOUTSIDE source bit) when a stalker corpse is eaten.
    # ------------------------------------------------------------------
    is_stalker = is_corpse & (safe_idx == jnp.int32(_STALKER_IDX_NP))
    rng, rng_st = jax.random.split(rng)
    # rn1(100, 50) = 50 + rn2(100)  → uniform in [50, 149]
    invis_turns = jnp.int32(50) + jax.random.randint(rng_st, (), 0, 100, dtype=jnp.int32)
    cur_invis = state.status.timed_intrinsics[int(Intrinsic.INVIS)]
    new_invis_t = jnp.where(is_stalker, jnp.maximum(cur_invis, invis_turns), cur_invis)
    new_t_intr = state.status.timed_intrinsics.at[int(Intrinsic.INVIS)].set(new_invis_t)
    # SEE_INVIS as permanent intrinsic (FROMOUTSIDE) — see vendor 1171.
    new_intrinsics2 = jnp.where(
        is_stalker,
        state.status.intrinsics.at[int(Intrinsic.SEE_INVIS)].set(True),
        state.status.intrinsics,
    )
    # Stalker FALLTHROUGH to PM_GIANT_BAT/PM_BAT — make_stunned((HStun&TIMEOUT)+30).
    # cite: vendor/nethack/src/eat.c::cpostfx lines 1174-1183.
    cur_stun = state.status.timed_statuses[int(TimedStatus.STUNNED)]
    new_stun = jnp.where(is_stalker, cur_stun + jnp.int32(30), cur_stun)
    new_ts_stalk = state.status.timed_statuses.at[int(TimedStatus.STUNNED)].set(new_stun)
    new_status3 = state.status.replace(
        timed_intrinsics=new_t_intr,
        intrinsics=new_intrinsics2,
        timed_statuses=new_ts_stalk,
    )
    state = state.replace(status=new_status3)

    # ------------------------------------------------------------------
    # Wave 17f — Displacer beast (PM_DISPLACER_BEAST).
    # cite: vendor/nethack/src/eat.c::cpostfx lines 1265-1268:
    #     if (!Displaced) toggle_displacement(...);
    #     incr_itimeout(&HDisplaced, d(6, 6));
    # ------------------------------------------------------------------
    is_displacer = is_corpse & (safe_idx == jnp.int32(_DISPLACER_BEAST_IDX_NP))
    rng, rng_dp = jax.random.split(rng)
    # d(6,6) = sum of 6 d6 rolls (triangular distribution [6, 36], mean 21).
    # Byte-equal to vendor `d(6, 6)`.
    _dp_keys = jax.random.split(rng_dp, 6)
    disp_turns = jnp.sum(jnp.stack([
        jax.random.randint(k, (), 1, 7, dtype=jnp.int32) for k in _dp_keys
    ])).astype(jnp.int32)
    cur_disp = state.status.timed_intrinsics[int(Intrinsic.DISPLACED)]
    new_disp = jnp.where(is_displacer, cur_disp + disp_turns, cur_disp)
    new_t_intr2 = state.status.timed_intrinsics.at[int(Intrinsic.DISPLACED)].set(new_disp)
    state = state.replace(status=state.status.replace(timed_intrinsics=new_t_intr2))

    return state


# ---------------------------------------------------------------------------
# Corpse age / rotten effects
# cite: vendor/nethack/src/eat.c::eatcorpse lines 1884-1916
# ---------------------------------------------------------------------------

# Age threshold (turns) beyond which a corpse is tainted.
# eat.c:1887 rotted=(moves-age)/(10+rn2(20)); tainted if rotted>5.
# Conservative worst-case: age>50 turns (50/10=5 exactly at minimum divisor).
_CORPSE_AGE_THRESHOLD: int = 50


def compute_rotted(
    rng: jax.Array,
    moves: jnp.ndarray,
    creation_turn: jnp.ndarray,
    blessed: jnp.ndarray = jnp.bool_(False),
    cursed: jnp.ndarray = jnp.bool_(False),
) -> jnp.ndarray:
    """Compute the vendor rotted level for a corpse.

    Byte-equal to vendor/nethack/src/eat.c::eatcorpse line 1887::

        rotted = (svm.moves - age) / (10L + rn2(20));
        if (otmp->cursed)   rotted += 2L;
        else if (otmp->blessed) rotted -= 2L;

    The divisor is uniform in [10, 29] (10 + rn2(20)), so each rot stage
    spans 10..29 turns.  Vendor treats ``rotted > 5`` as tainted.

    Parameters
    ----------
    rng           : JAX PRNG key (single rn2(20) draw).
    moves         : current game turn (svm.moves), int32 scalar.
    creation_turn : corpse age (svm.moves-age = turns since creation), int32.
    blessed       : corpse blessed status (bool scalar).
    cursed        : corpse cursed status (bool scalar).

    Returns
    -------
    jnp.int32 — rotted level (can be negative for blessed fresh corpses).
    """
    divisor = jnp.int32(10) + jax.random.randint(rng, (), 0, 20, dtype=jnp.int32)
    age = jnp.maximum(moves - creation_turn, jnp.int32(0))
    rotted = age // divisor
    rotted = jnp.where(cursed, rotted + jnp.int32(2), rotted)
    rotted = jnp.where(blessed & ~cursed, rotted - jnp.int32(2), rotted)
    return rotted.astype(jnp.int32)


def apply_old_corpse_effects(state, rng: jax.Array, is_old: jnp.ndarray):
    """Apply VOMITING + SICK when eating a tainted/old corpse.

    cite: vendor/nethack/src/eat.c::eatcorpse lines 1895-1916
      rotted > 5L -> make_sick(rn1(10,10), ..., SICK_VOMITABLE)
    When corpse_creation_turn is tracked: is_old = (timestep - creation) > 50.
    JIT-pure.
    """
    # Vendor eat.c:1909 `sick_time = (long) rn1(10, 10)` = 10 + rn2(10) → [10,19].
    rng, rng_sick = jax.random.split(rng)
    sick_time = (jnp.int32(10) + jax.random.randint(rng_sick, (), 0, 10, dtype=jnp.int32))

    cur_sick = state.status.timed_statuses[int(TimedStatus.SICK)]
    new_sick = jnp.where(is_old, jnp.maximum(cur_sick, sick_time), cur_sick)
    new_ts = state.status.timed_statuses.at[int(TimedStatus.SICK)].set(new_sick)

    # Vendor make_sick(..., SICK_VOMITABLE) sets up vomiting indirectly via the
    # sick timer (see status.c). We keep a separate VOMITING timer for parity
    # with the existing tests; rn1(15, 10) used by tin rotten path = 10+rn2(15).
    rng, rng_vom = jax.random.split(rng)
    vom_time = (jnp.int32(10) + jax.random.randint(rng_vom, (), 0, 15, dtype=jnp.int32))
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

import os as _os_brax
import sys as _sys_brax
if _os_brax.environ.get("NETHAX_BRAX_ALL", "0") == "1":
    _BRAX_ORIG = {"apply_corpse_postfx": apply_corpse_postfx}
    _BRAX_MAP = {"apply_corpse_postfx": ("items_misc_brax", "apply_corpse_postfx_brax")}
    _BRAX_CACHE = {}
    for _name in list(_BRAX_MAP):
        if _name in globals(): del globals()[_name]
    def __getattr__(name):
        if name not in _BRAX_MAP:
            raise AttributeError(name)
        mod_name, brax_name = _BRAX_MAP[name]
        full = f"Nethax.nethax.subsystems.{mod_name}"
        if full in _sys_brax.modules:
            spec = getattr(_sys_brax.modules[full], "__spec__", None)
            if spec is not None and getattr(spec, "_initializing", False):
                return _BRAX_ORIG[name]
        if name not in _BRAX_CACHE:
            _BRAX_CACHE[name] = getattr(__import__(full, fromlist=[brax_name]), brax_name)
        return _BRAX_CACHE[name]
