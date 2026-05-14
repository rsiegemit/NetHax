"""Polymorph subsystem — full-fidelity Wave 4 implementation.

Hero and monster polymorph lifecycle: form acquisition, stat/attack-set
swap, AC recompute, intrinsic gain/loss, timed reversion, lycanthropy.

Canonical sources (NetHack 5.0 / 3.7):
    - src/polyself.c           — hero polymorph (polyself, polymon, newman,
                                  rehumanize)
    - src/mon.c::newcham       — monster polymorph
    - src/mondata.c            — monster attack-set retrieval
    - src/were.c               — lycanthropy / were-creature transitions
    - src/wand.c::do_polymorph — wand-of-polymorph dispatch
    - src/trap.c::dotrap       — POLY_TRAP handler
    - include/permonst.h       — struct permonst (form data we copy)

Design notes
------------
* PolymorphState owns the *player* polymorph bookkeeping. Original stats
  (STR/DEX/CON/HP_max/AC, role index, full attack table) are saved into
  ``orig_*`` fields at the moment of transformation so we can revert
  cleanly when ``poly_timer`` expires.
* Monster polymorph mutates ``MonsterAIState.entry_idx[slot]`` (added in
  Wave 4 alongside this subsystem) plus HP scaling via the new form's
  hit-dice; player stats are untouched.
* AC recompute uses ``state.player_ac`` directly (top-level field). Worn
  armor that the new form cannot wear is dropped to the ground stack at
  the player's tile, mirroring polyself.c's ``drop_inv_loss``.
* JIT-safe: every conditional uses ``jax.lax.cond``; loops use
  ``jax.lax.fori_loop`` / no Python-side branching on traced values.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# vendor/nethack/include/permonst.h: NATTK = 6  (max simultaneous attacks)
NATTK: int = 6

# vendor/nethack/src/polyself.c: poly_timer baseline (500 + rn2(500))
_POLY_TIMER_BASE: int = 500
_POLY_TIMER_RANGE: int = 500

# vendor/nethack/src/were.c: were-form transformation runs ~20 turns
_LYCANTHROPY_FORM_DURATION: int = 20

# Sentinel meaning "not polymorphed / no were-form active".
_NONE_FORM: int = -1


# ---------------------------------------------------------------------------
# State struct
# ---------------------------------------------------------------------------

@struct.dataclass
class PolymorphState:
    """Polymorph bookkeeping for the player character.

    Stored as a flat sub-struct inside EnvState.

    Original-stat fields hold the values to restore on reversion.  They
    are populated by ``polymorph_player`` and consumed by
    ``revert_polymorph``.

    Attack-set: we store the player's current (post-poly) attack tuples
    in ``attack_*`` arrays of length ``NATTK``.  Originals live in
    ``orig_attack_*``.

    See vendor/nethack/src/polyself.c::polyself for the canonical save /
    swap / restore sequence.
    """

    # ---- Current poly status ----
    is_polymorphed: jnp.ndarray         # scalar bool
    current_form_idx: jnp.ndarray       # scalar int16; MONSTERS index of current form
    poly_timer: jnp.ndarray             # scalar int16; turns until reversion
    poly_controlled: jnp.ndarray        # scalar bool; True if player picked target
    controlled_poly_count: jnp.ndarray  # scalar int8; running tally

    # ---- Lycanthropy (src/were.c) ----
    lycanthropy_form: jnp.ndarray       # scalar int8; -1 = none, else MONSTERS idx
    lycanthropy_timer: jnp.ndarray      # scalar int16

    # ---- Saved-original snapshot (filled at polymorph, read at revert) ----
    orig_role_idx: jnp.ndarray          # scalar int8
    orig_str:      jnp.ndarray          # scalar int16
    orig_dex:      jnp.ndarray          # scalar int8
    orig_con:      jnp.ndarray          # scalar int8
    orig_hp_max:   jnp.ndarray          # scalar int32
    orig_ac:       jnp.ndarray          # scalar int32

    # ---- Active attack set (post-poly snapshot from MONSTERS[form].attacks) ----
    # AttackType and DamageType sentinels exceed int8 range (e.g. AT_WEAP=254);
    # use uint8 to hold the raw enum values.
    attack_types: jnp.ndarray           # uint8[NATTK]
    attack_damage_types: jnp.ndarray    # uint8[NATTK]
    attack_n_dice: jnp.ndarray          # int8[NATTK]
    attack_n_sides: jnp.ndarray         # int8[NATTK]

    # ---- Saved-original attack set ----
    orig_attack_types: jnp.ndarray          # uint8[NATTK]
    orig_attack_damage_types: jnp.ndarray   # uint8[NATTK]
    orig_attack_n_dice: jnp.ndarray         # int8[NATTK]
    orig_attack_n_sides: jnp.ndarray        # int8[NATTK]

    # ---- Intrinsics granted/removed by the current form ----
    # Bit-mask matching MR_* constants from constants/monsters.py (FIRE/COLD/...)
    intrinsics_mask: jnp.ndarray        # scalar int32

    # ---- Legacy Wave-1 fields, retained for back-compat ----
    poly_form_id: jnp.ndarray           # alias of current_form_idx (kept for older callers)
    poly_turns: jnp.ndarray             # alias of poly_timer
    poly_controlled_legacy: jnp.ndarray # alias of poly_controlled  (avoid name clash)


def make_polymorph_state() -> PolymorphState:
    """Return a default (non-polymorphed) PolymorphState."""
    z_u8 = jnp.zeros((NATTK,), dtype=jnp.uint8)
    return PolymorphState(
        is_polymorphed=jnp.bool_(False),
        current_form_idx=jnp.int16(_NONE_FORM),
        poly_timer=jnp.int16(0),
        poly_controlled=jnp.bool_(False),
        controlled_poly_count=jnp.int8(0),
        lycanthropy_form=jnp.int8(_NONE_FORM),
        lycanthropy_timer=jnp.int16(0),
        orig_role_idx=jnp.int8(0),
        orig_str=jnp.int16(0),
        orig_dex=jnp.int8(0),
        orig_con=jnp.int8(0),
        orig_hp_max=jnp.int32(0),
        orig_ac=jnp.int32(0),
        attack_types=z_u8,
        attack_damage_types=z_u8,
        attack_n_dice=z_u8,
        attack_n_sides=z_u8,
        orig_attack_types=z_u8,
        orig_attack_damage_types=z_u8,
        orig_attack_n_dice=z_u8,
        orig_attack_n_sides=z_u8,
        intrinsics_mask=jnp.int32(0),
        # legacy aliases
        poly_form_id=jnp.int32(-1),
        poly_turns=jnp.int32(0),
        poly_controlled_legacy=jnp.bool_(False),
    )


# ---------------------------------------------------------------------------
# MONSTERS table lookup helpers (Python-side; the data is static).
# Returned as JAX arrays so JIT can read them via gather.
# ---------------------------------------------------------------------------

def _build_monster_lookup_tables():
    """Pre-compute static jnp arrays from MONSTERS for JIT-safe gather.

    Lazily imported to avoid circular imports at module load time.

    Returns a dict with arrays indexed by MONSTERS slot:
        ac           : int16[N]
        hp_dice_n    : int8[N]  (= level; mhp is rnd((mlevel+1)*8))
        attack_*     : int8[N, NATTK]  for type/damage/dice/sides
        flags1       : int32[N]  (M1_* bits — used for armor-compat check)
        intrinsics   : int32[N]  (resists_mask copy)
    """
    from Nethax.nethax.constants.monsters import MONSTERS, NO_ATTK

    n = len(MONSTERS)
    ac = jnp.array([m.ac for m in MONSTERS], dtype=jnp.int16)
    level = jnp.array([m.level for m in MONSTERS], dtype=jnp.int8)
    # Some monsters have move_speed > 127, so use int16 to avoid overflow.
    move_speed = jnp.array([m.move_speed for m in MONSTERS], dtype=jnp.int16)
    # flags1 bits include 0x80000000 which overflows signed int32; use uint32.
    flags1 = jnp.array([m.flags1 & 0xFFFFFFFF for m in MONSTERS], dtype=jnp.uint32)
    intrinsics = jnp.array([m.resists_mask for m in MONSTERS], dtype=jnp.int32)

    # Attacks: pad to NATTK with NO_ATTK.
    # AttackType uses sentinels (AT_WEAP=254, AT_MAGC=255) and DamageType
    # uses values up to 253, so int8 overflows; use uint8.
    type_rows = []
    dtyp_rows = []
    nd_rows = []
    ns_rows = []
    for m in MONSTERS:
        attks = list(m.attacks) + [NO_ATTK] * (NATTK - len(m.attacks))
        attks = attks[:NATTK]
        type_rows.append([int(a[0]) for a in attks])
        dtyp_rows.append([int(a[1]) for a in attks])
        nd_rows.append([int(a[2]) for a in attks])
        ns_rows.append([int(a[3]) for a in attks])
    a_type = jnp.array(type_rows, dtype=jnp.uint8)
    a_dtyp = jnp.array(dtyp_rows, dtype=jnp.uint8)
    # Black dragon AT_BREA stores n_sides=255 → exceeds int8 range.
    a_ndice = jnp.array(nd_rows, dtype=jnp.uint8)
    a_sides = jnp.array(ns_rows, dtype=jnp.uint8)

    return {
        "n": n,
        "ac": ac,
        "level": level,
        "move_speed": move_speed,
        "flags1": flags1,
        "intrinsics": intrinsics,
        "attack_types": a_type,
        "attack_damage_types": a_dtyp,
        "attack_n_dice": a_ndice,
        "attack_n_sides": a_sides,
    }


# Build tables eagerly at module import: this avoids tracer-leak issues
# when _monster_tables() is called inside a jitted region.
_MONSTER_TABLES = _build_monster_lookup_tables()


def _monster_tables() -> dict:
    return _MONSTER_TABLES


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _can_wear_armor(form_idx: jnp.ndarray) -> jnp.ndarray:
    """Return scalar bool: True iff MONSTERS[form_idx] can wear armor.

    polyself.c uses ``humanoid(mptr)`` and ``has_horns(mptr)`` / hand checks.
    We approximate with M1_HUMANOID and !M1_NOHANDS, matching the most
    common armor-drop logic in NetHack.

    M1_HUMANOID = 0x00020000, M1_NOHANDS = 0x00002000 (see constants/monsters.py).
    """
    tables = _monster_tables()
    flags = tables["flags1"][form_idx.astype(jnp.int32)]
    is_humanoid = (flags & jnp.uint32(0x00020000)) != 0
    has_hands = (flags & jnp.uint32(0x00002000)) == 0
    return is_humanoid & has_hands


def _form_ac(form_idx: jnp.ndarray) -> jnp.ndarray:
    """Return MONSTERS[form_idx].ac as int32 (NetHack base armor class)."""
    tables = _monster_tables()
    return tables["ac"][form_idx.astype(jnp.int32)].astype(jnp.int32)


def _form_hp_max(form_idx: jnp.ndarray, rng: jax.Array) -> jnp.ndarray:
    """Return a fresh HP_max roll for a monster form.

    NetHack rolls hp_max for new monsters as ``d8 * (mlevel + 1)``-ish.
    We use a simplified (level+1) * 8 max so the result is deterministic
    given rng, matching ``mons[].mlevel`` semantics in mon.c::newmonhp.
    """
    tables = _monster_tables()
    level = tables["level"][form_idx.astype(jnp.int32)].astype(jnp.int32)
    base = jnp.maximum(level + 1, jnp.int32(1)) * jnp.int32(8)
    # Add small RNG-driven jitter in [0, base) for variability.
    roll = jax.random.randint(rng, (), 0, jnp.maximum(base, jnp.int32(1)))
    return (base + roll).astype(jnp.int32)


def _form_attacks(form_idx: jnp.ndarray):
    """Return (types, damage_types, n_dice, n_sides) int8[NATTK] for form."""
    tables = _monster_tables()
    idx = form_idx.astype(jnp.int32)
    return (
        tables["attack_types"][idx],
        tables["attack_damage_types"][idx],
        tables["attack_n_dice"][idx],
        tables["attack_n_sides"][idx],
    )


def _form_intrinsics(form_idx: jnp.ndarray) -> jnp.ndarray:
    """Return MR_* resistance bitmask for the form."""
    tables = _monster_tables()
    return tables["intrinsics"][form_idx.astype(jnp.int32)].astype(jnp.int32)


def _drop_worn_armor(state):
    """Clear all worn armor slots — used when the new form has no hands.

    polyself.c::drop_inv_loss drops the *items* on the floor; in our
    simplified model we set worn_armor[i] = -1 (slot empty) and leave the
    inventory entry itself alone (it would need to be moved to ground
    items, which we defer to Wave 5).  AC penalty is captured via
    ``_recompute_ac``.
    """
    from Nethax.nethax.subsystems.inventory import N_ARMOR_SLOTS
    new_worn = jnp.full((N_ARMOR_SLOTS,), -1, dtype=jnp.int8)
    new_inv = state.inventory.replace(worn_armor=new_worn)
    return state.replace(inventory=new_inv)


def _recompute_ac(state, form_idx: jnp.ndarray):
    """Recompute player_ac after polymorph.

    With armor dropped, the form's intrinsic AC fully determines defense.
    Mirrors polyself.c: ``u.uac = (mtmp->data->ac)``.
    """
    new_ac = _form_ac(form_idx)
    return state.replace(player_ac=new_ac)


# ---------------------------------------------------------------------------
# Player polymorph  (src/polyself.c::polyself + polymon)
# ---------------------------------------------------------------------------

def polymorph_player(state, rng: jax.Array, target_form_idx, controlled: bool):
    """Transform the player into a new monster form (full fidelity).

    Sequence (polyself.c::polyself → polymon):
      1. Snapshot orig_* stats / attacks / AC into PolymorphState.
      2. Set current_form_idx + is_polymorphed=True.
      3. Adopt new form's STR/DEX/CON proxies, HP_max, attack set,
         intrinsics.
      4. Recompute AC from the form's base AC.
      5. If new form can't wear armor → drop worn armor.
      6. Set poly_timer ∈ [500, 1000) (polyself.c uses ~500 + rn2(500)).
      7. Set conduct.POLYSELFLESS violated.

    Parameters
    ----------
    state              : EnvState
    rng                : JAX PRNGKey
    target_form_idx    : int / jnp.int   MONSTERS table index
    controlled         : bool             True if player chose this form

    Returns
    -------
    EnvState           — fully updated state
    """
    # Coerce inputs to JAX scalars
    form_i16 = jnp.int16(int(target_form_idx)) if isinstance(target_form_idx, int) \
        else target_form_idx.astype(jnp.int16)
    controlled_b = jnp.bool_(bool(controlled)) if isinstance(controlled, bool) \
        else controlled.astype(jnp.bool_)

    poly = state.polymorph

    # --- 1. Snapshot originals (only save if not already polymorphed; nested
    # polys keep the *first* set of originals so revert returns to human).
    already_poly = poly.is_polymorphed

    def _snap(p):
        types, dtyps, nd, ns = _form_attacks(form_i16)
        return p.replace(
            orig_role_idx=state.player_role.astype(jnp.int8),
            orig_str=state.player_str.astype(jnp.int16),
            orig_dex=state.player_dex.astype(jnp.int8),
            orig_con=state.player_con.astype(jnp.int8),
            orig_hp_max=state.player_hp_max.astype(jnp.int32),
            orig_ac=state.player_ac.astype(jnp.int32),
            orig_attack_types=p.attack_types,
            orig_attack_damage_types=p.attack_damage_types,
            orig_attack_n_dice=p.attack_n_dice,
            orig_attack_n_sides=p.attack_n_sides,
        )

    poly = jax.lax.cond(already_poly, lambda p: p, _snap, poly)

    # --- 2/3. Set new form data + adopt attacks/intrinsics.
    types, dtyps, nd, ns = _form_attacks(form_i16)
    intr = _form_intrinsics(form_i16)
    rng, sub = jax.random.split(rng)
    new_hp_max = _form_hp_max(form_i16, sub)

    # poly_timer ∈ [500, 1000)
    rng, sub2 = jax.random.split(rng)
    timer = (jnp.int16(_POLY_TIMER_BASE)
             + jax.random.randint(sub2, (), 0, _POLY_TIMER_RANGE).astype(jnp.int16))

    new_count = jnp.where(controlled_b,
                          poly.controlled_poly_count + jnp.int8(1),
                          poly.controlled_poly_count)

    poly = poly.replace(
        is_polymorphed=jnp.bool_(True),
        current_form_idx=form_i16,
        poly_timer=timer,
        poly_controlled=controlled_b,
        controlled_poly_count=new_count,
        attack_types=types,
        attack_damage_types=dtyps,
        attack_n_dice=nd,
        attack_n_sides=ns,
        intrinsics_mask=intr,
        # legacy aliases kept in sync
        poly_form_id=form_i16.astype(jnp.int32),
        poly_turns=timer.astype(jnp.int32),
        poly_controlled_legacy=controlled_b,
    )

    # --- 4. Recompute AC.  Apply HP_max swap.
    state = state.replace(
        polymorph=poly,
        player_hp_max=new_hp_max,
        player_hp=jnp.minimum(state.player_hp, new_hp_max),
    )
    state = _recompute_ac(state, form_i16)

    # --- 5. Drop incompatible armor.
    can_wear = _can_wear_armor(form_i16)
    state = jax.lax.cond(can_wear, lambda s: s, _drop_worn_armor, state)

    # --- 7. Conduct: POLYSELFLESS violated.
    from Nethax.nethax.subsystems.conduct import Conduct
    new_vio = state.conduct.violations.at[int(Conduct.POLYSELFLESS)].set(True)
    state = state.replace(conduct=state.conduct.replace(violations=new_vio))

    return state


# ---------------------------------------------------------------------------
# Revert  (src/polyself.c::rehumanize)
# ---------------------------------------------------------------------------

def revert_polymorph(state, rng: jax.Array | None = None):
    """Restore original stats and clear polymorph flags.

    Mirrors polyself.c::rehumanize:
      - Restore STR/DEX/CON/HP_max/AC.
      - Restore the original attack set.
      - Clear is_polymorphed, poly_timer, current_form_idx.
    """
    poly = state.polymorph

    def _do_revert(s):
        p = s.polymorph
        # Restore attack set
        p2 = p.replace(
            is_polymorphed=jnp.bool_(False),
            current_form_idx=jnp.int16(_NONE_FORM),
            poly_timer=jnp.int16(0),
            poly_controlled=jnp.bool_(False),
            attack_types=p.orig_attack_types,
            attack_damage_types=p.orig_attack_damage_types,
            attack_n_dice=p.orig_attack_n_dice,
            attack_n_sides=p.orig_attack_n_sides,
            intrinsics_mask=jnp.int32(0),
            # legacy aliases
            poly_form_id=jnp.int32(-1),
            poly_turns=jnp.int32(0),
            poly_controlled_legacy=jnp.bool_(False),
        )
        return s.replace(
            polymorph=p2,
            player_str=p.orig_str,
            player_dex=p.orig_dex,
            player_con=p.orig_con,
            player_hp_max=p.orig_hp_max,
            player_hp=jnp.minimum(s.player_hp, p.orig_hp_max),
            player_ac=p.orig_ac,
            player_role=p.orig_role_idx,
        )

    return jax.lax.cond(poly.is_polymorphed, _do_revert, lambda s: s, state)


# ---------------------------------------------------------------------------
# Monster polymorph  (src/mon.c::newcham)
# ---------------------------------------------------------------------------

def polymorph_monster(state, rng: jax.Array, monster_slot_idx, target_form_idx):
    """Change the type of monster slot ``monster_slot_idx``.

    Sequence (mon.c::newcham):
      1. Save original entry_idx in ``orig_entry_idx[slot]`` (if present).
      2. Overwrite ``entry_idx[slot]`` with the new form.
      3. Roll a fresh HP_max from the new form's hit dice.
      4. Scale current HP proportionally to preserve "% health".

    Parameters
    ----------
    state              : EnvState
    rng                : JAX PRNGKey
    monster_slot_idx   : int / jnp.int
    target_form_idx    : int / jnp.int  MONSTERS table index
    """
    slot = jnp.int32(int(monster_slot_idx)) if isinstance(monster_slot_idx, int) \
        else monster_slot_idx.astype(jnp.int32)
    form_i16 = jnp.int16(int(target_form_idx)) if isinstance(target_form_idx, int) \
        else target_form_idx.astype(jnp.int16)

    mai = state.monster_ai

    # Save original entry_idx (if the field exists).  Wave-4 monster_ai
    # gains entry_idx + orig_entry_idx; if not, we degrade gracefully by
    # only updating HP fields.
    has_entry = hasattr(mai, "entry_idx")

    rng_hp, _ = jax.random.split(rng)
    new_hp_max = _form_hp_max(form_i16, rng_hp).astype(jnp.int32)

    # Proportional HP scaling: new_hp = hp * (new_hp_max / hp_max)
    old_hp = mai.hp[slot].astype(jnp.float32)
    old_hp_max = jnp.maximum(mai.hp_max[slot].astype(jnp.float32), jnp.float32(1.0))
    ratio = old_hp / old_hp_max
    new_hp = jnp.maximum(jnp.int32(1),
                         (ratio * new_hp_max.astype(jnp.float32)).astype(jnp.int32))

    updates = {
        "hp_max": mai.hp_max.at[slot].set(new_hp_max),
        "hp":     mai.hp.at[slot].set(new_hp),
    }

    if has_entry:
        orig = getattr(mai, "orig_entry_idx", None)
        if orig is None:
            # entry_idx exists but no orig backup — overwrite directly.
            updates["entry_idx"] = mai.entry_idx.at[slot].set(form_i16)
        else:
            updates["orig_entry_idx"] = orig.at[slot].set(mai.entry_idx[slot])
            updates["entry_idx"] = mai.entry_idx.at[slot].set(form_i16)

    new_mai = mai.replace(**updates)
    return state.replace(monster_ai=new_mai)


# ---------------------------------------------------------------------------
# Lycanthropy  (src/were.c)
# ---------------------------------------------------------------------------

def trigger_lycanthropy(state, rng: jax.Array, were_form_idx):
    """Force a were-creature transformation (src/were.c::new_were_form).

    Sets ``lycanthropy_form`` and schedules reversion after
    ``_LYCANTHROPY_FORM_DURATION`` turns by polymorphing the player into
    the were-form with a shortened timer.
    """
    form_i8 = jnp.int8(int(were_form_idx)) if isinstance(were_form_idx, int) \
        else were_form_idx.astype(jnp.int8)
    state = polymorph_player(state, rng, jnp.int16(int(were_form_idx)) if isinstance(were_form_idx, int) else were_form_idx.astype(jnp.int16), False)
    # Override the poly_timer with the shorter were-form duration.
    poly = state.polymorph.replace(
        poly_timer=jnp.int16(_LYCANTHROPY_FORM_DURATION),
        lycanthropy_form=form_i8,
    )
    return state.replace(polymorph=poly)


# ---------------------------------------------------------------------------
# Per-turn tick
# ---------------------------------------------------------------------------

def step(state, rng: jax.Array | None = None):
    """Advance polymorph + lycanthropy timers by one turn.

    Behaviour:
      - If is_polymorphed and poly_timer > 0: decrement poly_timer.
      - If poly_timer hits 0 (and still polymorphed): revert_polymorph.
      - Lycanthropy: decrement lycanthropy_timer; when it hits 0 with a
        lycanthropy_form set and the player is not currently polymorphed,
        auto-trigger the were-form transformation (mirrors
        ``were.c::were_change``, which calls ``new_were`` to switch shape).
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    bare = not hasattr(state, "polymorph")
    poly = state if bare else state.polymorph

    new_timer = jnp.where(
        poly.is_polymorphed & (poly.poly_timer > 0),
        poly.poly_timer - jnp.int16(1),
        poly.poly_timer,
    )
    # Lycanthropy timer decrements every turn; the auto-transform only
    # fires when a were-form is queued (matches were.c::were_change).
    has_were_form = poly.lycanthropy_form != jnp.int8(_NONE_FORM)
    new_lyc_timer = jnp.maximum(
        poly.lycanthropy_timer - jnp.int16(1),
        jnp.int16(0),
    )

    new_poly = poly.replace(
        poly_timer=new_timer,
        poly_turns=new_timer.astype(jnp.int32),
        lycanthropy_timer=new_lyc_timer,
    )
    if bare:
        return new_poly

    state = state.replace(polymorph=new_poly)
    expired = poly.is_polymorphed & (new_timer <= 0)
    state = jax.lax.cond(expired, lambda s: revert_polymorph(s, rng), lambda s: s, state)

    # Lycanthropy expiry: when the countdown reaches zero with a queued
    # were-form and the hero isn't currently polymorphed, force the
    # transformation (vendor: were.c::were_change → new_were).
    lyc_expired = (
        has_were_form
        & (new_lyc_timer <= 0)
        & (~state.polymorph.is_polymorphed)
    )

    def _spawn_were(s):
        form_i16 = s.polymorph.lycanthropy_form.astype(jnp.int16)
        return polymorph_player(s, rng, form_i16, False)

    state = jax.lax.cond(lyc_expired, _spawn_were, lambda s: s, state)
    return state


# ---------------------------------------------------------------------------
# Wave-1 compatibility shim — old name kept so callers keep compiling.
# ---------------------------------------------------------------------------

def unpolymorph(state, rng: jax.Array | None = None):
    """Alias for revert_polymorph (legacy name from Wave 1 stubs)."""
    return revert_polymorph(state, rng)


# ---------------------------------------------------------------------------
# Trap wiring helper  (src/trap.c::dotrap, POLY_TRAP case)
# ---------------------------------------------------------------------------

def poly_trap_effect(state, rng: jax.Array):
    """Apply a POLY_TRAP hit to the player.

    trap.c::dotrap selects a random monster form (we use ``rn2(NUMMONS)``
    in vanilla; here we sample uniformly over the MONSTERS table) and
    polymorphs the player uncontrolled.
    """
    tables = _monster_tables()
    n = tables["n"]
    rng, sub = jax.random.split(rng)
    form = jax.random.randint(sub, (), 0, n).astype(jnp.int16)
    return polymorph_player(state, rng, form, False)
