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

# vendor/nethack/src/polyself.c:280 — Unchanging intrinsic bit (prop.h UNCHANGING=63)
UNCHANGING_MASK: int = 63  # index in status.intrinsics array


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
# Valid-form mask  (polyself.c:280 — choose_race / polyself filter logic)
# ---------------------------------------------------------------------------

def _build_poly_form_valid() -> jnp.ndarray:
    """Pre-compute bool[N_MONSTERS]: True iff a form is eligible for random poly.

    Filters out (polyself.c:280):
      - G_UNIQ monsters (Wizard of Yendor, Medusa, Riders, quest leaders, etc.)
      - M2_NOPOLY flagged monsters (werecreatures, some humanoids, shopkeepers)
      - Explicit Rider indices (Death, Pestilence, Famine) — also caught by G_UNIQ
        but named here for clarity, mirroring polyself.c's explicit rider check.

    Role-specific bans (Monk: no carnivore; Healer: no demon) are applied
    dynamically in choose_random_polymorph_form() using the state's role.
    """
    from Nethax.nethax.constants.monsters import MONSTERS, G_UNIQ, M2_NOPOLY

    n = len(MONSTERS)
    valid = []
    for i, m in enumerate(MONSTERS):
        is_uniq   = bool(m.generation_mask & G_UNIQ)
        is_nopoly = bool(m.flags2 & M2_NOPOLY)
        valid.append(not is_uniq and not is_nopoly)

    return jnp.array(valid, dtype=jnp.bool_)


_POLY_FORM_VALID: jnp.ndarray = _build_poly_form_valid()


def _build_form_hates_silver() -> jnp.ndarray:
    """Pre-compute bool[N_MONSTERS]: True iff form is harmed by silver.

    polyself.c::retouch_equipment — vampires (M2_UNDEAD+S_VAMPIRE),
    were-creatures (M2_WERE), and major demons (M2_DEMON) take burn damage
    from silver items.
    """
    from Nethax.nethax.constants.monsters import MONSTERS, M2_UNDEAD, M2_WERE, M2_DEMON, MonsterSymbol
    result = []
    for m in MONSTERS:
        hates = (
            bool(m.flags2 & M2_WERE)
            or bool(m.flags2 & M2_DEMON)
            or (bool(m.flags2 & M2_UNDEAD) and m.symbol == MonsterSymbol.S_VAMPIRE)
        )
        result.append(hates)
    return jnp.array(result, dtype=jnp.bool_)


_FORM_HATES_SILVER: jnp.ndarray = _build_form_hates_silver()


def _build_item_is_silver() -> jnp.ndarray:
    """Pre-compute bool[N_OBJECTS]: True iff the object is made of silver.

    polyself.c::retouch_equipment uses objects.c material checks.
    """
    from Nethax.nethax.constants.objects import OBJECTS, Material
    return jnp.array([o.material == Material.SILVER for o in OBJECTS], dtype=jnp.bool_)


_ITEM_IS_SILVER: jnp.ndarray = _build_item_is_silver()


def _build_form_flags2() -> jnp.ndarray:
    """Pre-compute int32[N_MONSTERS] of flags2 for JIT-safe gather."""
    from Nethax.nethax.constants.monsters import MONSTERS
    return jnp.array([m.flags2 & 0xFFFFFFFF for m in MONSTERS], dtype=jnp.uint32)


_FORM_FLAGS2: jnp.ndarray = _build_form_flags2()


def choose_random_polymorph_form(state, rng: jax.Array) -> jnp.ndarray:
    """Pick a random valid polymorph target form index.  JIT-pure.

    Vendor polyself.c:280 — rndmonst() filtered through poly_newcham() checks:
      - Skip G_UNIQ forms.
      - Skip M2_NOPOLY forms.
      - Role-specific bans:
          Monk  (role 9): M1_CARNIVORE forms banned.
          Healer (role 2): M2_DEMON forms banned.

    Uses lax.while_loop rejection sampling — statistically O(1) iterations
    since ~75% of forms are valid.

    Returns
    -------
    jnp.int32 scalar — MONSTERS table index of the chosen form.
    """
    from Nethax.nethax.constants.monsters import M1_CARNIVORE, M2_DEMON

    n = _MONSTER_TABLES["n"]
    flags1_arr = _MONSTER_TABLES["flags1"]   # uint32[N]
    flags2_arr = _FORM_FLAGS2                # uint32[N]

    # Role constants (Role enum indices matching vendor roles.h order)
    _ROLE_MONK   = jnp.int8(9)
    _ROLE_HEALER = jnp.int8(2)

    is_monk   = state.player_role.astype(jnp.int8) == _ROLE_MONK
    is_healer = state.player_role.astype(jnp.int8) == _ROLE_HEALER

    def _body(args):
        rng_inner, _form = args
        rng_inner, sub = jax.random.split(rng_inner)
        candidate = jax.random.randint(sub, (), 0, n).astype(jnp.int32)

        base_valid = _POLY_FORM_VALID[candidate]

        f1 = flags1_arr[candidate]
        f2 = flags2_arr[candidate]
        carnivore = (f1 & jnp.uint32(M1_CARNIVORE)) != jnp.uint32(0)
        is_demon  = (f2 & jnp.uint32(M2_DEMON))     != jnp.uint32(0)

        monk_ban   = is_monk   & carnivore
        healer_ban = is_healer & is_demon

        valid = base_valid & (~monk_ban) & (~healer_ban)
        # Keep candidate if valid, else keep -1 sentinel to loop again.
        chosen = jnp.where(valid, candidate, jnp.int32(-1))
        return rng_inner, chosen

    def _cond(args):
        _rng, form = args
        return form < jnp.int32(0)

    _, form = jax.lax.while_loop(_cond, _body, (rng, jnp.int32(-1)))
    return form.astype(jnp.int32)


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
    inventory entry itself alone.  AC penalty is captured via
    ``_recompute_ac``.

    Deprecated in favour of _drop_worn_armor_per_slot; retained as a
    fallback for non-per-slot callers.
    """
    from Nethax.nethax.subsystems.inventory import N_ARMOR_SLOTS
    new_worn = jnp.full((N_ARMOR_SLOTS,), -1, dtype=jnp.int8)
    new_inv = state.inventory.replace(worn_armor=new_worn)
    return state.replace(inventory=new_inv)


def _drop_worn_armor_per_slot(state, form_idx: jnp.ndarray):
    """Drop worn armor per-slot based on the new form's flags.

    vendor/nethack/src/polyself.c:1156 — break_armor() checks each worn
    slot against the new form's M1_NOHANDS / M1_NOHEAD / M1_SLITHY flags:

      M1_NOHANDS  → can't wear body/shield/gloves (all hand-dependent slots)
      M1_NOHEAD   → can't wear helm
      M1_SLITHY   → can't wear boots (no legs)
      M1_NOHANDS also covers helm/boots for fully limbless forms.

    For each incompatible slot:
      - Set worn_armor[slot] = -1.
      - Place the displaced item into ground_items at player_pos (first free
        stack slot, branch=0/level=0 for current level — Wave 6 simplification;
        full dungeon-level routing deferred to Wave 7).

    JIT-pure: uses jnp.where masks per slot.
    """
    from Nethax.nethax.subsystems.inventory import N_ARMOR_SLOTS, ArmorSlot

    tables   = _monster_tables()
    idx      = form_idx.astype(jnp.int32)
    f1       = tables["flags1"][idx]   # uint32

    M1_NOHANDS_U = jnp.uint32(0x00002000)
    M1_NOHEAD_U  = jnp.uint32(0x00008000)
    M1_SLITHY_U  = jnp.uint32(0x00080000)

    nohands = (f1 & M1_NOHANDS_U) != jnp.uint32(0)
    nohead  = (f1 & M1_NOHEAD_U)  != jnp.uint32(0)
    slithy  = (f1 & M1_SLITHY_U)  != jnp.uint32(0)

    # Per-slot incompatibility mask: True → must drop.
    # Slot order: BODY=0, SHIELD=1, HELM=2, GLOVES=3, BOOTS=4, CLOAK=5, SHIRT=6
    # nohands blocks body(0), shield(1), gloves(3); nohead blocks helm(2);
    # slithy blocks boots(4); nohands also blocks helm/boots for fully limbless.
    drop_mask = jnp.array([
        nohands,        # BODY
        nohands,        # SHIELD
        nohands | nohead,  # HELM
        nohands,        # GLOVES
        nohands | slithy,  # BOOTS
        jnp.bool_(False),  # CLOAK — no vendor restriction
        jnp.bool_(False),  # SHIRT — no vendor restriction
    ], dtype=jnp.bool_)

    worn      = state.inventory.worn_armor   # int8[N_ARMOR_SLOTS]
    new_worn  = jnp.where(drop_mask, jnp.int8(-1), worn)
    new_inv   = state.inventory.replace(worn_armor=new_worn)
    state     = state.replace(inventory=new_inv)

    # Move displaced items to ground at player_pos (branch 0, level 0).
    # We iterate over slots using lax.fori_loop to stay JIT-pure.
    ground = state.ground_items
    p_row  = state.player_pos[0].astype(jnp.int32)
    p_col  = state.player_pos[1].astype(jnp.int32)

    def _drop_slot(slot_i, carry):
        g, inv_items = carry
        was_worn = worn[slot_i].astype(jnp.int32)  # inv slot idx, or -1
        should_drop = drop_mask[slot_i] & (was_worn >= jnp.int32(0))

        # Find first free ground stack position (category == 0).
        ground_stack = g.category[0, 0, p_row, p_col]  # [MAX_GROUND_STACK]
        free_idx = jnp.argmax(ground_stack == jnp.int8(0)).astype(jnp.int32)

        # Copy item from inventory to ground stack.
        item_cat = inv_items.category[was_worn]
        item_tid = inv_items.type_id[was_worn]

        new_g_cat = jnp.where(
            should_drop,
            g.category[0, 0, p_row, p_col].at[free_idx].set(item_cat),
            g.category[0, 0, p_row, p_col],
        )
        new_g_tid = jnp.where(
            should_drop,
            g.type_id[0, 0, p_row, p_col].at[free_idx].set(item_tid),
            g.type_id[0, 0, p_row, p_col],
        )
        g = g.replace(
            category=g.category.at[0, 0, p_row, p_col].set(new_g_cat),
            type_id=g.type_id.at[0, 0, p_row, p_col].set(new_g_tid),
        )
        return g, inv_items

    new_ground, _ = jax.lax.fori_loop(
        0, N_ARMOR_SLOTS, _drop_slot, (ground, state.inventory.items)
    )
    return state.replace(ground_items=new_ground)


# ---------------------------------------------------------------------------
# retouch_equipment()  (vendor/nethack/src/polyself.c::retouch_equipment)
# ---------------------------------------------------------------------------

def _retouch_equipment_silver(state, form_idx: jnp.ndarray, rng: jax.Array):
    """Drop silver worn items and apply burn damage for silver-allergic forms.

    polyself.c::retouch_equipment — when polymorphing into a form that hates
    silver (vampires, were-creatures, demons), each worn item made of silver
    is dropped to ground_items and deals 1d6 burn damage per item.

    JIT-pure: fori_loop over armor slots.
    """
    from Nethax.nethax.subsystems.inventory import N_ARMOR_SLOTS

    idx = form_idx.astype(jnp.int32)
    form_hates = _FORM_HATES_SILVER[idx]

    worn = state.inventory.worn_armor
    ground = state.ground_items
    p_row = state.player_pos[0].astype(jnp.int32)
    p_col = state.player_pos[1].astype(jnp.int32)
    n_objects = _ITEM_IS_SILVER.shape[0]

    def _check_slot(slot_i, carry):
        new_worn, g, dmg_acc, rng_c = carry

        inv_idx = worn[slot_i].astype(jnp.int32)
        occupied = inv_idx >= jnp.int32(0)

        type_id = state.inventory.items.type_id[inv_idx].astype(jnp.int32)
        safe_tid = jnp.where(occupied, jnp.clip(type_id, 0, n_objects - 1), jnp.int32(0))
        is_silver = _ITEM_IS_SILVER[safe_tid] & occupied

        should_drop = form_hates & is_silver

        ground_stack_cat = g.category[0, 0, p_row, p_col]
        free_idx = jnp.argmax(ground_stack_cat == jnp.int8(0)).astype(jnp.int32)

        item_cat = state.inventory.items.category[inv_idx]
        item_tid = state.inventory.items.type_id[inv_idx]

        new_g_cat = jnp.where(
            should_drop,
            g.category[0, 0, p_row, p_col].at[free_idx].set(item_cat),
            g.category[0, 0, p_row, p_col],
        )
        new_g_tid = jnp.where(
            should_drop,
            g.type_id[0, 0, p_row, p_col].at[free_idx].set(item_tid),
            g.type_id[0, 0, p_row, p_col],
        )
        g = g.replace(
            category=g.category.at[0, 0, p_row, p_col].set(new_g_cat),
            type_id=g.type_id.at[0, 0, p_row, p_col].set(new_g_tid),
        )

        cleared = jnp.where(should_drop, jnp.int8(-1), new_worn[slot_i])
        new_worn = new_worn.at[slot_i].set(cleared)

        rng_c, sub = jax.random.split(rng_c)
        roll = jax.random.randint(sub, (), 1, 7).astype(jnp.int32)
        dmg_acc = dmg_acc + jnp.where(should_drop, roll, jnp.int32(0))

        return new_worn, g, dmg_acc, rng_c

    init_carry = (worn, ground, jnp.int32(0), rng)
    new_worn, new_ground, total_dmg, _ = jax.lax.fori_loop(
        0, N_ARMOR_SLOTS, _check_slot, init_carry
    )

    new_inv = state.inventory.replace(worn_armor=new_worn)
    state = state.replace(inventory=new_inv, ground_items=new_ground)

    new_hp = jnp.maximum(state.player_hp - total_dmg, jnp.int32(0))
    done = new_hp <= jnp.int32(0)
    return state.replace(player_hp=new_hp, done=state.done | done)


# ---------------------------------------------------------------------------
# newman()  (vendor/nethack/src/polyself.c:336)
# ---------------------------------------------------------------------------

def newman(state, rng: jax.Array):
    """Re-roll player stats when they polymorph into their own race form.

    vendor/nethack/src/polyself.c:336 — newman():
      - Re-roll player XL ± 2 (clamped 1..30).
      - Recompute HP_max from new XL  (8 * XL simplified).
      - Recompute PW_max from new XL  (4 * XL simplified).
      - Cure SICK and STONED status effects.

    Returns
    -------
    EnvState — updated state (does NOT set is_polymorphed; caller handles that).
    """
    from Nethax.nethax.subsystems.status_effects import TimedStatus

    rng, sub = jax.random.split(rng)
    xl_delta  = jax.random.randint(sub, (), -2, 3).astype(jnp.int32)  # [-2,+2]
    new_xl    = jnp.clip(state.player_xl.astype(jnp.int32) + xl_delta,
                         jnp.int32(1), jnp.int32(30))
    new_hp_max = jnp.maximum(new_xl * jnp.int32(8), jnp.int32(1))
    new_pw_max = jnp.maximum(new_xl * jnp.int32(4), jnp.int32(0))
    new_hp     = jnp.minimum(state.player_hp.astype(jnp.int32), new_hp_max)

    # Cure SICK and STONED.
    ts = state.status.timed_statuses
    ts = ts.at[int(TimedStatus.SICK)].set(jnp.int32(0))
    ts = ts.at[int(TimedStatus.STONED)].set(jnp.int32(0))
    new_status = state.status.replace(timed_statuses=ts)

    # polyself.c:336 — newman also restores nutrition to NORMAL (1000).
    new_status = new_status.replace(nutrition=jnp.int32(1000))

    return state.replace(
        player_xl=new_xl,
        player_hp_max=new_hp_max,
        player_pw_max=new_pw_max,
        player_hp=new_hp,
        status=new_status,
    )


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

    # --- 4. Recompute AC.  Apply HP_max swap.  Clamp Pw to current pw_max.
    # polyself.c — HP and Pw are both clamped on poly.
    state = state.replace(
        polymorph=poly,
        player_hp_max=new_hp_max,
        player_hp=jnp.minimum(state.player_hp, new_hp_max),
        player_pw=jnp.minimum(state.player_pw, state.player_pw_max),
    )
    state = _recompute_ac(state, form_i16)

    # --- 4b. Mount-on-poly: if riding and form cannot ride, force dismount.
    # polyself.c:1412 — when can_ride() fails after poly, dismount_steed().
    # We always dismount on poly for safety.  Apply 1d6 fall damage.
    rng, sub_fall = jax.random.split(rng)
    fall_roll = jax.random.randint(sub_fall, (), 1, 7).astype(jnp.int32)
    was_riding = state.player_steed_mid != jnp.uint32(0)

    def _dismount(s):
        new_hp = jnp.maximum(s.player_hp - fall_roll, jnp.int32(0))
        return s.replace(
            player_steed_mid=jnp.uint32(0),
            player_hp=new_hp,
            done=s.done | (new_hp <= jnp.int32(0)),
        )

    state = jax.lax.cond(was_riding, _dismount, lambda s: s, state)

    # --- 5. Drop incompatible armor per-slot (polyself.c:1156 break_armor).
    state = _drop_worn_armor_per_slot(state, form_i16)

    # --- 5b. retouch_equipment: silver items burn silver-allergic forms.
    # polyself.c::retouch_equipment — vampires/weres/demons drop silver gear
    # and take 1d6 burn damage per item.
    rng, sub_rt = jax.random.split(rng)
    state = _retouch_equipment_silver(state, form_i16, sub_rt)

    # TODO: polyself.c — if player has cursed-item-touch-while-polymorphed
    # conflict during prayer, alignment_record -= 2.  Not yet wired.

    # --- 5c. newman(): if target form matches player's own race, re-roll XL/HP/PW
    # and cure sick/stoned.  (polyself.c:336)
    # We approximate "same race" as M2_HUMAN flag in the form matching the
    # player_race == human (race=0).  For simplicity: if flags2 & M2_HUMAN and
    # player_race == 0 (Human), call newman.
    form_flags2 = _FORM_FLAGS2[form_i16.astype(jnp.int32)]
    form_is_human_race = (form_flags2 & jnp.uint32(0x00000008)) != jnp.uint32(0)  # M2_HUMAN=0x8
    player_is_human    = state.player_race.astype(jnp.int32) == jnp.int32(0)
    same_race          = form_is_human_race & player_is_human

    rng, sub_nm = jax.random.split(rng)
    state = jax.lax.cond(same_race,
                         lambda s: newman(s, sub_nm),
                         lambda s: s,
                         state)

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

    Mirrors polyself.c::rehumanize (polyself.c:1367):
      - Unchanging check: if UNCHANGING intrinsic is set, player dies (done=True,
        hp=0).  Cite: polyself.c:1367.
      - Restore STR/DEX/CON/HP_max/AC.
      - Restore the original attack set.
      - Clear is_polymorphed, poly_timer, current_form_idx.
      - If post-revert HP < 1, player dies.  Cite: polyself.c.
    """
    poly = state.polymorph

    def _do_revert(s):
        p = s.polymorph

        # Unchanging: rehumanizing while Unchanging kills the player.
        # polyself.c:1367 — "rehumanize: Unchanging → You die."
        has_unchanging = s.status.intrinsics[UNCHANGING_MASK].astype(jnp.bool_)

        def _unchanging_death(st):
            return st.replace(
                player_hp=jnp.int32(0),
                done=jnp.bool_(True),
            )

        def _normal_revert(st):
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
            reverted = st.replace(
                polymorph=p2,
                player_str=p.orig_str,
                player_dex=p.orig_dex,
                player_con=p.orig_con,
                player_hp_max=p.orig_hp_max,
                player_hp=jnp.minimum(st.player_hp, p.orig_hp_max),
                player_ac=p.orig_ac,
                player_role=p.orig_role_idx,
            )
            # Genocide-self check: polyself.c::rehumanize — if the player's own
            # race/species has been genocided, reverting to that form kills them.
            # polyself.c:233 ugenocided() check inside rehumanize.
            race_idx = reverted.player_race.astype(jnp.int32)
            n_genocided = reverted.genocided_species.shape[0]
            safe_race = jnp.clip(race_idx, 0, n_genocided - 1)
            self_genocided = reverted.genocided_species[safe_race]

            def _genocide_death(st2):
                return st2.replace(player_hp=jnp.int32(0), done=jnp.bool_(True))

            reverted = jax.lax.cond(self_genocided, _genocide_death, lambda st2: st2, reverted)

            # Post-revert: if HP < 1, player dies.  polyself.c rehumanize.
            hp_fatal = reverted.player_hp < jnp.int32(1)
            return jax.lax.cond(
                hp_fatal,
                lambda st2: st2.replace(player_hp=jnp.int32(0), done=jnp.bool_(True)),
                lambda st2: st2,
                reverted,
            )

        return jax.lax.cond(has_unchanging, _unchanging_death, _normal_revert, s)

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
