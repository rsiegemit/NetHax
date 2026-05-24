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


# ---------------------------------------------------------------------------
# Vendor-byte-equal monster-form lookup tables for polymon HP/STR/intrinsics
# Cite: vendor/nethack/src/polyself.c:735 (polymon HP formula),
#       vendor/nethack/src/makemon.c:2233 (golemhp table),
#       vendor/nethack/src/mondata.c:80   (poly_when_stoned),
#       vendor/nethack/src/polyself.c:75-110 (PROPSET adoption set).
# ---------------------------------------------------------------------------

# Vendor PM_ constants are recovered by name lookup at build time so they
# stay correct even when the local MONSTERS table reorders.  See
# _resolve_pm_indices below.
def _resolve_pm_indices() -> dict[str, int]:
    """Map vendor PM_ names → local MONSTERS index by `m.name` lookup.

    Vendor names (from include/monsters.h) and our MONSTERS[].name use the
    same spelling, so a 1:1 lookup works.  Returns -1 for any name we
    cannot find — defensive only; missing entries would indicate a real
    table divergence to investigate.
    """
    from Nethax.nethax.constants.monsters import MONSTERS
    wanted = (
        "gray dragon", "yellow dragon", "silver dragon",
        "straw golem", "paper golem", "rope golem", "gold golem",
        "leather golem", "wood golem", "flesh golem", "clay golem",
        "stone golem", "glass golem", "iron golem",
    )
    out: dict[str, int] = {n: -1 for n in wanted}
    for i, m in enumerate(MONSTERS):
        if m.name in out and out[m.name] == -1:
            out[m.name] = i
    return out


_PM = _resolve_pm_indices()

# Vendor PM_GRAY_DRAGON..PM_YELLOW_DRAGON spans the 9 adult chromatic
# dragons.  In the local MONSTERS table they are contiguous, so we use
# the resolved [gray, yellow] range to flag adult dragons.
PM_GRAY_DRAGON: int   = _PM["gray dragon"]
PM_YELLOW_DRAGON: int = _PM["yellow dragon"]
PM_STONE_GOLEM: int   = _PM["stone golem"]

# golemhp(type) lookup — vendor values from makemon.c:2233.  Keyed by
# local MONSTERS index resolved via name (PM_ enum values themselves
# vary across vendor versions; the HP values are the canonical data).
_GOLEM_HP_BY_PM: dict[int, int] = {
    _PM["straw golem"]:   20,
    _PM["paper golem"]:   20,
    _PM["rope golem"]:    30,
    _PM["gold golem"]:    60,
    _PM["leather golem"]: 40,
    _PM["wood golem"]:    50,
    _PM["flesh golem"]:   40,
    _PM["clay golem"]:    70,
    _PM["stone golem"]:  100,
    _PM["glass golem"]:   80,
    _PM["iron golem"]:   120,
}
# Drop any sentinel -1 entries (missing names) so they cannot poison the table.
_GOLEM_HP_BY_PM = {k: v for k, v in _GOLEM_HP_BY_PM.items() if k >= 0}


def _build_form_strongmonst() -> jnp.ndarray:
    """bool[N_MONSTERS]: True iff form has M2_STRONG flag.

    vendor/nethack/src/polyself.c:821 — strongmonst(&mons[mntmp]) test;
    if true, ABASE(STR)=AMAX(STR)=newMaxStr (=STR18/100, or 25 for giants).
    """
    from Nethax.nethax.constants.monsters import MONSTERS, M2_STRONG
    return jnp.array(
        [bool(m.flags2 & M2_STRONG) for m in MONSTERS], dtype=jnp.bool_,
    )


_FORM_STRONGMONST: jnp.ndarray = _build_form_strongmonst()


def _build_form_is_giant() -> jnp.ndarray:
    """bool[N_MONSTERS]: True iff form has M2_GIANT flag.

    vendor/nethack/include/mondata.h:107 — #define is_giant(ptr)
        (((ptr)->mflags2 & M2_GIANT) != 0L).
    Cite: vendor/nethack/src/polyself.c:1103 uasmon_maxStr — is_giant(ptr).
    Includes Lord Surtur, Cyclops, giant zombie, giant mummy in addition
    to the six S_GIANT monsters.
    """
    from Nethax.nethax.constants.monsters import MONSTERS, M2_GIANT
    return jnp.array(
        [bool(m.flags2 & M2_GIANT) for m in MONSTERS], dtype=jnp.bool_,
    )


_FORM_IS_GIANT: jnp.ndarray = _build_form_is_giant()


def _build_form_is_undead() -> jnp.ndarray:
    """bool[N_MONSTERS]: True iff form has M2_UNDEAD flag.

    vendor/nethack/include/mondata.h:95 — #define is_undead(ptr)
        (((ptr)->mflags2 & M2_UNDEAD) != 0L).
    Cite: vendor/nethack/src/polyself.c:1103 uasmon_maxStr — !is_undead(ptr)
    gates the giant STR19(19) bonus (giant zombies/mummies don't qualify).
    """
    from Nethax.nethax.constants.monsters import MONSTERS, M2_UNDEAD
    return jnp.array(
        [bool(m.flags2 & M2_UNDEAD) for m in MONSTERS], dtype=jnp.bool_,
    )


_FORM_IS_UNDEAD: jnp.ndarray = _build_form_is_undead()


def _build_form_is_golem() -> jnp.ndarray:
    """bool[N_MONSTERS]: True iff form is a golem (mlet == S_GOLEM).

    vendor/nethack/include/mondata.h:108 — #define is_golem(ptr) ((ptr)->mlet == S_GOLEM).
    """
    from Nethax.nethax.constants.monsters import MONSTERS, MonsterSymbol
    return jnp.array(
        [m.symbol == MonsterSymbol.S_GOLEM for m in MONSTERS], dtype=jnp.bool_,
    )


_FORM_IS_GOLEM: jnp.ndarray = _build_form_is_golem()


def _build_form_is_dragon() -> jnp.ndarray:
    """bool[N_MONSTERS]: True iff form is an adult dragon (S_DRAGON,
    PM_GRAY_DRAGON..PM_YELLOW_DRAGON range).  Babies excluded.

    Cite: vendor/nethack/src/polyself.c:860 — mlet == S_DRAGON && mntmp >= PM_GRAY_DRAGON.
    """
    from Nethax.nethax.constants.monsters import MONSTERS, MonsterSymbol
    result = []
    for i, m in enumerate(MONSTERS):
        is_adult = (m.symbol == MonsterSymbol.S_DRAGON
                    and PM_GRAY_DRAGON <= i <= PM_YELLOW_DRAGON)
        result.append(is_adult)
    return jnp.array(result, dtype=jnp.bool_)


_FORM_IS_ADULT_DRAGON: jnp.ndarray = _build_form_is_dragon()


def _build_form_is_home_elemental() -> jnp.ndarray:
    """bool[N_MONSTERS]: True iff form is a home (Plane) elemental.

    vendor/nethack/src/mondata.c::is_home_elemental — Plane-of-X resident
    elementals (fire/water/air/earth) on their respective Planes.  We
    approximate via M2_ELEMENTAL flag if present, else off (Planes-of-X
    are postgame).  Field gated by polymon's branch but currently always
    False since nethax flags do not record M2_ELEMENTAL distinctly.
    """
    from Nethax.nethax.constants.monsters import MONSTERS, MonsterSymbol
    # Vendor: elementals are mlet==S_ELEMENTAL; "home" means the player is
    # on the matching Plane (e.g. fire elemental on Plane of Fire).  Since
    # nethax does not currently model the Planes, we return False everywhere;
    # the *3 multiplier is left dormant until Plane state lands.
    _ = MONSTERS, MonsterSymbol  # silence linters
    return jnp.zeros((len(MONSTERS),), dtype=jnp.bool_)


_FORM_IS_HOME_ELEMENTAL: jnp.ndarray = _build_form_is_home_elemental()


def _build_form_golemhp() -> jnp.ndarray:
    """int32[N_MONSTERS]: golemhp(form_idx) result, 0 for non-golems.

    Cite: vendor/nethack/src/makemon.c:2233 (golemhp switch).
    """
    from Nethax.nethax.constants.monsters import MONSTERS
    arr = [_GOLEM_HP_BY_PM.get(i, 0) for i in range(len(MONSTERS))]
    return jnp.array(arr, dtype=jnp.int32)


_FORM_GOLEM_HP: jnp.ndarray = _build_form_golemhp()


def _build_form_can_ride() -> jnp.ndarray:
    """bool[N_MONSTERS]: True iff a hero polymorphed into this form can ride.

    Vendor cite: vendor/nethack/src/steed.c:169-174 can_ride()
        humanoid(youmonst.data) && !verysmall(youmonst.data)
        && !bigmonst(youmonst.data)
    where (mondata.h:11-12,65)
        verysmall(p) := msize < MZ_SMALL
        bigmonst(p)  := msize >= MZ_LARGE
        humanoid(p)  := (mflags1 & M1_HUMANOID) != 0
    The mtame and Underwater/is_swimmer clauses depend on the steed, not
    the rider form, so they're checked elsewhere; here we capture only
    the new-form gate consulted from polyself.c:963.
    """
    from Nethax.nethax.constants.monsters import (
        MONSTERS, MZ_SMALL, MZ_LARGE, M1_HUMANOID,
    )
    out = []
    for m in MONSTERS:
        humanoid_f = bool(int(m.flags1) & M1_HUMANOID)
        verysmall  = m.size < MZ_SMALL
        bigmon     = m.size >= MZ_LARGE
        out.append(humanoid_f and (not verysmall) and (not bigmon))
    return jnp.array(out, dtype=jnp.bool_)


_FORM_CAN_RIDE: jnp.ndarray = _build_form_can_ride()


# ---------------------------------------------------------------------------
# Vendor PROPSET adoption table (polyself.c:88-109)
#
# For each form, pre-compute the set of intrinsics that polymorphing into
# that form should grant (PROPSET sets the FROMOUTSIDE bit on the property).
# We store one bool[N_MONSTERS] table per property; status.intrinsics gets
# updated accordingly inside polymorph_player.
# ---------------------------------------------------------------------------

def _build_form_props():
    """Return a dict of bool[N_MONSTERS] arrays for each PROPSET property.

    Properties cited (polyself.c:88-109):
      TELEPORT       — can_teleport(mdat): M1_TPORT
      LEVITATION     — is_floater(mdat):   M1_FLOAT
      FLYING         — is_flyer(mdat) && !is_floater
      SWIMMING       — is_swimmer(mdat):   M1_SWIM
      PASSES_WALLS   — passes_walls(mdat): M1_WALLWALK
      REGENERATION   — regenerates(mdat):  M1_REGEN
      REFLECTING     — PM_SILVER_DRAGON
      BLINDED        — !haseyes(mdat)
      BLND_RES       — dmgtype(mdat, AD_BLND, AT_EXPL/AT_GAZE)
      MAGIC_BREATHING — M1_AMPHIBIOUS or M1_BREATHLESS
    """
    from Nethax.nethax.constants.monsters import (
        MONSTERS,
        M1_TPORT, M1_FLY, M1_SWIM, M1_AMPHIBIOUS, M1_BREATHLESS,
        M1_REGEN, MonsterSymbol,
    )
    # Vendor is_floater(ptr) := (ptr->mlet == S_EYE || ptr->mlet == S_LIGHT)
    # (mondata.h:20).  There is no M1_FLOAT bit; floater-ness is symbol-based.
    # M1_WALLWALK / M1_NOEYES bit positions (see constants/monsters.py:354,361)
    M1_WALLWALK = 0x00000008
    M1_NOEYES   = 0x00001000

    n = len(MONSTERS)
    teleport     = []
    levitation   = []
    flying       = []
    swimming     = []
    passes_walls = []
    regeneration = []
    reflecting   = []
    blinded      = []
    blnd_res     = []
    magic_breath = []

    # Silver dragon by name lookup
    silver_dragon_idx = None
    for i, m in enumerate(MONSTERS):
        if m.name == "silver dragon":
            silver_dragon_idx = i
            break

    from Nethax.nethax.constants.monsters import AttackType, DamageType
    for i, m in enumerate(MONSTERS):
        f1 = int(m.flags1)
        is_tport  = bool(f1 & M1_TPORT)
        # is_floater: mlet == S_EYE or S_LIGHT (mondata.h:20)
        is_float  = m.symbol in (MonsterSymbol.S_EYE, MonsterSymbol.S_LIGHT)
        is_fly    = bool(f1 & M1_FLY) and not is_float
        is_swim   = bool(f1 & M1_SWIM)
        is_wallw  = bool(f1 & M1_WALLWALK)
        is_regen  = bool(f1 & M1_REGEN)
        noeyes    = bool(f1 & M1_NOEYES)
        # MAGICAL_BREATHING: amphibious or breathless
        magbr     = bool(f1 & (M1_AMPHIBIOUS | M1_BREATHLESS))
        # AD_BLND attacks via AT_EXPL or AT_GAZE → BLND_RES
        has_blnd_atk = False
        for a in m.attacks:
            if int(a[1]) == int(DamageType.AD_BLND) and int(a[0]) in (
                int(AttackType.AT_EXPL), int(AttackType.AT_GAZE),
            ):
                has_blnd_atk = True
                break
        is_silver = (silver_dragon_idx is not None and i == silver_dragon_idx)

        teleport.append(is_tport)
        levitation.append(is_float)
        flying.append(is_fly)
        swimming.append(is_swim)
        passes_walls.append(is_wallw)
        regeneration.append(is_regen)
        reflecting.append(is_silver)
        blinded.append(noeyes)
        blnd_res.append(has_blnd_atk)
        magic_breath.append(magbr)

    return {
        "TELEPORT":        jnp.array(teleport,     dtype=jnp.bool_),
        "LEVITATION":      jnp.array(levitation,   dtype=jnp.bool_),
        "FLYING":          jnp.array(flying,       dtype=jnp.bool_),
        "SWIMMING":        jnp.array(swimming,     dtype=jnp.bool_),
        "PASSES_WALLS":    jnp.array(passes_walls, dtype=jnp.bool_),
        "REGEN":           jnp.array(regeneration, dtype=jnp.bool_),
        "REFLECTING":      jnp.array(reflecting,   dtype=jnp.bool_),
        "BLINDED":         jnp.array(blinded,      dtype=jnp.bool_),
        "BLND_RES":        jnp.array(blnd_res,     dtype=jnp.bool_),
        "MAGIC_BREATHING": jnp.array(magic_breath, dtype=jnp.bool_),
    }


_FORM_PROPS: dict = _build_form_props()


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
    """Return a fresh HP_max roll for the player polymorph form (byte-equal).

    vendor/nethack/src/polyself.c:860-870 — polymon u.mhmax computation::

        mlvl = mons[mntmp].mlevel;
        if (mlet == S_DRAGON && mntmp >= PM_GRAY_DRAGON) {
            u.mhmax = In_endgame ? (8 * mlvl) : (4 * mlvl + d(mlvl, 4));
        } else if (is_golem(mons[mntmp])) {
            u.mhmax = golemhp(mntmp);
        } else {
            if (!mlvl) u.mhmax = rnd(4);
            else       u.mhmax = d(mlvl, 8);
            if (is_home_elemental(mons[mntmp])) u.mhmax *= 3;
        }

    JIT-pure via lax.switch over the three branches; In_endgame is False
    in nethax (no Planes-of-X), so the dragon branch always uses the
    4*mlvl + d(mlvl,4) form.
    """
    tables = _monster_tables()
    idx = form_idx.astype(jnp.int32)
    mlvl = tables["level"][idx].astype(jnp.int32)

    is_dragon = _FORM_IS_ADULT_DRAGON[idx]
    is_golem  = _FORM_IS_GOLEM[idx]
    is_home_e = _FORM_IS_HOME_ELEMENTAL[idx]
    golem_hp  = _FORM_GOLEM_HP[idx].astype(jnp.int32)

    # Static-shape masked roll: d(N, S) with traced N.
    # Vendor d(N, S) = sum_{i=1..N} (1 + rn2(S)).
    _MAX_DICE = 32  # vendor mlvl ≤ ~20; 32 is safe upper bound for dice count.

    def _dice_sum(rng_in, n: jnp.ndarray, sides: jnp.ndarray) -> jnp.ndarray:
        # n,sides traced int32 scalars. Roll _MAX_DICE dice; mask first n.
        rolls = jax.random.randint(
            rng_in, (_MAX_DICE,), 0, jnp.maximum(sides, jnp.int32(1)),
            dtype=jnp.int32,
        ) + jnp.int32(1)
        active = jnp.arange(_MAX_DICE, dtype=jnp.int32) < n
        return jnp.sum(jnp.where(active, rolls, jnp.int32(0))).astype(jnp.int32)

    rng_dragon, rng_normal, rng_zero = jax.random.split(rng, 3)

    # Dragon branch: 4*mlvl + d(mlvl, 4)
    dragon_hp = jnp.int32(4) * mlvl + _dice_sum(rng_dragon, mlvl, jnp.int32(4))

    # Normal branch: mlvl==0 → rnd(4) = 1..4 ; mlvl>0 → d(mlvl, 8)
    rnd4 = (jax.random.randint(rng_zero, (), 0, jnp.int32(4), dtype=jnp.int32)
            + jnp.int32(1))
    d_mlvl_8 = _dice_sum(rng_normal, mlvl, jnp.int32(8))
    base_hp = jnp.where(mlvl == jnp.int32(0), rnd4, d_mlvl_8)
    base_hp = jnp.where(is_home_e, base_hp * jnp.int32(3), base_hp)

    # Pick branch: dragon → golem → normal (matching vendor priority).
    result = jnp.where(is_golem, golem_hp, base_hp)
    result = jnp.where(is_dragon, dragon_hp, result)
    return jnp.maximum(result, jnp.int32(1)).astype(jnp.int32)


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
      - Recompute HP_max via vendor formula at polyself.c:386-394:
        ``new_hp_max = (hp_max * rn1(4,8) // 10) + sum(newhp() per new_lvl)``
        where ``newhp()`` returns class-dependent HP per level ~ rnd(8)+1.
        We approximate ``sum(newhp())`` as ``new_lvl * 8`` (mean of the
        per-level rolls); the ``rn1(4,8)`` retain-factor and proportional
        current-HP carry are exact.  JAX-required divergence: vendor reads
        ``u.uhpinc[i]`` per-level history which Nethax does not store.
      - Recompute PW_max via the same form at polyself.c:401-410
        (mean newpw() ~ 4 PW/level).
      - Cure SICK and STONED status effects.

    Returns
    -------
    EnvState — updated state (does NOT set is_polymorphed; caller handles that).
    """
    from Nethax.nethax.subsystems.status_effects import TimedStatus

    rng, sub, sub_hp, sub_pw = jax.random.split(rng, 4)
    xl_delta  = jax.random.randint(sub, (), -2, 3).astype(jnp.int32)  # [-2,+2]
    old_xl    = state.player_xl.astype(jnp.int32)
    new_xl    = jnp.clip(old_xl + xl_delta, jnp.int32(1), jnp.int32(30))
    old_hp_max = state.player_hp_max.astype(jnp.int32)
    old_pw_max = state.player_pw_max.astype(jnp.int32)

    # Vendor polyself.c:386-397 formula:
    #   hpmax = u.uhpmax (minus per-level history we don't store)
    #   hpmax = rounddiv(hpmax * rn1(4,8), 10)         # retain 80-110%
    #   for i in newlvl: hpmax += newhp()              # ~rnd(8)+1 ≈ 5.5 mean
    #   hpmax = max(hpmax, ulevel)                     # floor at 1 HP/level
    hp_retain = jax.random.randint(sub_hp, (), 4, 12, dtype=jnp.int32)  # rn1(4,8)=[4,11]
    new_hp_max = (old_hp_max * hp_retain) // jnp.int32(10) + new_xl * jnp.int32(8)
    new_hp_max = jnp.maximum(new_hp_max, new_xl)
    # PW formula identical with newpw() mean ≈ 4.
    pw_retain = jax.random.randint(sub_pw, (), 4, 12, dtype=jnp.int32)
    new_pw_max = (old_pw_max * pw_retain) // jnp.int32(10) + new_xl * jnp.int32(4)
    new_pw_max = jnp.maximum(new_pw_max, new_xl)
    # Vendor polyself.c:396 — current HP retains the same proportion.
    safe_old_max = jnp.maximum(old_hp_max, jnp.int32(1))
    new_hp = (state.player_hp.astype(jnp.int32) * new_hp_max) // safe_old_max
    new_hp = jnp.minimum(new_hp, new_hp_max)

    # Cure SICK and STONED.
    ts = state.status.timed_statuses
    ts = ts.at[int(TimedStatus.SICK)].set(jnp.int32(0))
    ts = ts.at[int(TimedStatus.STONED)].set(jnp.int32(0))
    new_status = state.status.replace(timed_statuses=ts)

    # vendor/nethack/src/polyself.c:414 — u.uhunger = rn1(500, 500);
    # rn1(x, y) := rn2(x) + y, so nutrition resets to [500, 999] inclusive.
    rng, sub_h = jax.random.split(rng)
    new_status = new_status.replace(
        nutrition=jnp.int32(500)
        + jax.random.randint(sub_h, (), 0, 500, dtype=jnp.int32)
    )

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

def polymorph_player(state, rng: jax.Array, target_form_idx, controlled: bool = False):
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

    # poly_timer = rn1(500, 500) → [500, 1000).
    rng, sub2 = jax.random.split(rng)
    timer = (jnp.int16(_POLY_TIMER_BASE)
             + jax.random.randint(sub2, (), 0, _POLY_TIMER_RANGE).astype(jnp.int16))

    # vendor/nethack/src/polyself.c:874 — low-level chars get shorter poly
    # timers when transforming into high-level forms::
    #     if (u.ulevel < mlvl)
    #         u.mtimedone = u.mtimedone * u.ulevel / mlvl;
    # Skip when ulevel >= mlvl (no scaling).
    tables_local = _monster_tables()
    mlvl_form = tables_local["level"][form_i16.astype(jnp.int32)].astype(jnp.int32)
    ulevel = state.player_xl.astype(jnp.int32)
    safe_mlvl = jnp.maximum(mlvl_form, jnp.int32(1))
    scaled_timer = (timer.astype(jnp.int32) * ulevel // safe_mlvl).astype(jnp.int16)
    timer = jnp.where(ulevel < mlvl_form, scaled_timer, timer)

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

    # --- 4a. Adopt uasmon_maxStr (polyself.c:1077-1119 + 820-832).
    # vendor: newMaxStr = uasmon_maxStr();
    #   if (strongmonst(&mons[mntmp])) ABASE(A_STR) = AMAX(A_STR) = newMaxStr;
    #   else                           AMAX(A_STR) = newMaxStr;
    # uasmon_maxStr returns:
    #   strongmonst + is_giant + !is_undead → STR19(19) = 119
    #   strongmonst else                    → STR18(100) = 118
    #   !strongmonst                        → 18
    # (Race-based branch R->attrmax[A_STR] for orc/elf/dwarf/gnome forms is
    #  not applied here per task spec; player_str encoding is int16
    #  3..18 normal, 19..118 = STR18/01..100, 119..125 = STR19/+.)
    form_i32   = form_i16.astype(jnp.int32)
    is_strong  = _FORM_STRONGMONST[form_i32]
    is_giant_f = _FORM_IS_GIANT[form_i32]
    is_undead_f= _FORM_IS_UNDEAD[form_i32]
    live_H     = is_giant_f & (~is_undead_f)
    new_max_str = jnp.where(
        is_strong & live_H,
        jnp.int16(119),                        # STR19(19)
        jnp.where(is_strong, jnp.int16(118),   # STR18(100)
                  jnp.int16(18)),
    )
    cur_str = state.player_str.astype(jnp.int16)
    # strongmonst: set both ABASE and AMAX → player_str = new_max_str.
    # non-strongmonst: AMAX = new_max_str; current strength clamped down.
    updated_str = jnp.where(is_strong,
                            new_max_str,
                            jnp.minimum(cur_str, new_max_str))
    state = state.replace(player_str=updated_str.astype(state.player_str.dtype))

    # --- 4b. Mount-on-poly: if riding and new form cannot ride, force dismount.
    # Vendor cite: polyself.c:955-965 — if (u.usteed) { ... if (!can_ride(...))
    #   dismount_steed(DISMOUNT_POLY); }
    # can_ride() (steed.c:169-174) tests the *new* rider form: humanoid &&
    # !verysmall && !bigmonst.  We capture that via _FORM_CAN_RIDE[form_idx].
    # Vendor's DISMOUNT_POLY path (steed.c::dismount_steed) drops the hero in
    # place (no movement) and applies fall damage via Levitating gate.
    # Cite: steed.c::dismount_steed DISMOUNT_POLY branch — 1d6 fall damage
    # unless Levitating (status_effects.Intrinsic.LEVITATION).
    rng, sub_fall = jax.random.split(rng)
    fall_roll = jax.random.randint(sub_fall, (), 1, 7).astype(jnp.int32)
    was_riding = state.player_steed_mid != jnp.uint32(0)
    new_form_can_ride = _FORM_CAN_RIDE[form_i16.astype(jnp.int32)]
    from Nethax.nethax.subsystems.status_effects import Intrinsic as _Intr
    levitating = state.status.intrinsics[int(_Intr.LEVITATION)]
    do_dismount = was_riding & (~new_form_can_ride)

    def _dismount(s):
        # Vendor: fall damage skipped if Levitating (steed.c::dismount_steed).
        applied = jnp.where(levitating, jnp.int32(0), fall_roll)
        new_hp = jnp.maximum(s.player_hp - applied, jnp.int32(0))
        return s.replace(
            player_steed_mid=jnp.uint32(0),
            player_hp=new_hp,
            done=s.done | (new_hp <= jnp.int32(0)),
        )

    state = jax.lax.cond(do_dismount, _dismount, lambda s: s, state)

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

    # --- 7. Conduct: POLYSELFLESS violated — bump counter + set bit.
    # Vendor: u.uconduct.polyselfs++ (polyself.c::polyself); counter consumed
    # by insight.c::show_conduct line ~2178 ("changed form %ld time%s").
    from Nethax.nethax.subsystems.conduct import Conduct, increment_counter
    state = increment_counter(state, int(Conduct.POLYSELFLESS))

    # Emit "You turn into ..." message.
    # Cite: vendor/nethack/src/polyself.c::polymon — pline("You turn into ...").
    from Nethax.nethax.subsystems.messages import emit as _msg_emit, MessageId as _MsgId
    state = state.replace(messages=_msg_emit(state.messages, int(_MsgId.YOU_TURN_INTO)))

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

        reverted = jax.lax.cond(has_unchanging, _unchanging_death, _normal_revert, s)
        # Emit "You return to your old form." — only when actually reverting.
        # Cite: vendor/nethack/src/polyself.c::rehumanize (line ~1367)
        # pline("You return to %s form!", ...).
        from Nethax.nethax.subsystems.messages import emit as _msg_emit, MessageId as _MsgId
        return reverted.replace(
            messages=_msg_emit(reverted.messages, int(_MsgId.YOU_RETURN_TO_HUMAN)),
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
