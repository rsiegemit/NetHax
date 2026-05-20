"""Item property subsystem — BUC status, erosion, enchantment, per-item flags.

Canonical sources:
  vendor/nethack/src/mkobj.c   — item creation, initial BUC/enchant assignment
  vendor/nethack/src/objnam.c  — object naming, appearance-to-type glue
  vendor/nethack/include/obj.h — oeroded/oeroded2 bitfields, oerodeproof

Status: Wave 1 stub — enums, ItemEffects state slice, no-op functions.

TODO (Wave 3 — core item arithmetic):
  - apply_blessing / apply_curse: update buc_status, trigger prayer feedback
  - erode: decrement item quality, enforce oerodeproof flag,
           map erosion kind to oeroded vs oeroded2 bitfield (obj.h lines 124-132)
  - enchant: clamp to [-7, +7] for armor, [-5, +5] for weapons (mkobj.c limits)
  - Effect dispatch per category:
      POTION  → potion.c effect table
      SCROLL  → read.c effect table
      WAND    → apply.c / zap.c effect table
      RING    → worn.c setworn/setnotworn
      AMULET  → worn.c setworn/setnotworn
      SPELLBOOK → read.c spell-learn path
      TOOL    → apply.c tool-use dispatch

TODO (Wave 4 — full item lifecycle):
  - BUC sensing: altar/priest identification (insight.c)
  - Charges decrement on wand/tool use; recharge mechanic
  - Erosion ticks per turn for items in corrosive environments
  - Grease / erodeproof: prevent erosion when oerodeproof flag set
  - Named user items (oname / oextra->oname in obj.h)

TODO (Wave 5 — containers):
  - Items inside bags of holding have weight modified by container enchant
"""
from enum import IntEnum

import jax
import jax.numpy as jnp
import jax.lax as lax
from flax import struct


# ---------------------------------------------------------------------------
# BUC status
# ---------------------------------------------------------------------------

class BUCStatus(IntEnum):
    """Blessed/Uncursed/Cursed status.

    JAX-friendly storage note: the canonical NetHack values (-1/0/1/2) use a
    negative sentinel which is awkward in uint arrays.  We remap to 0-3 so
    the value fits cleanly in int8 / uint8 arrays.  Use buc_to_jax() and
    jax_to_buc() for conversion when comparing against NetHack source logic.

      JAX value  NetHack meaning
      ---------  ---------------
      0          UNKNOWN  (not yet sensed; stored as -1 in NetHack)
      1          CURSED   (0 in NetHack)
      2          UNCURSED (1 in NetHack)
      3          BLESSED  (2 in NetHack)
    """
    UNKNOWN  = 0
    CURSED   = 1
    UNCURSED = 2
    BLESSED  = 3


def buc_to_jax(nethack_value: int) -> int:
    """Convert NetHack BUC value (-1/0/1/2) to JAX-friendly (0/1/2/3)."""
    return nethack_value + 1


def jax_to_buc(jax_value: int) -> int:
    """Convert JAX BUC value (0/1/2/3) back to NetHack convention (-1/0/1/2)."""
    return jax_value - 1


# ---------------------------------------------------------------------------
# Erosion
# ---------------------------------------------------------------------------

class Erosion(IntEnum):
    """Erosion level for weapons and armor.

    vendor/nethack/include/obj.h:
      oeroded  (2-bit): rust (iron) or burn (leather/cloth)
      oeroded2 (2-bit): corrode (copper) or rot (organic)

    We flatten both axes into a single enum for JAX storage.
    The 2-bit fields each hold 0 (none), 1, 2, or 3 (maximum).

      NONE          — pristine
      RUSTY_*       — iron/ferrous items; from oeroded field
      BURNT_*       — leather/cloth items; from oeroded field
      CORRODED_*    — copper items; from oeroded2 field
      ROTTED_*      — organic items; from oeroded2 field

    Wave 3 will split this back into the two oeroded / oeroded2 fields when
    computing actual AC penalties (see do_wear.c:ARM_BONUS).
    """
    NONE        = 0
    RUSTY1      = 1
    RUSTY2      = 2
    RUSTY3      = 3
    BURNT1      = 4
    BURNT2      = 5
    BURNT3      = 6
    CORRODED1   = 7
    CORRODED2   = 8
    CORRODED3   = 9
    ROTTED1     = 10
    ROTTED2     = 11
    ROTTED3     = 12


# ---------------------------------------------------------------------------
# Per-item flags
# ---------------------------------------------------------------------------

@struct.dataclass
class ItemEffects:
    """Per-item boolean flags derived from obj struct bitfields.

    Stored separately from the main Item to keep Item compact for JIT tracing.
    Wave 3 will merge these into the hot Item path or pass alongside it.

    Fields
    ------
    blessed       : BUC is BLESSED (redundant with buc_status; kept for fast mask)
    cursed        : BUC is CURSED
    eroded        : oeroded level (0-3); iron/leather erosion (obj.h line 124)
    eroded2       : oeroded2 level (0-3); copper/organic erosion (obj.h line 125)
    erodeproof    : immune to erosion (oerodeproof flag, obj.h line 133)
    charged       : item has charges remaining (wands, tools)
    identified    : type-level identification has been applied
    named_user    : player has given item a personal name (oextra->oname)
    """
    blessed:    jnp.ndarray   # scalar bool
    cursed:     jnp.ndarray   # scalar bool
    eroded:     jnp.ndarray   # scalar int8, range [0, 3]
    eroded2:    jnp.ndarray   # scalar int8, range [0, 3]
    erodeproof: jnp.ndarray   # scalar bool
    charged:    jnp.ndarray   # scalar bool
    identified: jnp.ndarray   # scalar bool
    named_user: jnp.ndarray   # scalar bool

    @classmethod
    def default(cls) -> "ItemEffects":
        """Return zeroed ItemEffects for a freshly created item."""
        return cls(
            blessed=jnp.bool_(False),
            cursed=jnp.bool_(False),
            eroded=jnp.int8(0),
            eroded2=jnp.int8(0),
            erodeproof=jnp.bool_(False),
            charged=jnp.bool_(False),
            identified=jnp.bool_(False),
            named_user=jnp.bool_(False),
        )


# ---------------------------------------------------------------------------
# erode_obj — central erosion function (vendor/nethack/src/trap.c lines 171-354)
# ---------------------------------------------------------------------------

# ERODE_* type constants (vendor/nethack/include/obj.h lines 455-459).
ERODE_BURN:    int = 0
ERODE_RUST:    int = 1
ERODE_ROT:     int = 2
ERODE_CORRODE: int = 3
ERODE_CRACK:   int = 4  # crystal armor

# EF_* flag bits (vendor/nethack/include/obj.h lines 462-466).
EF_NONE:    int = 0
EF_GREASE:  int = 0x1  # check for greased object
EF_DESTROY: int = 0x2  # potentially destroy
EF_VERBOSE: int = 0x4  # print extra messages
EF_PAY:     int = 0x8  # player's fault

# ER_* return codes (vendor/nethack/include/obj.h lines 469-472).
ER_NOTHING:   int = 0
ER_GREASED:   int = 1
ER_DAMAGED:   int = 2
ER_DESTROYED: int = 3

# Maximum oeroded/oeroded2 level (vendor/nethack/include/obj.h:129).
MAX_ERODE: int = 3

# Material-class flags emitted by Nethax.nethax.obs.inv_strs._erosion_mat_class:
#   0 = none (no erosion)
#   1 = rustprone (IRON/METAL/MITHRIL/PLATINUM)
#   2 = flammable/rottable (LEATHER/WOOD/CLOTH/PAPER/WAX/VEGGY/FLESH)
#   3 = corrodeable-only (COPPER/SILVER/GOLD)
# Cite: vendor/nethack/include/objclass.h lines 199-212.
_MATCLASS_NONE:        int = 0
_MATCLASS_RUSTPRONE:   int = 1
_MATCLASS_FLAMMABLE:   int = 2
_MATCLASS_CORRODEONLY: int = 3


def _is_primary(kind: jnp.ndarray) -> jnp.ndarray:
    """True iff erosion writes to oeroded (primary) rather than oeroded2.

    Cite: vendor/nethack/src/trap.c::erode_obj lines 218 (ROT is_primary=FALSE),
          225 (CORRODE is_primary=FALSE), 230 (CRACK is_primary=TRUE);
          BURN/RUST default is_primary=TRUE.
    """
    k = kind.astype(jnp.int32)
    return (k == jnp.int32(ERODE_BURN)) | (k == jnp.int32(ERODE_RUST)) \
         | (k == jnp.int32(ERODE_CRACK))


def _vulnerable_for_kind(mat_class: jnp.ndarray, kind: jnp.ndarray) -> jnp.ndarray:
    """Material gate per erosion kind.

    Mirrors vendor/nethack/src/trap.c::erode_obj lines 204-236:
      ERODE_BURN    -> is_flammable     (mat_class 2)
      ERODE_RUST    -> is_rustprone     (mat_class 1)
      ERODE_ROT     -> is_rottable      (mat_class 2)
      ERODE_CORRODE -> is_corrodeable   (mat_class 1 or 3 — COPPER/IRON in vendor;
                                         we accept rustprone IRON too via mat_class 1)
      ERODE_CRACK   -> is_crackable     (glass armor; mat_class table treats GLASS as
                                         class 0 / never crackable for non-armor —
                                         we conservatively return False here)
    """
    m = mat_class.astype(jnp.int32)
    k = kind.astype(jnp.int32)
    is_burn    = k == jnp.int32(ERODE_BURN)
    is_rust    = k == jnp.int32(ERODE_RUST)
    is_rot     = k == jnp.int32(ERODE_ROT)
    is_corrode = k == jnp.int32(ERODE_CORRODE)
    burn_ok    = is_burn    & (m == jnp.int32(_MATCLASS_FLAMMABLE))
    rust_ok    = is_rust    & (m == jnp.int32(_MATCLASS_RUSTPRONE))
    rot_ok     = is_rot     & (m == jnp.int32(_MATCLASS_FLAMMABLE))
    corrode_ok = is_corrode & ((m == jnp.int32(_MATCLASS_RUSTPRONE))
                               | (m == jnp.int32(_MATCLASS_CORRODEONLY)))
    return burn_ok | rust_ok | rot_ok | corrode_ok


def erode_obj_slot(
    inventory_items,
    slot_idx,
    kind,
    force,
    rng=None,
):
    """JIT-pure erosion of inventory_items[slot_idx].

    Parameters
    ----------
    inventory_items : Item (batched, shape [N])
    slot_idx        : int32 scalar — slot to erode (must be in range; caller clips)
    kind            : int32 scalar — ERODE_BURN / ERODE_RUST / ERODE_ROT / ERODE_CORRODE
    force           : bool scalar — when True, bypass blessed-resist check
                      (vendor: ``ef_flags & EF_PAY`` style force; we treat ``force``
                      as overriding the ``otmp->blessed && !rnl(4)`` chance gate)
    rng             : optional JAX PRNG key. When provided, the blessed-resist
                      gate samples the vendor ``!rnl(4)`` (1/4 chance to resist)
                      using this key; when omitted, falls back to the deterministic
                      worst-case (blessed always blocks unless ``force`` is True).

    Returns
    -------
    new_items : Item   — inventory_items with oeroded/oeroded2 incremented if applicable.
    result    : int32  — ER_NOTHING / ER_GREASED / ER_DAMAGED.

    Erosion follows vendor/nethack/src/trap.c::erode_obj lines 171-354:
      1. greased + (kind∈{RUST,CORRODE}) → ER_GREASED, no damage.
      2. ``erosion_matters`` is approximated via mat_class != 0 (we don't gate
         on oclass since the caller has already chosen a worn-armor or wielded
         weapon slot).
      3. vulnerable check per kind (see ``_vulnerable_for_kind``).
      4. oerodeproof → ER_NOTHING.
      5. current erosion < MAX_ERODE → bump appropriate field, ER_DAMAGED.
      6. erosion saturated → ER_NOTHING (this function never destroys; the
         EF_DESTROY branch is left to higher-level callers since it requires
         item-removal bookkeeping outside this slot-level helper).
    """
    from Nethax.nethax.obs.inv_strs import _OBJECT_EROSION_CLASS

    sidx = jnp.asarray(slot_idx, dtype=jnp.int32)
    k    = jnp.asarray(kind, dtype=jnp.int32)
    f    = jnp.asarray(force, dtype=jnp.bool_)

    # Pull per-slot fields (read with single gather; .at[].set later).
    n_slots = inventory_items.oeroded.shape[0]
    safe = jnp.clip(sidx, 0, n_slots - 1)

    type_id   = inventory_items.type_id[safe].astype(jnp.int32)
    safe_type = jnp.clip(type_id, 0, _OBJECT_EROSION_CLASS.shape[0] - 1)
    matclass  = _OBJECT_EROSION_CLASS[safe_type].astype(jnp.int32)

    greased     = inventory_items.greased[safe]
    erodeproof  = inventory_items.oerodeproof[safe]
    oeroded     = inventory_items.oeroded[safe].astype(jnp.int32)
    oeroded2    = inventory_items.oeroded2[safe].astype(jnp.int32)
    buc         = inventory_items.buc_status[safe].astype(jnp.int32)
    # BUC: 3 == BLESSED in JAX remap (see BUCStatus enum above).
    blessed     = buc == jnp.int32(int(BUCStatus.BLESSED))

    primary     = _is_primary(k)
    vulnerable  = _vulnerable_for_kind(matclass, k)
    erosion_matters = matclass != jnp.int32(_MATCLASS_NONE)

    # Grease only protects against RUST and CORRODE (vendor trap.c lines 208,
    # 217: BURN clears check_grease; ROT sets check_grease=FALSE).
    grease_applies = greased & ((k == jnp.int32(ERODE_RUST))
                                | (k == jnp.int32(ERODE_CORRODE)))

    # Blessed has 1/4 chance to resist (vendor trap.c:257 ``otmp->blessed && !rnl(4)``).
    # When an ``rng`` is provided, sample the vendor 1/4 gate; otherwise fall
    # back to the deterministic worst case (blessed always blocks unless forced).
    if rng is not None:
        from Nethax.nethax.rng import rnl as _rnl
        # !rnl(4) is true iff the roll equals 0 (luck=0 default).
        blessed_resists = _rnl(rng, 4) == jnp.int32(0)
    else:
        blessed_resists = jnp.bool_(True)
    blessed_blocks = blessed & (~f) & blessed_resists

    cur_erosion = jnp.where(primary, oeroded, oeroded2)
    saturated   = cur_erosion >= jnp.int32(MAX_ERODE)

    # Outcome triage — order matches vendor:
    #   greased  -> ER_GREASED
    #   !erosion_matters -> ER_NOTHING
    #   !vulnerable | (erodeproof & rknown) -> ER_NOTHING
    #   erodeproof | (blessed && !rnl(4))    -> ER_NOTHING
    #   erosion < MAX_ERODE                  -> ER_DAMAGED
    is_greased   = grease_applies
    is_blocked   = (~erosion_matters) | (~vulnerable) | erodeproof | blessed_blocks
    can_damage   = (~is_greased) & (~is_blocked) & (~saturated)

    new_primary_val   = jnp.where(can_damage & primary,  oeroded  + jnp.int32(1), oeroded)
    new_secondary_val = jnp.where(can_damage & (~primary), oeroded2 + jnp.int32(1), oeroded2)

    new_oeroded_arr  = inventory_items.oeroded .at[safe].set(new_primary_val.astype(inventory_items.oeroded.dtype))
    new_oeroded2_arr = inventory_items.oeroded2.at[safe].set(new_secondary_val.astype(inventory_items.oeroded2.dtype))

    new_items = inventory_items.replace(
        oeroded=new_oeroded_arr,
        oeroded2=new_oeroded2_arr,
    )

    result = jnp.where(
        is_greased,
        jnp.int32(ER_GREASED),
        jnp.where(can_damage, jnp.int32(ER_DAMAGED), jnp.int32(ER_NOTHING)),
    )
    return new_items, result


def erode_obj(state, slot_idx, kind, force=False, rng=None):
    """High-level wrapper: erode ``state.inventory.items[slot_idx]`` by ``kind``.

    Cite: vendor/nethack/src/trap.c::erode_obj lines 171-354.
    Returns (new_state, result).  result is one of ER_NOTHING / ER_GREASED /
    ER_DAMAGED (ER_DESTROYED is not emitted here — caller handles destruction).

    When ``rng`` is provided, the blessed-resist gate uses the vendor
    ``otmp->blessed && !rnl(4)`` 1/4 roll (trap.c:257); otherwise the
    deterministic worst case is used (blessed always blocks unless ``force``).
    """
    new_items, result = erode_obj_slot(state.inventory.items, slot_idx, kind, force, rng=rng)
    new_inv = state.inventory.replace(items=new_items)
    return state.replace(inventory=new_inv), result


