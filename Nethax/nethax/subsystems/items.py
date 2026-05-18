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

import jax.numpy as jnp
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


