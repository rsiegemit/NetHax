"""Verify Item dataclass exposes all vendor obj.h gameplay bitfields.

Wave 6 polish: Item carries the gameplay-critical bitfields from
``vendor/nethack/include/obj.h::struct obj``.  The fields we omit
(pointers, save-only flags, UI scratchpads) are listed below for
intent-of-design auditability — they are deliberately not in Item.

Citations (vendor/nethack/include/obj.h line numbers):
  greased     line 142  — grease coating
  oeroded     line 124  — primary erosion tier (rust/burn)
  oeroded2    line 125  — secondary erosion tier (corrode/rot)
  oerodeproof line 133  — erosion-proof
  bknown      line 113  — BUC awareness
  lamplit     line 104  — light source lit
  olocked     line 134  — container locked
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp

from Nethax.nethax.subsystems.inventory import Item, make_empty_item, make_item


REQUIRED_VENDOR_FIELDS = (
    "greased",
    "oeroded",
    "oeroded2",
    "oerodeproof",
    "bknown",
    "lamplit",
    "olocked",
    "dknown",
    "rknown",
)

# Deliberately omitted (cosmetic / save-only / UI scratchpad / pointer state).
# Listed here so future audits can see this is an intentional decision.
DELIBERATELY_OMITTED = (
    "nobj", "cobj", "o_id", "where", "ox", "oy",     # pointers / world links
    "invlet",                                          # UI inventory letter
    "named_how", "ghostly", "how_lost", "pickup_prev",# save / history
    "unpaid", "no_charge",                             # shop bookkeeping
    "recharged", "in_use", "bypass",                   # transient flags
    "cknown", "lknown", "tknown",                      # secondary id flags
    # NB: dknown/rknown are now first-class Item fields (see REQUIRED_VENDOR_FIELDS).
    "globby", "obroken", "otrapped", "nomerge",        # rare-state flags
    "timed",                                           # timer queue link
)


def test_item_exposes_vendor_bitfields():
    """Every gameplay-critical vendor bitfield must be a field on Item."""
    item = make_empty_item()
    for fname in REQUIRED_VENDOR_FIELDS:
        assert hasattr(item, fname), f"Item missing vendor field: {fname}"


def test_item_default_values_match_vendor_init():
    """Vendor obj_init zeros the bitfields → Item defaults must be False/0."""
    item = make_empty_item()
    assert bool(item.greased) is False
    assert int(item.oeroded) == 0
    assert int(item.oeroded2) == 0
    assert bool(item.oerodeproof) is False
    assert bool(item.bknown) is False
    assert bool(item.lamplit) is False
    assert bool(item.olocked) is False
    assert bool(item.dknown) is False
    assert bool(item.rknown) is False


def test_item_erosion_tier_range():
    """oeroded / oeroded2 are 2-bit bitfields (0..3) in vendor obj.h."""
    item = make_empty_item().replace(
        oeroded=jnp.int8(3),
        oeroded2=jnp.int8(3),
    )
    assert int(item.oeroded) == 3
    assert int(item.oeroded2) == 3


def test_make_item_accepts_new_field_overrides():
    """Item.replace must accept the new fields just like any other field."""
    base = make_item(category=1, type_id=10, quantity=1, weight=30)
    greased = base.replace(greased=jnp.bool_(True))
    assert bool(greased.greased) is True
    assert int(greased.category) == 1  # other fields untouched


def test_item_field_count_full():
    """Lock in the full Item schema so accidental drops trigger a failure."""
    fields = set(Item.__dataclass_fields__.keys())
    expected = {
        # Core (pre-existing)
        "category", "type_id", "buc_status", "enchantment", "charges",
        "identified", "quantity", "weight", "ac_bonus", "is_two_handed",
        # Wave 6 vendor obj.h additions
        "greased", "oeroded", "oeroded2", "oerodeproof", "bknown",
        "lamplit", "olocked",
        # post-erosion-merge: corpse identity tracking field
        "corpse_entry_idx",
        # Wand recharge counter (vendor obj.h line 102 Bitfield(recharged,3)).
        "recharged",
        # Corpse age + poisoned-tin tracking (vendor eat.c::eatcorpse 1885,
        # consume_tin 1537).
        "corpse_creation_turn", "tin_poisoned",
        # Per-item description-known / rustproofing-known (vendor obj.h
        # lines 109-114).
        "dknown", "rknown",
    }
    assert fields == expected, (
        f"Item field-set drift:\n"
        f"  unexpected: {fields - expected}\n"
        f"  missing   : {expected - fields}"
    )
