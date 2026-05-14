"""Wave 6 Closing-Audit — bit-equal starting inventory BUC/enchant parity.

Vendor source: ``vendor/nethack/src/u_init.c`` defines a per-role ``trobj``
table whose fields are ``{ otyp, spe, oclass, qty_min, qty_max, trbless }``.
``ini_inv_adjust_obj`` (u_init.c:1208-1244) then applies these via:

    obj->cursed = 0;
    if (trop->trspe != UNDEF_SPE)
        obj->spe = trop->trspe;
    if (trop->trbless != UNDEF_BLESS)        // UNDEF_BLESS == 2
        obj->blessed = trop->trbless;

So the canonical interpretation is:
  - trbless == 1            → BLESSED (obj->blessed = 1)
  - trbless == 0            → UNCURSED (default, cursed=0, blessed=0)
  - trbless == UNDEF_BLESS  → UNCURSED (cursed=0, blessed unchanged default 0)
  - spe = trspe whenever trspe != UNDEF_SPE (else random; Nethax pins to 0)

This test file asserts each role's starting inventory matches vendor exactly
for BUC status (Item.buc_status) and enchantment (Item.enchantment).

# vendor/nethack/src/u_init.c:1208-1244 (ini_inv_adjust_obj)
# vendor/nethack/src/u_init.c:42-178 (per-role trobj arrays)
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax

from Nethax.nethax.constants.roles import Role
from Nethax.nethax.constants.races import Race
from Nethax.nethax.env import NethaxEnv
from Nethax.nethax.subsystems.character import (
    ObjType,
    STARTING_INVENTORY,
)
from Nethax.nethax.subsystems.inventory import ItemCategory


# BUC codes — same as character.py / inventory.py "0=unknown / 1=cursed
# / 2=uncursed / 3=blessed".  Note: BLESSED is 3.
_BUC_UNCURSED = 2
_BUC_BLESSED = 3


# ---------------------------------------------------------------------------
# Direct table assertions (no env reset needed — tests STARTING_INVENTORY).
# ---------------------------------------------------------------------------

def _find_item(kit, type_id):
    """Return the first Item in ``kit`` with the given type_id."""
    for item in kit:
        if int(item.type_id) == int(type_id):
            return item
    raise AssertionError(f"type_id={type_id} not found in kit")


# Valkyrie -----------------------------------------------------------------

def test_valkyrie_long_sword_uncursed_plus1():
    """Vendor u_init.c Valkyrie[0]: SPEAR spe=1, trbless=UNDEF_BLESS → UNCURSED.

    The audit doc names the primary as "long sword"; vendor truth is SPEAR.

    # vendor/nethack/src/u_init.c:161
    """
    kit = STARTING_INVENTORY[Role.VALKYRIE]
    primary = _find_item(kit, ObjType.SPEAR)
    assert int(primary.enchantment) == 1
    assert int(primary.buc_status) == _BUC_UNCURSED


def test_valkyrie_small_shield_uncursed_plus3():
    """Vendor u_init.c Valkyrie[2]: SMALL_SHIELD spe=3 trbless=UNDEF_BLESS.

    Audit doc says +0; vendor truth is +3.

    # vendor/nethack/src/u_init.c:163
    """
    kit = STARTING_INVENTORY[Role.VALKYRIE]
    shield = _find_item(kit, ObjType.SMALL_SHIELD)
    assert int(shield.enchantment) == 3
    assert int(shield.buc_status) == _BUC_UNCURSED


# Wizard -------------------------------------------------------------------

def test_wizard_quarterstaff_blessed_plus1():
    """Vendor u_init.c Wizard[0]: QUARTERSTAFF spe=1 trbless=1 → BLESSED.

    # vendor/nethack/src/u_init.c:168
    """
    kit = STARTING_INVENTORY[Role.WIZARD]
    qs = _find_item(kit, ObjType.QUARTERSTAFF)
    assert int(qs.enchantment) == 1
    assert int(qs.buc_status) == _BUC_BLESSED


def test_wizard_cloak_of_mr_uncursed_plus0():
    """Vendor u_init.c Wizard[1]: CLOAK_OF_MAGIC_RESISTANCE spe=0
    trbless=UNDEF_BLESS → UNCURSED +0.

    # vendor/nethack/src/u_init.c:169
    """
    kit = STARTING_INVENTORY[Role.WIZARD]
    cloak = _find_item(kit, ObjType.CLOAK_OF_MAGIC_RESISTANCE)
    assert int(cloak.enchantment) == 0
    assert int(cloak.buc_status) == _BUC_UNCURSED


# Knight -------------------------------------------------------------------

def test_knight_blessed_kit():
    """Vendor u_init.c Knight[]:
        LONG_SWORD spe=1 trbless=UNDEF_BLESS  → UNCURSED +1
        LANCE      spe=1 trbless=UNDEF_BLESS  → UNCURSED +1
        RING_MAIL  spe=1 trbless=UNDEF_BLESS  → UNCURSED +1
        HELMET     spe=0 trbless=UNDEF_BLESS  → UNCURSED +0
        SMALL_SHIELD spe=0 trbless=UNDEF_BLESS → UNCURSED +0
        LEATHER_GLOVES spe=0 trbless=UNDEF_BLESS → UNCURSED +0

    Audit doc says Knight gets BLESSED items; vendor truth is UNCURSED for
    every trobj entry where trbless==UNDEF_BLESS (because ini_inv_adjust_obj
    only sets blessed when trbless != UNDEF_BLESS, and the default state
    after ``obj->cursed=0`` plus zero-init blessed flag is UNCURSED).

    # vendor/nethack/src/u_init.c:90-99
    """
    kit = STARTING_INVENTORY[Role.KNIGHT]

    ls = _find_item(kit, ObjType.LONG_SWORD)
    assert int(ls.enchantment) == 1
    assert int(ls.buc_status) == _BUC_UNCURSED

    lance = _find_item(kit, ObjType.LANCE)
    assert int(lance.enchantment) == 1
    assert int(lance.buc_status) == _BUC_UNCURSED

    rm = _find_item(kit, ObjType.RING_MAIL)
    assert int(rm.enchantment) == 1
    assert int(rm.buc_status) == _BUC_UNCURSED

    helm = _find_item(kit, ObjType.HELMET)
    assert int(helm.enchantment) == 0
    assert int(helm.buc_status) == _BUC_UNCURSED

    shield = _find_item(kit, ObjType.SMALL_SHIELD)
    assert int(shield.enchantment) == 0
    assert int(shield.buc_status) == _BUC_UNCURSED

    gloves = _find_item(kit, ObjType.LEATHER_GLOVES)
    assert int(gloves.enchantment) == 0
    assert int(gloves.buc_status) == _BUC_UNCURSED


# Healer -------------------------------------------------------------------

def test_healer_4_potions_of_healing():
    """Vendor u_init.c Healer[3]: POT_HEALING qty=4 trbless=UNDEF_BLESS.

    All four potions live in a single stacked Item slot in Nethax;
    quantity=4 reproduces the vendor row exactly.

    # vendor/nethack/src/u_init.c:80
    """
    kit = STARTING_INVENTORY[Role.HEALER]
    pot = _find_item(kit, ObjType.POT_HEALING)
    assert int(pot.quantity) == 4
    assert int(pot.buc_status) == _BUC_UNCURSED


def test_healer_scalpel_uncursed_plus0():
    """Vendor u_init.c Healer[0]: SCALPEL spe=0 trbless=UNDEF_BLESS.

    # vendor/nethack/src/u_init.c:77
    """
    kit = STARTING_INVENTORY[Role.HEALER]
    scalpel = _find_item(kit, ObjType.SCALPEL)
    assert int(scalpel.enchantment) == 0
    assert int(scalpel.buc_status) == _BUC_UNCURSED


# Samurai ------------------------------------------------------------------

def test_samurai_katana_wakizashi_yumi_25_arrows():
    """Vendor u_init.c Samurai[]:
        KATANA      spe=0 trbless=UNDEF_BLESS → UNCURSED +0
        SHORT_SWORD (wakizashi) spe=0 trbless=UNDEF_BLESS → UNCURSED +0
        YUMI        spe=0 trbless=UNDEF_BLESS → UNCURSED +0
        YA          qty=26 spe=0 trbless=UNDEF_BLESS → UNCURSED +0
              (audit doc says 25; vendor's qty_min is 26, vendor truth==26)

    # vendor/nethack/src/u_init.c:142-148
    """
    kit = STARTING_INVENTORY[Role.SAMURAI]

    katana = _find_item(kit, ObjType.KATANA)
    assert int(katana.enchantment) == 0
    assert int(katana.buc_status) == _BUC_UNCURSED

    wakizashi = _find_item(kit, ObjType.SHORT_SWORD)
    assert int(wakizashi.enchantment) == 0
    assert int(wakizashi.buc_status) == _BUC_UNCURSED

    yumi = _find_item(kit, ObjType.YUMI)
    assert int(yumi.enchantment) == 0
    assert int(yumi.buc_status) == _BUC_UNCURSED

    ya = _find_item(kit, ObjType.YA)
    assert int(ya.enchantment) == 0
    assert int(ya.buc_status) == _BUC_UNCURSED
    assert int(ya.quantity) == 26


# Monk ---------------------------------------------------------------------

def test_monk_leather_gloves_plus2_robe_plus1():
    """Vendor u_init.c Monk[]:
        LEATHER_GLOVES spe=2 trbless=UNDEF_BLESS → UNCURSED +2
        ROBE           spe=1 trbless=UNDEF_BLESS → UNCURSED +1

    Audit doc says "MON leather armor UNCURSED +0"; vendor truth is the
    pair above.

    # vendor/nethack/src/u_init.c:102-103
    """
    kit = STARTING_INVENTORY[Role.MONK]

    gloves = _find_item(kit, ObjType.LEATHER_GLOVES)
    assert int(gloves.enchantment) == 2
    assert int(gloves.buc_status) == _BUC_UNCURSED

    robe = _find_item(kit, ObjType.ROBE)
    assert int(robe.enchantment) == 1
    assert int(robe.buc_status) == _BUC_UNCURSED


# Priest -------------------------------------------------------------------

def test_priest_mace_blessed_plus1():
    """Vendor u_init.c Priest[0]: MACE spe=1 trbless=1 → BLESSED +1.

    # vendor/nethack/src/u_init.c:115
    """
    kit = STARTING_INVENTORY[Role.PRIEST]
    mace = _find_item(kit, ObjType.MACE)
    assert int(mace.enchantment) == 1
    assert int(mace.buc_status) == _BUC_BLESSED


def test_priest_holy_water_blessed():
    """Vendor u_init.c Priest[3]: POT_WATER qty=4 trbless=1 → BLESSED (holy).

    # vendor/nethack/src/u_init.c:118
    """
    kit = STARTING_INVENTORY[Role.PRIEST]
    pot = _find_item(kit, ObjType.POT_WATER)
    assert int(pot.quantity) == 4
    assert int(pot.buc_status) == _BUC_BLESSED


# Caveman ------------------------------------------------------------------

def test_caveman_sling_plus2_club_plus1():
    """Vendor u_init.c Caveman[]: CLUB spe=1, SLING spe=2.

    # vendor/nethack/src/u_init.c:69-70
    """
    kit = STARTING_INVENTORY[Role.CAVEMAN]
    club = _find_item(kit, ObjType.CLUB)
    assert int(club.enchantment) == 1
    sling = _find_item(kit, ObjType.SLING)
    assert int(sling.enchantment) == 2


# ---------------------------------------------------------------------------
# End-to-end: env reset preserves BUC + enchant from STARTING_INVENTORY.
# ---------------------------------------------------------------------------

def _reset(role: Role, seed: int = 4242):
    rng = jax.random.PRNGKey(seed)
    env = NethaxEnv()
    state, _ = env.reset(rng, role=role, race=Race.HUMAN, alignment=0)
    return state


def test_env_reset_valkyrie_preserves_spear_plus1():
    """After NethaxEnv.reset, the Valkyrie's wielded spear retains spe=+1
    and UNCURSED BUC (the vendor trobj values flow through unmodified).

    # vendor/nethack/src/u_init.c:161
    """
    state = _reset(Role.VALKYRIE)
    items = state.inventory.items
    # Find the spear in inventory.
    n = items.category.shape[0]
    spear_slot = -1
    for i in range(n):
        if int(items.category[i]) == int(ItemCategory.WEAPON) and \
           int(items.type_id[i]) == int(ObjType.SPEAR):
            spear_slot = i
            break
    assert spear_slot >= 0, "Valkyrie should have a spear after reset"
    assert int(items.enchantment[spear_slot]) == 1
    assert int(items.buc_status[spear_slot]) == _BUC_UNCURSED


def test_env_reset_wizard_preserves_blessed_quarterstaff():
    """After reset, Wizard's quarterstaff retains spe=+1 and BLESSED BUC.

    # vendor/nethack/src/u_init.c:168
    """
    state = _reset(Role.WIZARD)
    items = state.inventory.items
    n = items.category.shape[0]
    qs_slot = -1
    for i in range(n):
        if int(items.category[i]) == int(ItemCategory.WEAPON) and \
           int(items.type_id[i]) == int(ObjType.QUARTERSTAFF):
            qs_slot = i
            break
    assert qs_slot >= 0, "Wizard should have a quarterstaff after reset"
    assert int(items.enchantment[qs_slot]) == 1
    assert int(items.buc_status[qs_slot]) == _BUC_BLESSED
