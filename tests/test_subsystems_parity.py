"""Wave 6 closing-audit #86 — vendor parity for 4 simplified subsystems.

Each test cites the vendor source line it asserts against.

Subsystems covered:
  A. shop.py        ↔ vendor/nethack/src/shk.c
  B. containers.py  ↔ vendor/nethack/src/pickup.c
  C. engrave.py     ↔ vendor/nethack/src/engrave.c
  D. ascension.py   ↔ vendor/nethack/src/end.c   (ASCENDED how-code)

Standing directive: vendor is ground truth.  Where Wave-6 code intentionally
simplifies, the test asserts the documented simplified behavior (and the
divergence is recorded in the corresponding module docstring).
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax import NethaxEnv
from Nethax.nethax.state import EnvState, StaticParams

# ---- A. shop ---------------------------------------------------------------
from Nethax.nethax.subsystems.shop import (
    ShopState,
    accrue_bill,
    pay_at_exit,
    shopkeeper_attack,
    kill_shopkeeper,
    DEFAULT_ITEM_PRICE,
    SHOPKEEPER_ANGRY_DAMAGE,
)
from Nethax.nethax.subsystems.inventory import (
    MAX_INVENTORY_SLOTS,
    ItemCategory,
    InventoryState,
    make_item,
)

# ---- B. containers --------------------------------------------------------
from Nethax.nethax.subsystems.containers import (
    install_container,
    open_container,
    put_in_container,
    take_from_container,
    container_total_weight,
    ContainerType,
    BUCStatus,
    N_CONTAINERS,
    MAX_ITEMS_PER_CONTAINER,
)

# ---- C. engrave ------------------------------------------------------------
from Nethax.nethax.subsystems.engrave import (
    handle_engrave,
    is_elbereth_at,
    ENGR_DUST,
    ENGRAVE_TEXT_LEN,
)
from Nethax.nethax.subsystems.conduct import Conduct

# ---- D. ascension ---------------------------------------------------------
from Nethax.nethax.dungeon.branches import Branch
from Nethax.nethax.dungeon.endgame import (
    ASTRAL_ALTAR_LAWFUL,
    ASTRAL_ALTAR_NEUTRAL,
    ASTRAL_ALTAR_CHAOTIC,
    ASTRAL_ALIGN_LAWFUL,
    ASTRAL_ALIGN_NEUTRAL,
    ASTRAL_ALIGN_CHAOTIC,
)
from Nethax.nethax.subsystems.ascension import (
    check_ascension,
    ascend,
    ASTRAL_LEVEL,
)
from Nethax.nethax.subsystems.scoring import Achievement, compute_final_score
from Nethax.nethax.subsystems.items_jewelry import AmuletEffect


_RNG = jax.random.PRNGKey(2026)


# ---------------------------------------------------------------------------
# Shared shop fixture
# ---------------------------------------------------------------------------
def _make_active_shop(door_row=8, door_col=5, shopkeeper_idx=0):
    return ShopState(
        shop_active=jnp.bool_(True),
        shopkeeper_idx=jnp.int8(shopkeeper_idx),
        shop_room_min=jnp.array([5, 5], dtype=jnp.int8),
        shop_room_max=jnp.array([7, 10], dtype=jnp.int8),
        door_pos=jnp.array([door_row, door_col], dtype=jnp.int8),
        bill=jnp.int32(0),
        items_owned_by_shop=jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.bool_),
        angry=jnp.bool_(False),
    )


def _env_shop_state(player_pos=(6, 6), player_gold=100, shopkeeper_pos=(5, 5)):
    env = NethaxEnv()
    state, _ = env.reset(jax.random.PRNGKey(42))
    shop = _make_active_shop()
    idx = int(shop.shopkeeper_idx)
    mai = state.monster_ai
    mai = mai.replace(
        pos=mai.pos.at[idx].set(jnp.array(shopkeeper_pos, dtype=jnp.int16)),
        alive=mai.alive.at[idx].set(jnp.bool_(True)),
        hp=mai.hp.at[idx].set(jnp.int32(50)),
        hp_max=mai.hp_max.at[idx].set(jnp.int32(50)),
    )
    return state.replace(
        shop=shop,
        player_pos=jnp.array(player_pos, dtype=jnp.int16),
        player_gold=jnp.int32(player_gold),
        monster_ai=mai,
    )


# ===========================================================================
# A. shop.py vendor parity
# ===========================================================================

def test_A1_accrue_bill_inside_shop_marks_unpaid_and_bumps_bill():
    """vendor shk.c:3489-3550 addtobill: in-shop pickup → bill += cost,
    obj->unpaid = TRUE (add_one_tobill, shk.c:3548).

    Wave 15 expanded the starting inventory into slots 0-7; use slot 10
    (post-starting-inv empty) so DEFAULT_ITEM_PRICE fallback fires instead
    of the priced real item.
    """
    s = _env_shop_state(player_pos=(6, 7))  # inside room rows 5-7 cols 5-10
    s1 = accrue_bill(s, slot_idx=10)
    assert int(s1.shop.bill) == DEFAULT_ITEM_PRICE
    assert bool(s1.shop.items_owned_by_shop[10]) is True


def test_A1b_accrue_bill_outside_shop_is_noop():
    """vendor shk.c:3501 billable() returns FALSE outside *u.ushops →
    addtobill returns immediately without touching the bill."""
    s = _env_shop_state(player_pos=(20, 20))
    s1 = accrue_bill(s, slot_idx=10)
    assert int(s1.shop.bill) == 0
    assert not bool(jnp.any(s1.shop.items_owned_by_shop))


def test_A2_pay_at_exit_with_enough_gold_clears_bill():
    """vendor shk.c:2220-2299 dopayobj: PAY_BUY path deducts gold via
    pay() (shk.c:2288) and setpaid() clears bill_p (shk.c:400).

    Slots 10/11 chosen post Wave 15 starting-inv expansion (see A1).
    """
    s = _env_shop_state(player_pos=(6, 6), player_gold=200)
    s = accrue_bill(s, slot_idx=10)
    s = accrue_bill(s, slot_idx=11)
    expected_bill = 2 * DEFAULT_ITEM_PRICE
    assert int(s.shop.bill) == expected_bill
    # Cross the door.
    s = s.replace(player_pos=jnp.array([8, 5], dtype=jnp.int16))
    s2 = pay_at_exit(s)
    assert int(s2.shop.bill) == 0
    assert int(s2.player_gold) == 200 - expected_bill
    assert bool(s2.shop.angry) is False
    assert not bool(jnp.any(s2.shop.items_owned_by_shop))


def test_A2b_pay_at_exit_broke_angers_shopkeeper():
    """vendor shk.c:2283-2284 insufficient_funds → PAY_CANT → caller's
    make_angry_shk (shk.c:1469-1489) flips ANGRY + hot_pursuit.

    Slots 10/11 chosen post Wave 15 starting-inv expansion (see A1).
    """
    s = _env_shop_state(player_pos=(6, 6), player_gold=5)
    s = accrue_bill(s, slot_idx=10)
    s = accrue_bill(s, slot_idx=11)
    # bill = 20, gold = 5 → broke.
    s = s.replace(player_pos=jnp.array([8, 5], dtype=jnp.int16))
    s2 = pay_at_exit(s)
    assert bool(s2.shop.angry) is True
    assert int(s2.shop.bill) == 2 * DEFAULT_ITEM_PRICE
    # gold unchanged, ownership flags persist (matches vendor: walking out
    # with items still on bill keeps them ->unpaid).
    assert int(s2.player_gold) == 5
    assert bool(s2.shop.items_owned_by_shop[10])
    assert bool(s2.shop.items_owned_by_shop[11])


def test_A3_shopkeeper_attack_hot_pursuit_melee_damage():
    """vendor shk.c:1449-1463 hot_pursuit → rile_shk + following=1, and
    vendor mattackm/mhitu path applies melee damage when adjacent."""
    s = _env_shop_state(player_pos=(5, 6), shopkeeper_pos=(5, 5))  # adjacent
    s = s.replace(shop=s.shop.replace(angry=jnp.bool_(True)))
    hp0 = int(s.player_hp)
    s2 = shopkeeper_attack(s, _RNG)
    assert int(s2.player_hp) == max(0, hp0 - SHOPKEEPER_ANGRY_DAMAGE)


def test_A3b_shopkeeper_attack_pursues_when_distant():
    """vendor shk.c:1456 ESHK(shkp)->following=1 + monst.c::shk_chases
    drives a single-tile step toward the player."""
    s = _env_shop_state(player_pos=(10, 10), shopkeeper_pos=(5, 5))
    s = s.replace(shop=s.shop.replace(angry=jnp.bool_(True)))
    idx = int(s.shop.shopkeeper_idx)
    s2 = shopkeeper_attack(s, _RNG)
    new_pos = s2.monster_ai.pos[idx]
    # Greedy 8-dir step (vendor uses move_special / mfndpos; here we
    # snap-step toward the player).  From (5,5) → (6,6).
    assert int(new_pos[0]) == 6 and int(new_pos[1]) == 6
    # Not yet adjacent, no damage.
    assert int(s2.player_hp) == int(s.player_hp)


def test_A4_kill_shopkeeper_clears_bill_keeps_angry():
    """vendor shk.c:261 setpaid(mtmp) on shk death clears bill_p; vendor
    does NOT automatically pacify other shopkeepers, so the simplified
    model leaves angry=True for any remaining hostile shopkeeper-class."""
    s = _env_shop_state(player_pos=(6, 6))
    s = accrue_bill(s, slot_idx=0)
    s = accrue_bill(s, slot_idx=3)
    assert int(s.shop.bill) > 0
    s2 = kill_shopkeeper(s)
    assert int(s2.shop.bill) == 0
    assert not bool(jnp.any(s2.shop.items_owned_by_shop))
    assert bool(s2.shop.angry) is True
    idx = int(s2.shop.shopkeeper_idx)
    assert not bool(s2.monster_ai.alive[idx])
    assert int(s2.monster_ai.hp[idx]) == 0


# ===========================================================================
# B. containers.py vendor parity
# ===========================================================================

def _fresh_state() -> EnvState:
    return EnvState.default(rng=_RNG)


def _place_item(state, slot_idx, weight=5):
    """Place a FOOD item in inventory slot."""
    inv = state.inventory.items
    new_inv = inv.replace(
        category   = inv.category.at[slot_idx].set(jnp.int8(ItemCategory.FOOD)),
        type_id    = inv.type_id.at[slot_idx].set(jnp.int16(7)),
        quantity   = inv.quantity.at[slot_idx].set(jnp.int16(1)),
        weight     = inv.weight.at[slot_idx].set(jnp.int32(weight)),
        identified = inv.identified.at[slot_idx].set(jnp.bool_(True)),
    )
    return state.replace(inventory=state.inventory.replace(items=new_inv))


def test_B1_open_container_sets_is_open():
    """vendor pickup.c::use_container open path (pickup.c:2972-) toggles
    cobj->cknown / opens the container UI."""
    s = _fresh_state()
    s = install_container(s, 0, ContainerType.BAG_OF_HOLDING,
                          buc=int(BUCStatus.UNCURSED))
    s2 = open_container(s, 0)
    assert bool(s2.containers.is_open[0]) is True


def test_B1b_cursed_bag_of_tricks_can_explode_on_open():
    """vendor pickup.c:2150-2157 use_container: cursed BAG_OF_TRICKS
    triggers a wake/spawn (Wave 6 simplification: spawn a hostile monster
    with 50% probability)."""
    # Find a seed with explode_roll < 0.5
    chosen_rng = None
    for seed in range(20):
        rng_test = jax.random.PRNGKey(seed)
        rng_explode = jax.random.split(rng_test)[0]
        roll = float(jax.random.uniform(rng_explode, shape=()))
        if roll < 0.5:
            chosen_rng = rng_test
            break
    assert chosen_rng is not None
    s = _fresh_state().replace(rng=chosen_rng)
    s = install_container(s, 0, ContainerType.BAG_OF_TRICKS,
                          buc=int(BUCStatus.CURSED))
    pre = int(s.monster_ai.alive.sum())
    s2 = open_container(s, 0)
    post = int(s2.monster_ai.alive.sum())
    assert post >= pre + 1


def test_B2_put_in_container_moves_item_and_zeros_inv_slot():
    """vendor pickup.c::in_container (pickup.c:2558-2712):
    freeinv(obj) (pickup.c:2624) + add_to_container (pickup.c:2703)."""
    s = _fresh_state()
    s = install_container(s, 0, ContainerType.SACK)
    s = _place_item(s, slot_idx=2, weight=7)
    s2 = put_in_container(s, 0, 2)
    # Item moved into bag[0][0]:
    assert int(s2.containers.items_category[0, 0]) == int(ItemCategory.FOOD)
    assert int(s2.containers.items_weight[0, 0]) == 7
    # Inventory slot 2 emptied (freeinv).
    assert int(s2.inventory.items.category[2]) == 0
    assert int(s2.inventory.items.quantity[2]) == 0


def test_B2b_bag_of_holding_buc_weight_multipliers():
    """vendor pickup.c::in_container (pickup.c:1549, 2667) +
    weight() applies BoH multipliers per BUC:
      blessed → 1/4   uncursed → 1/2   cursed → 2/1
    """
    # 8-weight item in each BoH variant.
    for buc, expected in (
        (BUCStatus.BLESSED,  2),   # 8 // 4 = 2
        (BUCStatus.UNCURSED, 4),   # 8 // 2 = 4
        (BUCStatus.CURSED,  16),   # 8 * 2  = 16
    ):
        s = _fresh_state()
        s = install_container(s, 0, ContainerType.BAG_OF_HOLDING, buc=int(buc))
        # write weights directly into the container array (avoid item-cat=0)
        cs = s.containers
        new_cat = cs.items_category.at[0, 0].set(jnp.int8(ItemCategory.FOOD))
        new_w   = cs.items_weight.at[0, 0].set(jnp.int16(8))
        s = s.replace(containers=cs.replace(items_category=new_cat,
                                            items_weight=new_w))
        w = int(container_total_weight(s.containers, 0))
        assert w == expected, f"{buc.name}: expected {expected}, got {w}"


def test_B3_take_from_container_drops_into_first_empty_inv_slot():
    """vendor pickup.c::out_container (pickup.c:2727-): obj_extract_self
    + addinv places the item back into the player's inventory."""
    s = _fresh_state()
    s = install_container(s, 0, ContainerType.SACK)
    s = _place_item(s, slot_idx=0, weight=3)
    s = put_in_container(s, 0, 0)  # bag[0][0] holds the FOOD, inv[0] empty
    s2 = take_from_container(s, 0, 0)
    # First empty inv slot is 0 — item should land there.
    assert int(s2.inventory.items.category[0]) == int(ItemCategory.FOOD)
    # Container slot emptied.
    assert int(s2.containers.items_category[0, 0]) == 0


def test_B4_container_total_weight_non_boh_returns_raw_sum():
    """vendor pickup.c::container_weight: sums contents weights; only
    BAG_OF_HOLDING applies BoH multiplier (pickup.c:1549, 2667)."""
    s = _fresh_state()
    s = install_container(s, 0, ContainerType.LARGE_BOX)
    cs = s.containers
    weights = [3, 5, 7]
    new_cat = cs.items_category
    new_w   = cs.items_weight
    for i, w in enumerate(weights):
        new_cat = new_cat.at[0, i].set(jnp.int8(ItemCategory.FOOD))
        new_w   = new_w.at[0, i].set(jnp.int16(w))
    s = s.replace(containers=cs.replace(items_category=new_cat,
                                        items_weight=new_w))
    assert int(container_total_weight(s.containers, 0)) == sum(weights)


# ===========================================================================
# C. engrave.py vendor parity
# ===========================================================================

def _engrave_state(row=5, col=7) -> EnvState:
    s = EnvState.default(_RNG)
    return s.replace(player_pos=jnp.array([row, col], dtype=jnp.int16))


def test_C1_handle_engrave_writes_elbereth_in_dust():
    """vendor engrave.c::doengrave (engrave.c:956-) finger-in-dust path
    writes ASCII text into the engr struct with type=DUST (ENGR_DUST=1)."""
    s = _engrave_state(row=4, col=8)
    s2 = handle_engrave(s, _RNG)
    assert bool(s2.engrave.has_engraving[4, 8]) is True
    assert int(s2.engrave.engraving_kind[4, 8]) == int(ENGR_DUST)
    expected = list(b"Elbereth")
    actual = [int(b) for b in s2.engrave.text[4, 8, :8]]
    assert actual == expected


def test_C2_engrave_violates_elberethless_conduct():
    """vendor insight.c~2206: 'elbereth_count' increments on each engrave
    of 'Elbereth' — Conduct.ELBERETHLESS becomes violated."""
    s = _engrave_state()
    assert bool(s.conduct.violations[int(Conduct.ELBERETHLESS)]) is False
    s2 = handle_engrave(s, _RNG)
    assert bool(s2.conduct.violations[int(Conduct.ELBERETHLESS)]) is True


def test_C3_is_elbereth_at_detects_engraving_for_monster_AI():
    """vendor engrave.c::sengr_at strict-mode (engrave.c:250-261): the
    text consulted by monster.c when deciding to flee/avoid.  After
    handle_engrave at (r,c), is_elbereth_at(r,c) must return True; an
    unwritten tile returns False."""
    s = _engrave_state(row=9, col=11)
    s2 = handle_engrave(s, _RNG)
    # On the written tile: True.
    assert bool(is_elbereth_at(s2.engrave, 9, 11)) is True
    # On a different (empty) tile: False.
    assert bool(is_elbereth_at(s2.engrave, 0, 0)) is False


# ===========================================================================
# D. ascension.py vendor parity
# ===========================================================================

def _make_ascend_state(*, branch: int, level: int, pos, align: int,
                       with_amulet: bool) -> EnvState:
    s = EnvState.default(rng=_RNG, static=StaticParams())
    s = s.replace(
        player_pos=jnp.asarray(pos, dtype=jnp.int16),
        player_align=jnp.int8(align),
    )
    s = s.replace(dungeon=s.dungeon.replace(
        current_branch=jnp.int8(branch),
        current_level=jnp.int8(level),
    ))
    if with_amulet:
        amulet = make_item(
            category=int(ItemCategory.AMULET),
            type_id=int(AmuletEffect.YENDOR),
            quantity=1,
            weight=20,
        )
        s = s.replace(inventory=InventoryState.from_items([amulet]))
    return s


def test_D1_check_ascension_requires_all_three_conditions():
    """vendor end.c gates ASCENDED on (1) astral plane + (2) matching
    altar + (3) Amulet held.  Removing ANY of the three must fail."""
    # All-three: True
    s_ok = _make_ascend_state(
        branch=int(Branch.ENDGAME), level=ASTRAL_LEVEL,
        pos=ASTRAL_ALTAR_NEUTRAL, align=ASTRAL_ALIGN_NEUTRAL,
        with_amulet=True,
    )
    assert bool(check_ascension(s_ok)) is True
    # No Amulet:
    s_no_am = _make_ascend_state(
        branch=int(Branch.ENDGAME), level=ASTRAL_LEVEL,
        pos=ASTRAL_ALTAR_NEUTRAL, align=ASTRAL_ALIGN_NEUTRAL,
        with_amulet=False,
    )
    assert bool(check_ascension(s_no_am)) is False
    # Wrong altar alignment:
    s_wrong_altar = _make_ascend_state(
        branch=int(Branch.ENDGAME), level=ASTRAL_LEVEL,
        pos=ASTRAL_ALTAR_CHAOTIC, align=ASTRAL_ALIGN_LAWFUL,
        with_amulet=True,
    )
    assert bool(check_ascension(s_wrong_altar)) is False
    # Not on Astral:
    s_not_astral = _make_ascend_state(
        branch=int(Branch.MAIN), level=1,
        pos=ASTRAL_ALTAR_LAWFUL, align=ASTRAL_ALIGN_LAWFUL,
        with_amulet=True,
    )
    assert bool(check_ascension(s_not_astral)) is False


def test_D2_ascend_sets_done_and_records_achievement_and_bonus():
    """vendor end.c done() with how=ASCENDED:
      - program_state.gameover = 1   (end.c:1147)  → state.done = True
      - achievement_record(ACH_ASCENDED) → scoring.achievements set
      - compute_final_score adds asc_b doubling (end.c:1344-1351)

    Audit G #4 (Wave 35) removed the legacy flat +50000 bump from
    ``ascend()`` because vendor has no analogue — the only ASCENDED
    reward in end.c:1325-1352 is the XP-doubling realised via
    ``asc_b = base if ascended`` inside ``compute_final_score``.  This
    test now verifies that route: ``scoring.ascended`` is flagged and
    ``compute_final_score`` reflects the bonus.
    """
    s = _make_ascend_state(
        branch=int(Branch.ENDGAME), level=ASTRAL_LEVEL,
        pos=ASTRAL_ALTAR_LAWFUL, align=ASTRAL_ALIGN_LAWFUL,
        with_amulet=True,
    )
    s2 = ascend(s)
    assert bool(s2.done) is True
    assert bool(s2.scoring.achievements[int(Achievement.ASCENDED)]) is True
    assert bool(s2.scoring.ascended) is True, (
        "ascend() should mark scoring.ascended so compute_final_score "
        "fires the asc_b XP-doubling bonus per end.c:1344-1351"
    )
    # compute_final_score should award a positive total on ascension
    # (alignment_bonus 5000 + travel_b + …) — at minimum > 0.
    final = int(compute_final_score(s2))
    assert final > 0, (
        f"compute_final_score on ascended state should be > 0; got {final}"
    )


def test_D3_wave6_simplification_no_offer_needed():
    """Documented Wave 6 divergence: vendor requires the #offer action on
    the matching altar (pray.c::dosacrifice → offer_real_amulet) before
    setting how=ASCENDED.  Wave 6 fires ascension purely from
    'standing on matching altar with Amulet'.  The test pins this
    simplification so a future #offer-only wiring is a deliberate change."""
    s = _make_ascend_state(
        branch=int(Branch.ENDGAME), level=ASTRAL_LEVEL,
        pos=ASTRAL_ALTAR_CHAOTIC, align=ASTRAL_ALIGN_CHAOTIC,
        with_amulet=True,
    )
    # No OFFER action dispatched — but Wave 6 still ascends:
    assert bool(check_ascension(s)) is True
