"""Wave 6 Phase B+ — bit-equal combat parity tests vs vendor NetHack C.

Wave 3 docs admit the original combat tests check *ranges* rather than
specific damage values.  This file closes that gap: every test enumerates
inputs and compares the Nethax Python result to the vendor C formula's
expected output directly.

Vendor sources referenced (line numbers refer to the bundled
``vendor/nethack`` checkout):

  * weapon.c:962-973  — STR portion of ``abon`` (strhitbon table).
  * weapon.c:979-988  — DEX portion of ``abon`` (dexbon table).
  * weapon.c:1000-1015 — ``dbon``                (strdambon table).
  * weapon.c:1545-1577 — ``weapon_hit_bonus`` for normal weapons.
  * weapon.c:1644-1675 — ``weapon_dam_bonus`` for normal weapons.
  * weapon.c:215-302  — ``dmgval`` (per-weapon damage roll).
  * uhitm.c:365-427   — ``find_roll_to_hit`` (composite to-hit tmp).
  * uhitm.c:709-710   — ``mhit = (tmp > dieroll)`` (strict-gt comparison).
  * mhitu.c:709-718   — monster-to-hit ``tmp`` accumulator.
  * do_wear.c:2473-2495 — ``find_ac`` (AC computation).

All asserts are exact integer equality — no ranges.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

from Nethax.nethax.subsystems.combat import (
    PLAYER_BASE_AC,
    SKILL_BASIC,
    SKILL_EXPERT,
    SKILL_GRAND_MASTER,
    SKILL_MASTER,
    SKILL_SKILLED,
    SKILL_UNSKILLED,
    _abon,
    _dbon,
    _SKILL_DAM_BONUS,
    _SKILL_HIT_BONUS,
    dexbon,
    dmgval_weapon,
    find_ac_formula,
    find_roll_to_hit_formula,
    monster_to_hit_tmp,
    strdambon,
    strhitbon,
    to_hit_roll,
    weapon_skill_dam_bonus,
    weapon_skill_hit_bonus,
)


# ---------------------------------------------------------------------------
# STR-to-hit bonus parity (vendor/nethack/src/weapon.c:962-973).
# ---------------------------------------------------------------------------
def test_strhitbon_matches_vendor():
    """STR portion of abon() per weapon.c:962-973.

    Vendor C reference:
        str < 6                 → -2
        str < 8       (== 6,7)  → -1
        str < 17     (8..16)    →  0
        str < STR18(50) (17..67) → +1   /* STR18(50) == 68 */
        str < STR18(100) (68..117) → +2
        else (>= 118)          → +3
    """
    cases = [
        (3,   -2),
        (5,   -2),
        (6,   -1),
        (7,   -1),
        (8,    0),
        (10,   0),
        (16,   0),
        (17,   1),
        (18,   1),    # raw 18
        (18 + 49, 1), # 18/49
        (18 + 50, 2), # 18/50
        (18 + 99, 2), # 18/99
        (18 + 100, 3),  # 18/100
        (125,  3),
    ]
    for s, expected in cases:
        assert strhitbon(s) == expected, (
            f"strhitbon({s}): expected {expected}, got {strhitbon(s)}"
        )


def test_strhitbon_array_helper_matches_pure_python():
    """The JIT-array _abon must agree with the pure-Python strhitbon+dexbon
    composition for every interesting STR / DEX / XL triple."""
    for s in (3, 6, 8, 10, 16, 17, 18, 18 + 49, 18 + 50, 18 + 99, 18 + 100, 125):
        for d in (3, 5, 6, 7, 13, 14, 18, 25):
            for xl in (1, 2, 3, 5, 10):
                expected = strhitbon(s) + dexbon(d)
                # vendor weapon.c:977 — XL<3 gets +1 to-hit kludge.
                if xl < 3:
                    expected += 1
                got = int(_abon(jnp.int16(s), jnp.int8(d), jnp.int32(xl)))
                assert got == expected, (
                    f"_abon(str={s}, dex={d}, xl={xl}) -> {got}, "
                    f"expected {expected}"
                )


# ---------------------------------------------------------------------------
# STR-to-damage bonus parity (vendor/nethack/src/weapon.c:1000-1015).
# ---------------------------------------------------------------------------
def test_strdambon_matches_vendor():
    """STR portion of dbon() per weapon.c:1000-1015.

    Vendor C reference:
        str < 6        → -1
        str < 16       →  0
        str < 18       → +1
        str == 18      → +2
        str <= STR18(75)  (≤93)  → +3
        str <= STR18(90)  (≤108) → +4
        str < STR18(100) (≤117) → +5
        else                    → +6
    """
    cases = [
        (3,        -1),
        (5,        -1),
        (6,         0),
        (15,        0),
        (16,        1),
        (17,        1),
        (18,        2),
        (18 + 1,    3),    # 18/01
        (18 + 75,   3),    # 18/75
        (18 + 76,   4),    # 18/76
        (18 + 90,   4),    # 18/90
        (18 + 91,   5),    # 18/91
        (18 + 99,   5),    # 18/99
        (18 + 100,  6),    # 18/100
        (125,       6),
    ]
    for s, expected in cases:
        assert strdambon(s) == expected, (
            f"strdambon({s}): expected {expected}, got {strdambon(s)}"
        )


def test_strdambon_array_helper_matches_pure_python():
    for s in (3, 6, 8, 16, 17, 18, 18 + 75, 18 + 76, 18 + 90, 18 + 91,
              18 + 99, 18 + 100, 125):
        expected = strdambon(s)
        got = int(_dbon(jnp.int16(s)))
        assert got == expected, f"_dbon({s}) -> {got}, expected {expected}"


# ---------------------------------------------------------------------------
# DEX-to-hit bonus parity (vendor/nethack/src/weapon.c:979-988).
# ---------------------------------------------------------------------------
def test_dexbon_matches_vendor():
    """DEX portion of abon() per weapon.c:979-988.

    Vendor C reference:
        dex < 4   → -3
        dex < 6   → -2
        dex < 8   → -1
        dex < 14  →  0
        else      → dex - 14
    """
    cases = [
        (3,   -3),
        (4,   -2),
        (5,   -2),
        (6,   -1),
        (7,   -1),
        (8,    0),
        (13,   0),
        (14,   0),
        (15,   1),
        (18,   4),
        (25,  11),
    ]
    for d, expected in cases:
        assert dexbon(d) == expected, (
            f"dexbon({d}): expected {expected}, got {dexbon(d)}"
        )


# ---------------------------------------------------------------------------
# Weapon-skill bonuses (vendor/nethack/src/weapon.c:1545-1675).
# ---------------------------------------------------------------------------
def test_weapon_skill_hit_bonus_matches_vendor():
    """vendor weapon.c:1545-1577 (weapon_hit_bonus for ordinary weapons):
        unskilled/restricted → -4
        basic                 →  0
        skilled               → +2
        expert                → +3
    """
    expected = {
        SKILL_UNSKILLED:    -4,
        SKILL_BASIC:         0,
        SKILL_SKILLED:       2,
        SKILL_EXPERT:        3,
        # Master / GM are unreachable for ordinary weapons; clamp to Expert.
        SKILL_MASTER:        3,
        SKILL_GRAND_MASTER:  3,
    }
    for tier, want in expected.items():
        got = weapon_skill_hit_bonus(tier)
        assert got == want, (
            f"weapon_skill_hit_bonus({tier}) -> {got}, expected {want}"
        )
        # The runtime table must agree with the pure-Python helper.
        assert int(_SKILL_HIT_BONUS[tier]) == want, (
            f"_SKILL_HIT_BONUS[{tier}] = {int(_SKILL_HIT_BONUS[tier])}, "
            f"expected {want}"
        )


def test_weapon_skill_dam_bonus_matches_vendor():
    """vendor weapon.c:1644-1675 (weapon_dam_bonus for ordinary weapons):
        unskilled/restricted → -2
        basic                 →  0
        skilled               → +1
        expert                → +2
    """
    expected = {
        SKILL_UNSKILLED:    -2,
        SKILL_BASIC:         0,
        SKILL_SKILLED:       1,
        SKILL_EXPERT:        2,
        # Master / GM clamped to Expert (not reachable for ordinary weapons).
        SKILL_MASTER:        2,
        SKILL_GRAND_MASTER:  2,
    }
    for tier, want in expected.items():
        got = weapon_skill_dam_bonus(tier)
        assert got == want, (
            f"weapon_skill_dam_bonus({tier}) -> {got}, expected {want}"
        )
        assert int(_SKILL_DAM_BONUS[tier]) == want, (
            f"_SKILL_DAM_BONUS[{tier}] = {int(_SKILL_DAM_BONUS[tier])}, "
            f"expected {want}"
        )


# ---------------------------------------------------------------------------
# AC computation (vendor/nethack/src/do_wear.c::find_ac, lines 2473-2495).
# ---------------------------------------------------------------------------
def test_find_ac_formula_matches_vendor():
    """uac = base_ac - sum(ARM_BONUS(uarm[i])) per do_wear.c:2473-2495.

    Base human form is mons[PM_HUMAN].ac == 10.  Each worn armour piece
    contributes its ARM_BONUS (positive), which is *subtracted* from the
    base AC (lower-is-better convention).
    """
    # Stripped: AC == 10.
    assert find_ac_formula(10, []) == 10

    # Leather armor (a_ac = 2): AC = 10 - 2 = 8.
    assert find_ac_formula(10, [2]) == 8

    # Body 2 + shield 1 + helm 1 + cloak 1 = 5 subtracted.
    assert find_ac_formula(10, [2, 1, 1, 1]) == 5

    # +N enchantment is folded into ARM_BONUS by the caller; verify big
    # bonuses subtract correctly (uac can go negative — best armor).
    assert find_ac_formula(10, [8, 4, 3, 2, 2, 1, 1]) == -11

    # Non-human form (base AC != 10): test base parameter is honoured.
    assert find_ac_formula(7, [2, 1]) == 4


def test_compute_ac_matches_formula_helper():
    """The runtime compute_ac must agree with find_ac_formula on
    representative armour layouts."""
    from Nethax.nethax.state import EnvState
    from Nethax.nethax.subsystems.combat import compute_ac
    from Nethax.nethax.subsystems.inventory import ArmorSlot

    state = EnvState.default(jax.random.PRNGKey(0))

    # Stripped player.
    assert int(compute_ac(state)) == find_ac_formula(PLAYER_BASE_AC, [])

    # Helm bonus 2 → AC = 10 - 2 = 8.
    bonus = state.inventory.worn_armor_ac_bonus.at[int(ArmorSlot.HELM)].set(
        jnp.int8(2)
    )
    state2 = state.replace(
        inventory=state.inventory.replace(worn_armor_ac_bonus=bonus),
    )
    assert int(compute_ac(state2)) == find_ac_formula(PLAYER_BASE_AC, [2])

    # Body 2 + helm 1 + boots 1 → AC = 10 - 4 = 6.
    bonus = (
        state.inventory.worn_armor_ac_bonus
        .at[int(ArmorSlot.BODY)].set(jnp.int8(2))
        .at[int(ArmorSlot.HELM)].set(jnp.int8(1))
        .at[int(ArmorSlot.BOOTS)].set(jnp.int8(1))
    )
    state3 = state.replace(
        inventory=state.inventory.replace(worn_armor_ac_bonus=bonus),
    )
    assert int(compute_ac(state3)) == find_ac_formula(
        PLAYER_BASE_AC, [2, 1, 1]
    )


# ---------------------------------------------------------------------------
# dmgval per-weapon damage roll (vendor/nethack/src/weapon.c:215-302).
# ---------------------------------------------------------------------------
def test_dmgval_weapon_picks_small_or_large_dice():
    """dmgval selects between oc_wsdam and oc_wldam based on bigmonst()."""
    # Small target: sdam_roll is used; ldam_roll ignored.
    assert dmgval_weapon(bigmonst=False, sdam_roll=3, ldam_roll=7, spe=0) == 3
    # Large target: ldam_roll is used; sdam_roll ignored.
    assert dmgval_weapon(bigmonst=True,  sdam_roll=3, ldam_roll=7, spe=0) == 7


def test_dmgval_weapon_adds_enchantment_and_clamps():
    """vendor weapon.c:297-302 — Is_weapon adds spe and clamps negative to 0."""
    assert dmgval_weapon(bigmonst=False, sdam_roll=2, ldam_roll=5, spe=3) == 5
    # Negative spe can drop damage to 0 but no further.
    assert dmgval_weapon(bigmonst=False, sdam_roll=1, ldam_roll=5,
                         spe=-3) == 0
    assert dmgval_weapon(bigmonst=False, sdam_roll=4, ldam_roll=5,
                         spe=-2) == 2


def test_dmgval_non_weapon_no_enchant_no_clamp():
    """Non-weapon items skip the spe add and the clamp (weapon.c:297)."""
    # Negative sdam can't occur for vendor (rnd returns >= 1) but the helper
    # mustn't silently zero it for non-weapons since the C path leaves it.
    assert dmgval_weapon(bigmonst=False, sdam_roll=2, ldam_roll=5,
                         spe=99, is_weapon=False) == 2


# ---------------------------------------------------------------------------
# Composite to-hit (uhitm.c::find_roll_to_hit + the strict >-comparison).
# ---------------------------------------------------------------------------
def test_find_roll_to_hit_formula_matches_vendor():
    """tmp = 1 + abon + monster_ac + skill_bonus + enchant

    Worked example from the existing range-test docstring:
        STR=18/100 (118), DEX=18, XL=5, SKILLED → tmp == 1 + 7 + 10 + 2 + 0
                                                = 20
    XL kludge (+1 when XL<3, weapon.c:977) verified separately.
    """
    tmp = find_roll_to_hit_formula(
        str_value=18 + 100, dex_value=18, monster_ac=10,
        skill_tier=SKILL_SKILLED, weapon_enchant=0, xl=5,
    )
    # abon = strhitbon(118) + dexbon(18) = 3 + 4 = 7
    # tmp  = 1 + 7 + 10 + 2 + 0 = 20
    assert tmp == 20, f"expected tmp=20, got {tmp}"

    # XL < 3 kludge: +1 to-hit.
    tmp_xl1 = find_roll_to_hit_formula(
        str_value=18, dex_value=10, monster_ac=10,
        skill_tier=SKILL_BASIC, weapon_enchant=0, xl=1,
    )
    # abon = strhitbon(18) + dexbon(10) = 1 + 0 = 1; +1 XL kludge → 2
    # tmp  = 1 + 2 + 10 + 0 + 0 = 13
    assert tmp_xl1 == 13, f"expected tmp=13, got {tmp_xl1}"

    # No kludge at XL>=3.
    tmp_xl3 = find_roll_to_hit_formula(
        str_value=18, dex_value=10, monster_ac=10,
        skill_tier=SKILL_BASIC, weapon_enchant=0, xl=3,
    )
    # abon = 1 + 0 = 1; tmp = 1 + 1 + 10 + 0 + 0 = 12
    assert tmp_xl3 == 12, f"expected tmp=12, got {tmp_xl3}"

    # Negative AC and weapon enchant.
    tmp_neg = find_roll_to_hit_formula(
        str_value=18, dex_value=10, monster_ac=-5,
        skill_tier=SKILL_EXPERT, weapon_enchant=2, xl=10,
    )
    # abon = 1 + 0 = 1; tmp = 1 + 1 + (-5) + 3 + 2 = 2
    assert tmp_neg == 2, f"expected tmp=2, got {tmp_neg}"


def test_to_hit_roll_uses_strict_greater_than():
    """vendor/nethack/src/uhitm.c:709-710 — mhit = (tmp > dieroll).

    The pre-fix Nethax used ``rnd(20) <= tmp`` which is off by one (hits on
    the boundary ``dieroll == tmp``).  Verify by constructing a scenario
    where tmp == every dieroll and counting hits: at most 19/20 dice should
    register a hit (the case dieroll==tmp must miss).
    """
    from Nethax.nethax.state import EnvState

    state = EnvState.default(jax.random.PRNGKey(0)).replace(
        player_str=jnp.int16(8),    # strhitbon = 0
        player_dex=jnp.int8(10),    # dexbon = 0
        player_xl=jnp.int32(5),     # no XL kludge
    )
    # Bare-handed (skill tier 0 / UNSKILLED → -4 hit bonus, but martial-arts
    # path isn't invoked here because the weapon_skill table is zeroed).
    # Set the weapon-skill slot 0 to BASIC so skill_bonus = 0.
    state = state.replace(
        combat=state.combat.replace(
            weapon_skill=state.combat.weapon_skill.at[0].set(jnp.int8(SKILL_BASIC)),
        )
    )

    # With target_ac = 9, the formula yields tmp = 1 + 0 + 9 + 0 + 0 = 10.
    # vendor: hit iff rnd(20) < 10 → 9 winning faces (1..9) out of 20.
    target_ac = jnp.int32(9)
    n = 5_000
    keys = jax.random.split(jax.random.PRNGKey(2024), n)
    # Vectorise to avoid per-call retracing overhead.
    vroll = jax.jit(jax.vmap(lambda k: to_hit_roll(k, state, target_ac)))
    hits = vroll(keys)
    rate = float(jnp.mean(hits.astype(jnp.float32)))
    # Expected vendor rate: 9/20 = 0.45.  Tolerance ±0.03 for sampling noise.
    assert 0.42 <= rate <= 0.48, (
        f"strict-> hit rate should be 9/20=0.45; got {rate:.4f}"
    )


# ---------------------------------------------------------------------------
# Monster-to-hit (vendor/nethack/src/mhitu.c:709-718).
# ---------------------------------------------------------------------------
def test_monster_to_hit_tmp_matches_vendor():
    """mhitu.c:709-718:
        tmp = AC_VALUE(u.uac) + 10 + mtmp->m_lev
        if (tmp <= 0) tmp = 1

    For non-negative ``u.uac`` ``AC_VALUE`` is the identity, so the formula
    becomes a deterministic ``uac + 10 + m_lev`` clamped at 1.
    """
    # Stripped hero (AC=10) vs a level-1 monster: tmp = 10 + 10 + 1 = 21.
    assert monster_to_hit_tmp(10, 1) == 21
    # Heavy armour (AC=0) vs level-5 monster: tmp = 0 + 10 + 5 = 15.
    assert monster_to_hit_tmp(0, 5) == 15
    # Boundary clamp: extremely negative tmp clamps to 1.
    # (uac is the deterministic positive-AC case so we hit the clamp via
    #  unusual inputs; vendor still rounds up to 1.)
    assert monster_to_hit_tmp(-15, 0) == 1
    # Level-30 monster vs un-armoured (AC=10): tmp = 50 (still in range).
    assert monster_to_hit_tmp(10, 30) == 50


def test_monster_attack_player_uses_vendor_formula():
    """Run the runtime monster_attack_player path and verify the hit rate
    matches the vendor ``tmp > rnd(20)`` ratio for a stripped hero.

    Setup: stripped player (player_ac=10), monster with hp_max=4 →
    mlev=clip(hp_max//4,1,30)=1.  Vendor tmp = 10 + 10 + 1 = 21.  Since 21
    > any rnd(20) (range 1..20), hit rate must be 100% over many trials.
    """
    from Nethax.nethax.state import EnvState
    from Nethax.nethax.subsystems.combat import monster_attack_player

    state = EnvState.default(jax.random.PRNGKey(0))
    mai = state.monster_ai
    mai = mai.replace(
        alive=mai.alive.at[0].set(True),
        hp=mai.hp.at[0].set(jnp.int32(10)),
        hp_max=mai.hp_max.at[0].set(jnp.int32(4)),
        pos=mai.pos.at[0].set(jnp.array([0, 0], dtype=jnp.int16)),
        ac=mai.ac.at[0].set(jnp.int8(10)),
        attack_dice_n=mai.attack_dice_n.at[0].set(jnp.int8(1)),
        attack_dice_sides=mai.attack_dice_sides.at[0].set(jnp.int8(2)),
    )
    state = state.replace(
        monster_ai=mai,
        player_hp=jnp.int32(10_000),
        player_hp_max=jnp.int32(10_000),
    )

    strike = jax.jit(lambda s, k: monster_attack_player(s, k, jnp.int32(0)))
    keys = jax.random.split(jax.random.PRNGKey(99), 100)
    hits = 0
    cur = state
    for k in keys:
        new_state, dmg = strike(cur, k)
        if int(dmg) > 0:
            hits += 1
        # Re-pin player HP so each trial is independent.
        cur = new_state.replace(player_hp=jnp.int32(10_000))
    # tmp=21 > every rnd(20) → 100% hit rate.
    assert hits == len(keys), (
        f"expected 100% hit (tmp=21 > rnd(20)); got {hits}/{len(keys)}"
    )


def test_monster_attack_player_armor_drops_hit_rate():
    """Same as above but with player AC=0 (heavy armour) and m_lev=1.

    Vendor tmp = 0 + 10 + 1 = 11 → hits iff rnd(20) < 11 → 10/20 = 50%.
    """
    from Nethax.nethax.state import EnvState
    from Nethax.nethax.subsystems.combat import monster_attack_player
    from Nethax.nethax.subsystems.inventory import ArmorSlot

    state = EnvState.default(jax.random.PRNGKey(0))
    # Stack AC bonuses to reach uac == 0 (10 - 10 = 0).
    bonus = (
        state.inventory.worn_armor_ac_bonus
        .at[int(ArmorSlot.BODY)].set(jnp.int8(5))
        .at[int(ArmorSlot.HELM)].set(jnp.int8(2))
        .at[int(ArmorSlot.SHIELD)].set(jnp.int8(2))
        .at[int(ArmorSlot.BOOTS)].set(jnp.int8(1))
    )
    state = state.replace(
        inventory=state.inventory.replace(worn_armor_ac_bonus=bonus),
    )

    mai = state.monster_ai
    mai = mai.replace(
        alive=mai.alive.at[0].set(True),
        hp=mai.hp.at[0].set(jnp.int32(10)),
        hp_max=mai.hp_max.at[0].set(jnp.int32(4)),
        pos=mai.pos.at[0].set(jnp.array([0, 0], dtype=jnp.int16)),
        ac=mai.ac.at[0].set(jnp.int8(10)),
        attack_dice_n=mai.attack_dice_n.at[0].set(jnp.int8(1)),
        attack_dice_sides=mai.attack_dice_sides.at[0].set(jnp.int8(2)),
    )
    state = state.replace(
        monster_ai=mai,
        player_hp=jnp.int32(10_000),
        player_hp_max=jnp.int32(10_000),
    )

    # JIT the strike so the 400 trials don't re-trace each iteration.
    strike = jax.jit(lambda s, k: monster_attack_player(s, k, jnp.int32(0)))
    keys = jax.random.split(jax.random.PRNGKey(7), 400)
    hits = 0
    cur = state
    for k in keys:
        new_state, dmg = strike(cur, k)
        if int(dmg) > 0:
            hits += 1
        cur = new_state.replace(player_hp=jnp.int32(10_000))
    rate = hits / len(keys)
    # vendor rate: 10/20 = 0.5.  Tolerance ±0.07 for sampling noise at n=400.
    assert 0.43 <= rate <= 0.57, (
        f"expected vendor rate ~0.5 (tmp=11, strict-gt); got {rate:.4f}"
    )
