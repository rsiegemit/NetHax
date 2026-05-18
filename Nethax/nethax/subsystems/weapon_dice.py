"""Per-weapon damage dice tables.

Vendor reference: vendor/nethack/src/weapon.c::dmgval (lines 225-295).

For each object type_id, the effective damage is:
  small target: roll(WEAPON_SDAM1_N[t], WEAPON_SDAM1_SIDES[t])
              + roll(WEAPON_SDAM2_N[t], WEAPON_SDAM2_SIDES[t])
  large target: roll(WEAPON_LDAM1_N[t], WEAPON_LDAM1_SIDES[t])
              + roll(WEAPON_LDAM2_N[t], WEAPON_LDAM2_SIDES[t])

Component 1 comes from objects[].oc_wsdam / oc_wldam (objects.py sdam/ldam fields).
Component 2 comes from the per-weapon switch bonuses in weapon.c:228-295.
Index 0 is the fists sentinel (bare-hands): 1d2 small / 1d1 large.
"""
import jax.numpy as jnp


def _build() -> tuple:
    from Nethax.nethax.constants.objects import OBJECTS, NUM_OBJECTS

    n = NUM_OBJECTS
    sdam1_n     = [0] * n
    sdam1_sides = [0] * n
    sdam2_n     = [0] * n
    sdam2_sides = [0] * n
    ldam1_n     = [0] * n
    ldam1_sides = [0] * n
    ldam2_n     = [0] * n
    ldam2_sides = [0] * n

    for i, obj in enumerate(OBJECTS):
        sn, ss = obj.sdam
        ln, ls = obj.ldam
        sdam1_n[i]     = sn
        sdam1_sides[i] = ss
        ldam1_n[i]     = ln
        ldam1_sides[i] = ls

    # Fists sentinel at index 0: 1d2 small / 1d1 large
    # vendor/nethack/src/weapon.c bare-hands path
    sdam1_n[0]     = 1
    sdam1_sides[0] = 2
    ldam1_n[0]     = 1
    ldam1_sides[0] = 1

    # Extra dice from vendor/nethack/src/weapon.c switch statements:
    #
    # Small target (weapon.c:266-295):
    #   +1   -> TRIDENT(16), SPETUM(44), MACE(56), WAR_HAMMER(58), FLAIL(63)
    #   +1d4 -> BATTLE_AXE(28), BARDICHE(48), BILL_GUISARME(53), GUISARME(52),
    #           LUCERN_HAMMER(54), MORNING_STAR(57), RANSEUR(43), BROADSWORD(35),
    #           ELVEN_BROADSWORD(36), RUNESWORD(41), VOULGE(49)
    #
    # Large target (weapon.c:228-261):
    #   +1   -> MORNING_STAR(57), PARTISAN(42), RUNESWORD(41), ELVEN_BROADSWORD(36), BROADSWORD(35)
    #   +1d4 -> FLAIL(63), RANSEUR(43), VOULGE(49)
    #   +1d6 -> HALBERD(47), SPETUM(44)
    #   +2d4 -> BATTLE_AXE(28), BARDICHE(48), TRIDENT(16)
    #   +2d6 -> TWO_HANDED_SWORD(38), TSURUGI(40), DWARVISH_MATTOCK(50)

    for idx in (16, 44, 56, 58, 63):       # small +1 (1d1)
        sdam2_n[idx]     = 1
        sdam2_sides[idx] = 1

    for idx in (28, 35, 36, 41, 43, 48, 49, 52, 53, 54, 57):  # small +1d4
        sdam2_n[idx]     = 1
        sdam2_sides[idx] = 4

    for idx in (35, 36, 41, 42, 57):       # large +1 (1d1)
        ldam2_n[idx]     = 1
        ldam2_sides[idx] = 1

    for idx in (43, 49, 63):               # large +1d4
        ldam2_n[idx]     = 1
        ldam2_sides[idx] = 4

    for idx in (44, 47):                   # large +1d6
        ldam2_n[idx]     = 1
        ldam2_sides[idx] = 6

    for idx in (16, 28, 48):               # large +2d4
        ldam2_n[idx]     = 2
        ldam2_sides[idx] = 4

    for idx in (38, 40, 50):               # large +2d6
        ldam2_n[idx]     = 2
        ldam2_sides[idx] = 6

    return (
        jnp.array(sdam1_n,     dtype=jnp.int8),
        jnp.array(sdam1_sides, dtype=jnp.int8),
        jnp.array(sdam2_n,     dtype=jnp.int8),
        jnp.array(sdam2_sides, dtype=jnp.int8),
        jnp.array(ldam1_n,     dtype=jnp.int8),
        jnp.array(ldam1_sides, dtype=jnp.int8),
        jnp.array(ldam2_n,     dtype=jnp.int8),
        jnp.array(ldam2_sides, dtype=jnp.int8),
    )


(
    WEAPON_SDAM1_N,
    WEAPON_SDAM1_SIDES,
    WEAPON_SDAM2_N,
    WEAPON_SDAM2_SIDES,
    WEAPON_LDAM1_N,
    WEAPON_LDAM1_SIDES,
    WEAPON_LDAM2_N,
    WEAPON_LDAM2_SIDES,
) = _build()


def weapon_damage_dice(type_id: jnp.ndarray, target_large: jnp.ndarray):
    """Return (n1, s1, n2, s2) for the effective damage roll.

    JIT-pure: jnp.clip clamps type_id=-1 (bare-hands) to 0 (fists sentinel).

    Vendor reference: weapon.c::dmgval lines 225-295.
    """
    from Nethax.nethax.constants.objects import NUM_OBJECTS
    safe = jnp.clip(type_id, 0, NUM_OBJECTS - 1)

    sn1 = jnp.take(WEAPON_SDAM1_N,     safe).astype(jnp.int32)
    ss1 = jnp.take(WEAPON_SDAM1_SIDES, safe).astype(jnp.int32)
    sn2 = jnp.take(WEAPON_SDAM2_N,     safe).astype(jnp.int32)
    ss2 = jnp.take(WEAPON_SDAM2_SIDES, safe).astype(jnp.int32)
    ln1 = jnp.take(WEAPON_LDAM1_N,     safe).astype(jnp.int32)
    ls1 = jnp.take(WEAPON_LDAM1_SIDES, safe).astype(jnp.int32)
    ln2 = jnp.take(WEAPON_LDAM2_N,     safe).astype(jnp.int32)
    ls2 = jnp.take(WEAPON_LDAM2_SIDES, safe).astype(jnp.int32)

    n1 = jnp.where(target_large, ln1, sn1)
    s1 = jnp.where(target_large, ls1, ss1)
    n2 = jnp.where(target_large, ln2, sn2)
    s2 = jnp.where(target_large, ls2, ss2)
    return n1, s1, n2, s2
