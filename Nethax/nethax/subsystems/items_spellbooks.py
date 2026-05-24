"""Spellbook reading subsystem — study_book + cursed_book logic.

Canonical sources:
  vendor/nethack/src/spell.c::study_book   — learning chance, memory init
  vendor/nethack/src/spell.c::cursed_book  — 7-branch backfire (lines 130-185)
  vendor/nethack/include/spell.h           — KEEN = 20000, SPELL_LEV_PW
  vendor/nethack/include/objects.h         — SPELL() table (level, delay)

Vendor formula (spell.c::study_book lines 577-603):
  if (!blessed && otyp != SPE_BOOK_OF_THE_DEAD):
      if (cursed):       too_hard = TRUE      → cursed_book backfire
      else:              read_ability = INT + 4 + xl/2 - 2*oc_level
                                       + (lenses ? 2 : 0)
                         if rnd(20) > read_ability: too_hard = TRUE
  # blessed (and SPE_BOOK_OF_THE_DEAD) skip the failure roll entirely.

  We use player_int for A_INT and player_xl for u.ulevel.  For the Wizard
  role (role_id=12) we apply an additional +_WIZARD_STUDY_BONUS to model
  the wizard-only "too difficult?" early check (spell.c line 587).

  On success: spell_known[spell_id] = True, spell_memory = KEEN + 1
    (vendor incrnknow(i, 1); spell.c line 22 + lines 410/428)
  On failure: no change.

Cursed-book backfire (vendor spell.c::cursed_book lines 130-185):
  switch (rn2(lev))  where lev = objects[booktype].oc_level:
    case 0  (line 137):  tele()                 — random teleport
    case 1  (line 141):  aggravate()            — wake monsters on level
    case 2  (line 145):  make_blinded(rn1(100, 250))
    case 3  (line 148):  take_gold()            — leprechaun-style theft
    case 4  (line 151):  make_confused(rn1(7, 16))
    case 5  (line 155):  poison_strdmg(...)     — STR drain
    case 6  (line 169):  dmg = 2*rnd(10) + 5    — explode (Antimagic→0)
    default (line 180):  rndcurse()             — only fires for lev≥8 (never
                                                  in vanilla NetHack; lev≤7).

Wave-15+ simplifications and deferrals (audited):
  - Lenses (spell.c line 584): not modelled — no worn-blindfold slot.
  - Dull-book sleep (spell.c lines 474-494): deferred — vendor's "dull"
    description is per-game-procedural; no Item field tracks it.
  - MAX_SPELL_STUDY faded-book path (spell.c lines 401-411): deferred —
    requires a per-Item `spestudied` counter that doesn't yet exist.
  - Antimagic intrinsic: not yet modelled — treated as always FALSE so the
    explode-branch Antimagic gate is a no-op (full damage applies).
  - Multi-turn occupation (spell.c line 608 `nomul + set_occupation(learn)`):
    deferred — nethax has no occupation primitive; we apply the full
    study_book_delay() turn cost atomically on success.
"""

import jax
import jax.numpy as jnp

from Nethax.nethax.rng import rnd, rn1, rn2
from Nethax.nethax.subsystems.magic import (
    KEEN,
    MagicState,
    MAX_SPELL_MEMORY,
    N_SPELLS,
    SpellId,
    _SPELL_LEVELS,
)


# Blank-book sentinel: slot_spell_id == -1 means no spell (blank or novel).
BLANK_SPELL_ID = -1

# Wizard role index (role.c: PM_WIZARD = 12).
_ROLE_WIZARD = 12

# Wizard comprehension bonus (models Wizard's higher studied INT advantage).
# Vendor: wizards get an early-out "too difficult?" check (spell.c line 587)
# plus lens bonus (+2).  We give +2 flat for Wizard role.
_WIZARD_STUDY_BONUS = 2

# BUC status constants (containers.BUCStatus — 1=cursed, 2=uncursed, 3=blessed).
_BUC_CURSED   = 1
_BUC_UNCURSED = 2
_BUC_BLESSED  = 3

# Blessed books skip the failure roll entirely (vendor spell.c line 577:
# ``if (!spellbook->blessed && spellbook->otyp != SPE_BOOK_OF_THE_DEAD)``);
# there is no separate "+N" bonus.  Kept as a legacy alias = 0 so importers
# that reference it (e.g. older tests) still resolve to a non-effect.
_BLESSED_STUDY_BONUS = 0


# ---------------------------------------------------------------------------
# Study-time formula
# Cite: vendor/nethack/src/spell.c::study_book lines 537-559.
# Vendor switches on objects[booktype].oc_level:
#     level 1-2: delay = -oc_delay
#     level 3-4: delay = -(oc_level - 1) * oc_delay
#     level 5-6: delay = -oc_level * oc_delay
#     level 7:   delay = -8 * oc_delay
# The negative sign in svc.context.spbook.delay simply marks it as a study
# interruption; magnitude is the turn count.  Per-spell oc_delay values are
# the SPELL() macro's `delay` field (vendor objects.h line 1277:
# SPELL(name, desc, sub, prob, delay, level, mgc, dir, color, sn)).
# ---------------------------------------------------------------------------

# oc_delay per spell, indexed by SpellId (objects.h SPELL() entries).
# Source: vendor/nethack/include/objects.h lines 1293-1396.
_SPELL_OC_DELAYS: tuple = (
    6,  # DIG
    2,  # MAGIC_MISSILE
    4,  # FIREBALL
    7,  # CONE_OF_COLD
    1,  # SLEEP
    10, # FINGER_OF_DEATH
    1,  # LIGHT
    1,  # DETECT_MONSTERS
    5,  # HEALING
    1,  # KNOCK
    2,  # FORCE_BOLT
    2,  # CONFUSE_MONSTER
    2,  # CURE_BLINDNESS
    2,  # DRAIN_LIFE
    2,  # SLOW_MONSTER
    3,  # WIZARD_LOCK
    3,  # CREATE_MONSTER
    3,  # DETECT_FOOD
    3,  # CAUSE_FEAR
    3,  # CLAIRVOYANCE
    3,  # CURE_SICKNESS
    6,  # CHARM_MONSTER
    4,  # HASTE_SELF
    3,  # DETECT_UNSEEN
    4,  # LEVITATION
    5,  # EXTRA_HEALING
    5,  # RESTORE_ABILITY
    5,  # INVISIBILITY
    4,  # DETECT_TREASURE
    5,  # REMOVE_CURSE
    5,  # MAGIC_MAPPING
    3,  # IDENTIFY
    8,  # TURN_UNDEAD
    7,  # POLYMORPH
    6,  # TELEPORT_AWAY
    7,  # CREATE_FAMILIAR
    8,  # CANCELLATION
    1,  # PROTECTION
    3,  # JUMPING
    3,  # STONE_TO_FLESH
    2,  # CHAIN_LIGHTNING
    1,  # FLAME_SPHERE
    1,  # FREEZE_SPHERE
)


def study_book_delay(book_level: int, oc_delay: int) -> int:
    """Return the number of turns required to study a spellbook.

    Mirrors the vendor switch in spell.c::study_book (lines 537-559):
        level 1-2: delay = oc_delay
        level 3-4: delay = (level - 1) * oc_delay
        level 5-6: delay = level * oc_delay
        level 7:   delay = 8 * oc_delay

    Cite: vendor/nethack/src/spell.c::study_book lines 537-559.
    """
    if book_level <= 2:
        return oc_delay
    if book_level <= 4:
        return (book_level - 1) * oc_delay
    if book_level <= 6:
        return book_level * oc_delay
    return 8 * oc_delay


def _assign_letter(magic: MagicState, spell_id: int) -> MagicState:
    """Assign the first unused a-z / A-Z letter to this spell.

    Letters 0-25 map to 'a'-'z', 26-51 map to 'A'-'Z'.
    If all 52 are taken, leave spell_letter[spell_id] at -1.
    """
    used = magic.spell_letter  # [N_SPELLS] int8, -1 = unbound
    for letter_idx in range(52):
        taken = bool(jnp.any(used == jnp.int8(letter_idx)))
        if not taken:
            new_letters = magic.spell_letter.at[spell_id].set(jnp.int8(letter_idx))
            return magic.replace(spell_letter=new_letters)
    return magic


def study_success_chance(
    player_int: int,
    player_xl: int,
    book_level: int,
    role_id: int = 0,
    buc_status: int = 2,
) -> float:
    """Return success probability in [0.0, 1.0] for studying a spellbook.

    Vendor formula (spell.c::study_book lines 577-602):
        if (!blessed && otyp != SPE_BOOK_OF_THE_DEAD):
            if (cursed) too_hard = TRUE   → backfire, chance 0.0
            else read_ability = INT + 4 + xl//2 - 2*book_level
                 success iff rnd(20) <= read_ability   (rnd(20) in 1..20)
                 => chance = clamp(read_ability, 0, 20) / 20
        else: blessed (or Book of the Dead) → automatic success, chance 1.0

    Wizard (+_WIZARD_STUDY_BONUS) bonus is applied before clamping.

    Cite: vendor/nethack/src/spell.c::study_book lines 577-602.
    This function is used outside JIT (e.g. for tests / host-side validation).
    """
    # Cursed books always backfire — never learn (vendor lines 577-580).
    if buc_status == _BUC_CURSED:
        return 0.0
    # Blessed books skip the failure roll entirely (vendor line 577).
    if buc_status == _BUC_BLESSED:
        return 1.0
    ra = player_int + 4 + player_xl // 2 - 2 * book_level
    if role_id == _ROLE_WIZARD:
        ra += _WIZARD_STUDY_BONUS
    ra = max(0, min(ra, 20))
    return ra / 20.0


# ---------------------------------------------------------------------------
# Cursed-book backfire — 8-branch vendor switch
# Cite: vendor/nethack/src/spell.c::cursed_book lines 130-185.
#
# Vendor: switch (rn2(lev)) where lev = objects[booktype].oc_level (1..7).
#   case 0   tele()                          rloc-style random teleport
#   case 1   aggravate()                     wake all monsters on level
#   case 2   make_blinded(rn1(100, 250))     250..349 turns of blindness
#   case 3   take_gold()                     leprechaun-style gold theft
#   case 4   make_confused(rn1(7, 16))       16..22 turns of confusion
#   case 5   poison_strdmg → STR drain       1d4 STR drop (+/- res rolls)
#   case 6   dmg = 2*rnd(10) + 5; book gone  Antimagic gates damage to 0
#   default  rndcurse()                      vanilla unreachable (lev≤7)
# ---------------------------------------------------------------------------

# Antimagic gate: vendor spell.c:170 checks the Antimagic intrinsic.  Nethax
# does not yet model that intrinsic (see items_potions.py:474, 483 — same
# deferral note).  Treated as always FALSE so case 6 always inflicts damage.
_PLAYER_HAS_ANTIMAGIC = False


def _cursed_book_backfire(state, rng: jax.Array, slot_idx: int, book_level: int):
    """Apply a cursed-book backfire effect (vendor cursed_book switch).

    Cite: vendor/nethack/src/spell.c::cursed_book lines 130-185.

    ``book_level`` is the spell's ``oc_level`` (1..7).  Branch index is
    ``rn2(book_level)``; with vanilla lev≤7 the default branch (rndcurse)
    is structurally unreachable but kept in the switch for vendor parity.

    All 8 branches return the same pytree shape so this is safe for both
    eager and JIT execution.  The driver (:func:`read_spellbook`) extracts
    spell_id / buc_status with Python ``int(...)`` so the outer scope is
    not JIT-compiled today; the inner ``lax.switch`` keeps us compatible
    with a future JIT lowering pass.
    """
    from Nethax.nethax.subsystems.status_effects import TimedStatus, _roll_rn1

    # One RNG split for every branch's stochastic decisions.  We split into
    # disjoint sub-keys so any branch can consume its own without reuse.
    (rng_branch, sub_tele, sub_blind, sub_gold_amt, sub_gold_pos,
     sub_confuse, sub_poison, sub_dmg, sub_curse) = jax.random.split(rng, 9)

    lev = max(1, int(book_level))      # static; safe for lax.switch arity

    def b0_teleport(s):
        # vendor spell.c:139  tele();  — random teleport to a FLOOR tile.
        # Cite: vendor/nethack/src/teleport.c::dotele (rloc-equivalent).
        from Nethax.nethax.constants.tiles import TileType
        br = s.dungeon.current_branch.astype(jnp.int32)
        lv = s.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)
        level_tiles = s.terrain[br, lv]                       # [H, W]
        floor_mask = level_tiles == jnp.int8(int(TileType.FLOOR))
        flat_mask = floor_mask.reshape(-1).astype(jnp.float32)
        total = jnp.sum(flat_mask)
        H, W = level_tiles.shape
        has_floor = total > 0
        probs = jnp.where(
            has_floor,
            flat_mask / jnp.maximum(total, jnp.float32(1.0)),
            jnp.ones((H * W,), dtype=jnp.float32) / jnp.float32(H * W),
        )
        flat_idx = jax.random.choice(sub_tele, H * W, p=probs).astype(jnp.int32)
        new_row = (flat_idx // W).astype(jnp.int16)
        new_col = (flat_idx % W).astype(jnp.int16)
        new_pos = jnp.stack([new_row, new_col])
        out_pos = jnp.where(has_floor, new_pos, s.player_pos)
        return s.replace(player_pos=out_pos)

    def b1_aggravate(s):
        # vendor spell.c:143  aggravate();  — wakes every monster on the
        # current level.  Vendor scans fmon (all monsters); we approximate
        # with wake_monsters_near using a level-spanning Chebyshev radius.
        # Cite: vendor/nethack/src/wizard.c::aggravate lines 493-511.
        from Nethax.nethax.subsystems.monster_ai import wake_monsters_near
        return wake_monsters_near(s, s.player_pos, radius=999, petcall=False)

    def b2_blind(s):
        # vendor spell.c:146  make_blinded(BlindedTimeout + rn1(100, 250), TRUE);
        # rn1(100, 250) → 250..349 turns added to current BLIND timer.
        # Cite: vendor/nethack/src/spell.c::cursed_book line 146.
        ts = s.status.timed_statuses
        add = _roll_rn1(sub_blind, 100, 250)
        new_blind = ts[int(TimedStatus.BLIND)] + add
        new_ts = ts.at[int(TimedStatus.BLIND)].set(new_blind)
        return s.replace(status=s.status.replace(timed_statuses=new_ts))

    def b3_take_gold(s):
        # vendor spell.c:149  take_gold();  — leprechaun-style theft.
        # Vendor mhitu.c::take_gold steals min(igold, somegold(igold)).
        # Cite: vendor/nethack/src/spell.c::cursed_book line 149.
        gold = s.player_gold.astype(jnp.int32)
        # Vendor somegold() bracketed rn1 ranges, mirroring
        # monster_actions.py::_leprechaun_steal_gold (steal.c:14-34).
        bracket_n = jnp.where(
            gold < jnp.int32(50), jnp.int32(1),
            jnp.where(gold < jnp.int32(100), gold - jnp.int32(25) + jnp.int32(1),
            jnp.where(gold < jnp.int32(500), gold - jnp.int32(50) + jnp.int32(1),
            jnp.where(gold < jnp.int32(1000), gold - jnp.int32(100) + jnp.int32(1),
            jnp.where(gold < jnp.int32(5000), gold - jnp.int32(500) + jnp.int32(1),
            jnp.where(gold < jnp.int32(10000), gold - jnp.int32(1000) + jnp.int32(1),
                                                gold - jnp.int32(5000) + jnp.int32(1)))))),
        )
        bracket_x = jnp.where(
            gold < jnp.int32(50), jnp.int32(0),
            jnp.where(gold < jnp.int32(100), jnp.int32(25),
            jnp.where(gold < jnp.int32(500), jnp.int32(50),
            jnp.where(gold < jnp.int32(1000), jnp.int32(100),
            jnp.where(gold < jnp.int32(5000), jnp.int32(500),
            jnp.where(gold < jnp.int32(10000), jnp.int32(1000),
                                                jnp.int32(5000)))))),
        )
        safe_n = jnp.maximum(bracket_n, jnp.int32(1))
        rn2_roll = jax.random.randint(sub_gold_amt, (), 0, safe_n, dtype=jnp.int32)
        rn1_result = (bracket_x + rn2_roll).astype(jnp.int32)
        stolen = jnp.where(gold < jnp.int32(50), gold, rn1_result)
        stolen = jnp.minimum(stolen, gold)
        new_gold = jnp.maximum(gold - stolen, jnp.int32(0)).astype(jnp.int32)
        return s.replace(player_gold=new_gold)

    def b4_confuse(s):
        # vendor spell.c:153  make_confused(HConfusion + rn1(7, 16), FALSE);
        # rn1(7, 16) → 16..22 turns added to current CONFUSION timer.
        # Cite: vendor/nethack/src/spell.c::cursed_book line 153.
        ts = s.status.timed_statuses
        add = _roll_rn1(sub_confuse, 7, 16)
        new_conf = ts[int(TimedStatus.CONFUSION)] + add
        new_ts = ts.at[int(TimedStatus.CONFUSION)].set(new_conf)
        return s.replace(status=s.status.replace(timed_statuses=new_ts))

    def b5_poison(s):
        # vendor spell.c:164  poison_strdmg(rn1(2,1)|rn1(4,3), rnd(6|10), ...);
        # STR is drained by 1d4 (we use 1) and ATTRIBUTE_AWAY is set so the
        # drain re-applies via the existing attribute_away timeout machinery.
        # Cite: vendor/nethack/src/spell.c::cursed_book lines 155-168.
        ts = s.status.timed_statuses
        new_attr = jnp.int32(10)
        new_ts = ts.at[int(TimedStatus.ATTRIBUTE_AWAY)].set(new_attr)
        # str -= rnd(2); floor at 3 to match vendor poison_strdmg semantics.
        drain = rnd(sub_poison, 2)
        new_str = jnp.maximum(
            s.player_str - drain.astype(jnp.int16), jnp.int16(3)
        ).astype(jnp.int16)
        return s.replace(
            player_str=new_str,
            status=s.status.replace(timed_statuses=new_ts),
        )

    def b6_explode(s):
        # vendor spell.c:169-179:
        #     if (Antimagic) { pline("...unharmed!"); }
        #     else {
        #       dmg = 2 * rnd(10) + 5;
        #       losehp(Maybe_Half_Phys(dmg), "exploding rune", KILLED_BY_AN);
        #     }
        #     return TRUE;  /* caller destroys the book */
        # Antimagic intrinsic not yet modelled in nethax → always FALSE.
        # Cite: vendor/nethack/src/spell.c::cursed_book lines 169-179.
        dmg = jnp.int32(2) * rnd(sub_dmg, 10) + jnp.int32(5)
        gated_dmg = jnp.where(_PLAYER_HAS_ANTIMAGIC, jnp.int32(0), dmg)
        new_hp = jnp.maximum(s.player_hp - gated_dmg, jnp.int32(1))
        # Book is destroyed (vendor return TRUE → useup(book) in study_book).
        new_qty = s.inventory.items.quantity.at[slot_idx].set(jnp.int16(0))
        new_items = s.inventory.items.replace(quantity=new_qty)
        new_inv = s.inventory.replace(items=new_items)
        return s.replace(player_hp=new_hp, inventory=new_inv)

    def b_default(s):
        # vendor spell.c:181  rndcurse();
        # Cite: vendor/nethack/src/sit.c::rndcurse — flips one or more random
        # inventory items toward cursed.  Reuses the shared scrolls helper.
        from Nethax.nethax.subsystems.items_scrolls import rndcurse
        return rndcurse(s, sub_curse)

    branches = (
        b0_teleport, b1_aggravate, b2_blind, b3_take_gold,
        b4_confuse,  b5_poison,    b6_explode, b_default,
    )

    # Branch index = rn2(lev).  rn2_lev ∈ [0, lev); for lev≤7 the default
    # branch (index 7) is unreachable, matching vendor (lev never ≥ 8).
    # lax.switch clamps the index into [0, len(branches)-1], so an
    # accidental lev=8 still selects the rndcurse default.
    rn2_lev = rn2(rng_branch, lev).astype(jnp.int32)
    return jax.lax.switch(rn2_lev, branches, state)


def read_spellbook(state, rng: jax.Array, slot_idx: int):
    """Read a spellbook from inventory slot `slot_idx`.

    Looks up the spell_id stored in inventory item at slot_idx via
    ``state.inventory.items[slot_idx].type_id``.

    Returns updated state.

    BUC handling (vendor spell.c::study_book lines 577-603):
      CURSED  (buc_status == 1): skips success roll, calls cursed_book()
        which selects a backfire branch via ``switch (rn2(oc_level))``
        — see :func:`_cursed_book_backfire` for the 8 vendor branches.
        Spell is never learned on the cursed path.
        Cite: vendor/nethack/src/spell.c::cursed_book lines 130-185.
      UNCURSED (buc_status == 2): standard formula (see below).
      BLESSED  (buc_status == 3): vendor short-circuits the failure check
        entirely (spell.c line 577: ``!blessed && otyp != SPE_BOOK_OF_THE_DEAD``
        gates the whole roll); blessed books always succeed.

    Vendor study check (spell.c::study_book lines 582-599, uncursed path):
        read_ability = INT + 4 + xl//2 - 2 * book_level
        [+ _WIZARD_STUDY_BONUS for Wizard role; spell.c line 587]
        roll = jax.random.randint in [1..20]
        success iff roll <= read_ability

    On success:
        spell_known[spell_id]  = True
        spell_memory[spell_id] = KEEN + 1  (vendor incrnknow(i, 1);
            spell.c line 22 + lines 410/428)
        assign inventory letter if not yet assigned

    On failure:
        state unchanged
    """
    from Nethax.nethax.subsystems.status_effects import TimedStatus

    # Blind / stunned: cannot read (vendor/nethack/src/read.c::doread early checks).
    is_blind   = state.status.timed_statuses[int(TimedStatus.BLIND)]   > jnp.int32(0)
    is_stunned = state.status.timed_statuses[int(TimedStatus.STUNNED)] > jnp.int32(0)
    if bool(is_blind) or bool(is_stunned):
        return state

    # --- resolve spell_id and buc_status from inventory slot ---
    spell_id   = int(state.inventory.items.type_id[slot_idx])
    buc_status = int(state.inventory.items.buc_status[slot_idx])

    # Blank / novel book: no-op
    if spell_id == BLANK_SPELL_ID or spell_id < 0 or spell_id >= N_SPELLS:
        return state

    book_level = int(_SPELL_LEVELS[spell_id])

    # --- CURSED path (vendor spell.c::cursed_book lines 130-185) ---
    # Vendor: switch (rn2(lev)) where lev = objects[booktype].oc_level.
    #   case 0   tele()                              [spell.c:137-140]
    #   case 1   aggravate()                         [spell.c:141-144]
    #   case 2   make_blinded(rn1(100, 250))         [spell.c:145-147]
    #   case 3   take_gold()                         [spell.c:148-150]
    #   case 4   make_confused(rn1(7, 16))           [spell.c:151-154]
    #   case 5   poison_strdmg → STR drain           [spell.c:155-168]
    #   case 6   explode dmg = 2*rnd(10)+5; book gone[spell.c:169-179]
    #   default  rndcurse()                          [spell.c:180-182]
    #            (unreachable for vanilla lev≤7)
    if buc_status == _BUC_CURSED:
        return _cursed_book_backfire(state, rng, slot_idx, book_level)

    # --- UNCURSED / BLESSED path ---
    # Vendor (spell.c lines 577-602):
    #   if (!blessed && otyp != SPE_BOOK_OF_THE_DEAD) {
    #       if (cursed) too_hard = TRUE;             # handled above
    #       else { read_ability = ... ; if (rnd(20) > read_ability) too_hard; }
    #   }
    # i.e. blessed books skip the failure roll entirely (always succeed).
    # Cite: vendor/nethack/src/spell.c::study_book lines 577-602.
    if buc_status != _BUC_BLESSED:
        player_int = int(state.player_int)
        player_xl  = int(state.player_xl)
        role_id    = int(state.player_role)

        # Vendor formula: read_ability = INT + 4 + xl//2 - 2*book_level
        # (lenses +2 deferred — no worn-blindfold slot in nethax).
        read_ability = player_int + 4 + player_xl // 2 - 2 * book_level
        if role_id == _ROLE_WIZARD:
            read_ability += _WIZARD_STUDY_BONUS
        read_ability = max(0, min(read_ability, 20))

        # Roll 1..20; success if roll <= read_ability.
        rng, sub = jax.random.split(rng)
        roll = int(jax.random.randint(sub, (), 1, 21))

        if roll > read_ability:
            return state

    # --- update MagicState ---
    # Vendor: study_book success calls ``incrnknow(i, 1)`` which sets
    # ``sp_know = KEEN + 1`` (vendor/nethack/src/spell.c lines 410, 428;
    # macro defined line 22).  Byte-equal: KEEN + 1 = 20001.
    magic = state.magic
    new_known = magic.spell_known.at[spell_id].set(True)
    new_mem   = magic.spell_memory.at[spell_id].set(jnp.int32(KEEN + 1))
    magic = magic.replace(spell_known=new_known, spell_memory=new_mem)

    # Assign letter if not yet bound
    if int(magic.spell_letter[spell_id]) == -1:
        magic = _assign_letter(magic, spell_id)

    # Advance timestep by study delay (vendor spell.c::study_book switch
    # at lines 537-559).  Cite via study_book_delay above.
    oc_delay = int(_SPELL_OC_DELAYS[spell_id]) if 0 <= spell_id < len(_SPELL_OC_DELAYS) else 1
    delay_turns = study_book_delay(book_level, oc_delay)
    new_timestep = (state.timestep.astype(jnp.int32) + jnp.int32(delay_turns))

    return state.replace(magic=magic, timestep=new_timestep)


def handle_read_spellbook(state, rng: jax.Array, slot_idx: int):
    """Entry point called from items read-dispatch for SPBOOK_CLASS items.

    Delegates to read_spellbook.
    """
    new_state = read_spellbook(state, rng, slot_idx)
    # Conduct: vendor/nethack/src/read.c::study_book — ILLITERATE broken on
    # reading a spellbook (insight.c ~2147, u.uconduct.literate).
    from Nethax.nethax.subsystems.conduct import Conduct, mark_violated
    return mark_violated(new_state, int(Conduct.ILLITERATE))
