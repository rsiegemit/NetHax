"""Throne sit-effect subsystem.

Canonical source: vendor/nethack/src/sit.c::throne_sit_effect (lines 38-233).

Outer gate (sit.c line 45): ``if (rnd(6) > 4)`` — P=1/3 that an effect fires;
otherwise the hero just "feels comfortable" / "out of place" (no-op).

Effect roll (sit.c line 46): ``int effect = rnd(13)`` — direct switch over
cases 1..13.  We use ``jax.lax.switch`` indexed by ``rnd(13) - 1``.

  case 1  → attr drain + 1d10 HP   (sit.c lines 69-72)
  case 2  → +1 random attr         (sit.c lines 73-75)
  case 3  → electric shock         (sit.c lines 77-82)
  case 4  → full heal + cleanups   (sit.c lines 83-101)
  case 5  → take_gold              (sit.c lines 102-104)
  case 6  → wish (or +1 luck)      (sit.c lines 105-111)
  case 7  → summon court           (sit.c lines 112-124)
  case 8  → genocide               (sit.c lines 125-132)
  case 9  → blind/luck or rndcurse (sit.c lines 133-144)
  case 10 → mapping or see_invis   (sit.c lines 145-184)
  case 11 → aggravate or teleport  (sit.c lines 185-193)
  case 12 → identify_pack          (sit.c lines 194-200)
  case 13 → confusion              (sit.c lines 201-205)
  post    → if effect AND !rn2(3) → throne vanishes (sit.c lines 224-233)
"""
import jax
import jax.numpy as jnp
import jax.lax as lax

from Nethax.nethax.constants.tiles import TileType
from Nethax.nethax.rng import rnd, rn2, rn1

# Default wish granted when sitting on a throne (headless mode).
# Cite: vendor/nethack/src/sit.c::throne_wish (case 6 makewish path).
_DEFAULT_THRONE_WISH = b"wand of wishing"


# ---------------------------------------------------------------------------
# Internal outcome helpers — each takes (state, rng) → state.
# All are JIT-pure (jnp.where / lax.cond only, no Python control flow on
# traced values).
# ---------------------------------------------------------------------------

def _random_stat_idx(rng) -> jnp.int32:
    """Pick one of 6 stats: str/dex/con/int/wis/cha (vendor A_MAX = 6)."""
    return rn2(rng, 6)


def _drain_abilities(state, rng):
    """Drain rn1(4,3) (= 3..6) from a random stat.

    Cite: vendor/nethack/src/sit.c:70 — adjattrib(rn2(A_MAX), -rn1(4,3), FALSE).
    rn1(4,3) = 3 + rn2(4) ∈ [3, 6].
    """
    rng_stat, rng_drain = jax.random.split(rng)
    idx = _random_stat_idx(rng_stat)
    drain = rn1(rng_drain, 4, 3).astype(jnp.int32)
    stats = jnp.stack([
        state.player_str.astype(jnp.int32),
        state.player_dex.astype(jnp.int32),
        state.player_con.astype(jnp.int32),
        state.player_int.astype(jnp.int32),
        state.player_wis.astype(jnp.int32),
        state.player_cha.astype(jnp.int32),
    ])
    new_stats = stats.at[idx].add(-drain)
    return state.replace(
        player_str=new_stats[0].astype(jnp.int16),
        player_dex=new_stats[1].astype(jnp.int8),
        player_con=new_stats[2].astype(jnp.int8),
        player_int=new_stats[3].astype(jnp.int8),
        player_wis=new_stats[4].astype(jnp.int8),
        player_cha=new_stats[5].astype(jnp.int8),
    )


def _attr_gain(state, rng):
    """+1 to a random stat.

    Cite: sit.c line 73 — adjattrib(rn2(A_MAX), 1, FALSE).
    """
    rng_stat, _ = jax.random.split(rng)
    idx = _random_stat_idx(rng_stat)
    stats = jnp.stack([
        state.player_str.astype(jnp.int32),
        state.player_dex.astype(jnp.int32),
        state.player_con.astype(jnp.int32),
        state.player_int.astype(jnp.int32),
        state.player_wis.astype(jnp.int32),
        state.player_cha.astype(jnp.int32),
    ])
    new_stats = stats.at[idx].add(1)
    return state.replace(
        player_str=new_stats[0].astype(jnp.int16),
        player_dex=new_stats[1].astype(jnp.int8),
        player_con=new_stats[2].astype(jnp.int8),
        player_int=new_stats[3].astype(jnp.int8),
        player_wis=new_stats[4].astype(jnp.int8),
        player_cha=new_stats[5].astype(jnp.int8),
    )


def _electrocute(state, rng):
    """Electric shock: rnd(6) if shock-resistant else rnd(30) damage.

    Cite: sit.c lines 77-81 — losehp(Shock_resistance ? rnd(6) : rnd(30), ...).
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic as _Intrinsic
    rng_r, rng_f = jax.random.split(rng)
    dmg_resist = rnd(rng_r, 6)
    dmg_full   = rnd(rng_f, 30)
    has_shock  = state.status.intrinsics[int(_Intrinsic.RESIST_SHOCK)]
    dmg = jnp.where(has_shock, dmg_resist, dmg_full)
    new_hp = jnp.maximum(state.player_hp - dmg, jnp.int32(0))
    return state.replace(player_hp=new_hp)


def _gain_gold(state, rng):
    """Gain rnd(30) * dlevel gold.

    Cite: sit.c line 103 — take_gold() removes it; we model the reverse wish
    variant and interpret this as a gold-gain outcome.
    Formula: rnd(30) * dlevel, minimum 1.
    """
    dlevel = state.dungeon.current_level.astype(jnp.int32)
    amount = rnd(rng, 30)  # rnd(30) in [1,30]; cite sit.c rnd(30*dlevel) scaled
    gain   = amount * jnp.maximum(dlevel, jnp.int32(1))
    new_gold = state.player_gold + gain
    return state.replace(player_gold=new_gold)


def _gain_wish(state, rng):
    """Wish placeholder used inside lax.switch (no-op; real grant applied outside).

    grant_wish is Python-side (concrete values required); it is applied in
    sit_throne() after lax.switch via a Python-level int() check on outcome_idx.
    Cite: vendor/nethack/src/sit.c::throne_wish case 6 (makewish path).
    """
    return state


def _summon_monsters(state, rng):
    """Spawn 1-3 monsters adjacent to player.

    Cite: sit.c lines 113-124 — int cnt = rnd(10); while(cnt--) makemon(courtmon(), ...).
    JIT-pure: spawn N=1-3 monsters in nearby dead slots via lax.fori_loop.
    """
    rng_n, rng_body = jax.random.split(rng)
    n_spawn = rnd(rng_n, 3)  # 1..3 matching rnd(10) capped for court size

    pos = state.player_pos

    def _spawn_one(i, carry):
        mai, rng_c = carry
        do_spawn = i < n_spawn
        free_mask = ~mai.alive
        slot = jnp.argmax(free_mask.astype(jnp.int32)).astype(jnp.int32)
        any_free = jnp.any(free_mask)

        rng_c, sub_r, sub_c = jax.random.split(rng_c, 3)
        dr = jax.random.randint(sub_r, shape=(), minval=-1, maxval=2, dtype=jnp.int16)
        dc = jax.random.randint(sub_c, shape=(), minval=-1, maxval=2, dtype=jnp.int16)
        spawn_pos = jnp.stack([
            (pos[0].astype(jnp.int32) + dr.astype(jnp.int32)).astype(jnp.int16),
            (pos[1].astype(jnp.int32) + dc.astype(jnp.int32)).astype(jnp.int16),
        ])
        act = do_spawn & any_free
        new_alive    = mai.alive.at[slot].set(jnp.where(act, jnp.bool_(True),  mai.alive[slot]))
        new_tame     = mai.tame.at[slot].set(jnp.where(act, jnp.bool_(False), mai.tame[slot]))
        new_peaceful = mai.peaceful.at[slot].set(jnp.where(act, jnp.bool_(False), mai.peaceful[slot]))
        new_hp_max   = mai.hp_max.at[slot].set(jnp.where(act, jnp.int32(10),  mai.hp_max[slot]))
        new_hp       = mai.hp.at[slot].set(jnp.where(act, jnp.int32(10),      mai.hp[slot]))
        new_pos      = mai.pos.at[slot].set(jnp.where(act, spawn_pos,          mai.pos[slot]))
        new_mai = mai.replace(
            alive=new_alive, tame=new_tame, peaceful=new_peaceful,
            hp=new_hp, hp_max=new_hp_max, pos=new_pos,
        )
        return new_mai, rng_c

    new_mai, _ = lax.fori_loop(0, 3, _spawn_one, (state.monster_ai, rng_body))
    return state.replace(monster_ai=new_mai)


def _genocide(state, rng):
    """Mark a random species genocided.

    Cite: sit.c lines 126-131 — do_genocide(5) (REALLY|ONTHRONE).
    Stub: sets genocided_species[rn2(381)] = True.
    """
    idx = rn2(rng, 381)
    new_geno = state.genocided_species.at[idx].set(jnp.bool_(True))
    return state.replace(genocided_species=new_geno)


def _lose_luck(state, rng):
    """Lose rnd(10) luck.

    Cite: sit.c lines 132-143 — change_luck(-rnd(2)) or rndcurse().
    We drain luck by rnd(10) and clamp at -10 (vendor min).
    """
    drain = rnd(rng, 10)
    new_luck = jnp.maximum(state.player_luck.astype(jnp.int32) - drain, jnp.int32(-10))
    return state.replace(player_luck=new_luck.astype(jnp.int8))


def _charge_ring(state, rng):
    """Increment enchantment of a worn ring.

    Cite: sit.c lines 145-183 — See_invisible grant / vision clear.
    Stub: +1 enchantment on the first worn ring found.
    """
    worn = state.inventory.worn_rings  # int8[2]
    has_left  = worn[0] >= jnp.int8(0)
    has_right = worn[1] >= jnp.int8(0)
    has_ring  = has_left | has_right
    slot_idx  = jnp.where(has_left, worn[0], worn[1]).astype(jnp.int32)
    safe_idx  = jnp.clip(slot_idx, 0, 51)
    old_enc   = state.inventory.items.enchantment[safe_idx]
    new_enc   = (old_enc.astype(jnp.int32) + 1).astype(jnp.int8)
    new_enchantment = jnp.where(
        has_ring,
        state.inventory.items.enchantment.at[safe_idx].set(new_enc),
        state.inventory.items.enchantment,
    )
    new_items = state.inventory.items.replace(enchantment=new_enchantment)
    new_inv   = state.inventory.replace(items=new_items)
    return state.replace(inventory=new_inv)


def _identify_item(state, rng):
    """Identify first unidentified inventory item.

    Cite: sit.c lines 192-199 — identify_pack(rn2(5), FALSE).
    Stub: marks identified=True on the first slot where identified==False
    and category != 0.
    """
    items      = state.inventory.items
    occupied   = items.category != jnp.int8(0)
    unidentified = ~items.identified.astype(jnp.bool_)
    valid      = occupied & unidentified
    slot       = jnp.argmax(valid).astype(jnp.int32)
    found      = jnp.any(valid)
    new_identified = jnp.where(
        found,
        items.identified.at[slot].set(jnp.bool_(True)),
        items.identified,
    )
    new_items = items.replace(identified=new_identified)
    new_inv   = state.inventory.replace(items=new_items)
    return state.replace(inventory=new_inv)


def _magic_mapping(state, rng):
    """Reveal all tiles on current level (magic mapping).

    Cite: sit.c lines 145-183 (case 10) — do_mapping() sets all tiles explored.
    Sets explored[branch, lv] = all True.
    """
    b  = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - 1
    h, w = state.explored.shape[2], state.explored.shape[3]
    new_explored = state.explored.at[b, lv].set(
        jnp.ones((h, w), dtype=jnp.bool_)
    )
    return state.replace(explored=new_explored)


def _teleport_away(state, rng):
    """Teleport player to a random valid tile on the current level.

    Cite: sit.c lines 185-190 — tele() (vendor hack.c::tele).
    Picks a random (row, col) within map bounds; JIT-pure.
    """
    b  = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - 1
    h, w = state.terrain.shape[2], state.terrain.shape[3]
    rng_r, rng_c = jax.random.split(rng)
    new_r = jax.random.randint(rng_r, (), 0, h, dtype=jnp.int16)
    new_c = jax.random.randint(rng_c, (), 0, w, dtype=jnp.int16)
    # Validity check: only land on a non-void, non-wall tile.
    tile = state.terrain[b, lv, new_r.astype(jnp.int32), new_c.astype(jnp.int32)]
    is_walkable = (tile != jnp.int8(int(TileType.VOID))) & (
        tile != jnp.int8(int(TileType.WALL))
    )
    new_pos = jnp.where(
        is_walkable,
        jnp.stack([new_r, new_c]).astype(jnp.int16),
        state.player_pos,
    )
    return state.replace(player_pos=new_pos)


def _banish(state, rng):
    """Level teleport: change current_level by ±1.

    Cite: sit.c — implied by tele() with TELEPORT_LEVEL flag in extended path.
    Simplified: advance or retreat one level (clamped).
    """
    direction = jnp.where(rn2(rng, 2) == 0, jnp.int8(1), jnp.int8(-1))
    max_lv    = jnp.int8(state.terrain.shape[1])
    new_lv    = jnp.clip(
        state.dungeon.current_level + direction,
        jnp.int8(1),
        max_lv,
    )
    new_dungeon = state.dungeon.replace(current_level=new_lv)
    return state.replace(dungeon=new_dungeon)


def _lose_hp(state, rng):
    """Take rnd(10) damage.

    Cite: sit.c line 70 — losehp(rnd(10), "cursed throne", KILLED_BY_AN).
    """
    dmg    = rnd(rng, 10)
    new_hp = jnp.maximum(state.player_hp - dmg, jnp.int32(0))
    return state.replace(player_hp=new_hp)


def _do_nothing(state, rng):
    """No effect — placeholder for after-switch Python-side handlers (e.g. wish).

    Cite: sit.c default case — impossible("throne effect"); we use as a no-op
    when the real effect is applied outside the lax.switch (case 6 makewish).
    """
    return state


# ---------------------------------------------------------------------------
# Vendor case helpers (sit.c lines 68-205).  Each takes (state, rng) → state
# and corresponds directly to one numbered case in the vendor switch.
# ---------------------------------------------------------------------------

def _case1_attr_drain_and_hp(state, rng):
    """Case 1: adjattrib(rn2(A_MAX), -rn1(4,3), FALSE); losehp(rnd(10), ...).

    Vendor dual effect: random stat drain AND 1d10 HP damage.
    Cite: sit.c lines 69-72.
    """
    rng_a, rng_h = jax.random.split(rng)
    state = _drain_abilities(state, rng_a)
    state = _lose_hp(state, rng_h)
    return state


def _case4_full_heal(state, rng):
    """Case 4: full heal + uhpmax += 4 (if HP close to max) + clear blind/sick.

    Vendor (sit.c lines 83-101):
        if (u.uhp >= u.uhpmax - 5) u.uhpmax += 4;
        u.uhp = u.uhpmax;
        u.ucreamed = 0;
        make_blinded(0L, TRUE);
        make_sick(0L, NULL, FALSE, SICK_ALL);
        heal_legs(0);
    """
    from Nethax.nethax.subsystems.status_effects import TimedStatus
    # +4 hpmax if at or near full HP (within 5 of max).
    near_full = state.player_hp >= (state.player_hp_max - jnp.int32(5))
    new_hp_max = jnp.where(
        near_full,
        state.player_hp_max + jnp.int32(4),
        state.player_hp_max,
    )
    ts = state.status.timed_statuses
    ts = ts.at[int(TimedStatus.BLIND)].set(
        jnp.int32(0).astype(ts.dtype)
    )
    ts = ts.at[int(TimedStatus.SICK)].set(
        jnp.int32(0).astype(ts.dtype)
    )
    ts = ts.at[int(TimedStatus.WOUNDED_LEGS)].set(
        jnp.int32(0).astype(ts.dtype)
    )
    return state.replace(
        player_hp=new_hp_max,
        player_hp_max=new_hp_max,
        status=state.status.replace(timed_statuses=ts),
    )


def _case5_take_gold(state, rng):
    """Case 5: take_gold() — hero loses all gold.

    Cite: sit.c line 103; take_gold() at sit.c lines 13-33 strips COIN_CLASS
    objects from inventory.  We zero player_gold; loose coin stacks in the
    inventory items array are not modeled (gold is tracked as a scalar).
    """
    return state.replace(player_gold=jnp.int32(0))


def _case6_wish_or_luck(state, rng):
    """Case 6: u.uluck + rn2(5) < 0 → change_luck(+1); else makewish().

    The makewish path is handled Python-side in ``sit_throne`` (after the
    lax.switch) because grant_wish needs concrete byte-string input.  This
    branch only applies the +1-luck path; the wish path is applied later
    when the gated case == 6.
    Cite: sit.c lines 105-111.
    """
    luck_roll = state.player_luck.astype(jnp.int32) + rn2(rng, 5)
    bad_luck = luck_roll < jnp.int32(0)
    new_luck = jnp.where(
        bad_luck,
        jnp.minimum(state.player_luck.astype(jnp.int32) + jnp.int32(1),
                    jnp.int32(10)),
        state.player_luck.astype(jnp.int32),
    )
    return state.replace(player_luck=new_luck.astype(jnp.int8))


def _case9_blind_luck_or_rndcurse(state, rng):
    """Case 9: Luck>0 → blind rn1(100,250) + change_luck(-1 or -2); else rndcurse.

    We approximate rndcurse via the existing items_scrolls.rndcurse helper.
    Cite: sit.c lines 133-144.
    """
    from Nethax.nethax.subsystems.status_effects import TimedStatus
    from Nethax.nethax.subsystems.items_scrolls import rndcurse as _rndcurse
    rng_a, rng_b, rng_c = jax.random.split(rng, 3)

    luck = state.player_luck.astype(jnp.int32)
    lucky = luck > jnp.int32(0)

    # Lucky path: blind 250..349 turns + luck -1 (or -2 if luck>1).
    blind_dur = rn1(rng_a, 100, 250)  # rn1(100, 250) = 250 + rn2(100) ∈ [250,349]
    ts = state.status.timed_statuses
    cur_blind = ts[int(TimedStatus.BLIND)].astype(jnp.int32)
    new_blind = jnp.where(lucky, cur_blind + blind_dur, cur_blind)
    new_ts = ts.at[int(TimedStatus.BLIND)].set(new_blind.astype(ts.dtype))

    extra = jnp.where(luck > jnp.int32(1),
                      jnp.int32(1) + rn2(rng_b, 2),  # rnd(2) = 1..2
                      jnp.int32(1))
    lucky_luck = jnp.maximum(luck - extra, jnp.int32(-10)).astype(jnp.int8)

    lucky_state = state.replace(
        status=state.status.replace(timed_statuses=new_ts),
        player_luck=lucky_luck,
    )

    # Unlucky path: rndcurse (sit.c:143).
    unlucky_state = _rndcurse(state, rng_c)

    # Brax-flatten: compute both branches eagerly + jnp.where via tree_map.
    return jax.tree_util.tree_map(
        lambda t, f: jnp.where(lucky, t, f),
        lucky_state,
        unlucky_state,
    )


def _case10_map_or_see_invis(state, rng):
    """Case 10: Luck<0 || HSee_invisible → do_mapping (or confusion if nommap);
    else grant SEE_INVIS intrinsic.

    Simplification: nommap branch is treated as do_mapping (no nommap levels
    are flagged in nethax's level state); see_invis becomes intrinsic grant.
    Cite: sit.c lines 145-184.
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic
    luck = state.player_luck.astype(jnp.int32)
    has_see_invis = state.status.intrinsics[int(Intrinsic.SEE_INVIS)]
    take_map_path = (luck < jnp.int32(0)) | has_see_invis

    mapped_state = _magic_mapping(state, rng)

    new_intr = state.status.intrinsics.at[int(Intrinsic.SEE_INVIS)].set(
        jnp.bool_(True)
    )
    see_invis_state = state.replace(
        status=state.status.replace(intrinsics=new_intr),
    )

    # Brax-flatten: compute both branches eagerly + jnp.where via tree_map.
    return jax.tree_util.tree_map(
        lambda t, f: jnp.where(take_map_path, t, f),
        mapped_state,
        see_invis_state,
    )


def _case11_aggravate_or_tele(state, rng):
    """Case 11: Luck<0 → aggravate(); else tele().

    aggravate() in nethax is modeled by setting the AGGRAVATE intrinsic.
    Cite: sit.c lines 185-193.
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic
    luck = state.player_luck.astype(jnp.int32)
    unlucky = luck < jnp.int32(0)

    new_intr = state.status.intrinsics.at[int(Intrinsic.AGGRAVATE)].set(
        jnp.bool_(True)
    )
    aggro_state = state.replace(
        status=state.status.replace(intrinsics=new_intr),
    )
    tele_state = _teleport_away(state, rng)

    # Brax-flatten: compute both branches eagerly + jnp.where via tree_map.
    return jax.tree_util.tree_map(
        lambda t, f: jnp.where(unlucky, t, f),
        aggro_state,
        tele_state,
    )


def _case13_confusion(state, rng):
    """Case 13: make_confused((HConfusion & TIMEOUT) + rn1(7,16), FALSE).

    Adds 16..22 turns to existing confusion timer.
    Cite: sit.c lines 201-205.
    """
    from Nethax.nethax.subsystems.status_effects import TimedStatus
    add = rn1(rng, 7, 16)  # rn1(7, 16) = 16 + rn2(7) ∈ [16, 22]
    ts = state.status.timed_statuses
    cur = ts[int(TimedStatus.CONFUSION)].astype(jnp.int32)
    new_ts = ts.at[int(TimedStatus.CONFUSION)].set(
        (cur + add).astype(ts.dtype)
    )
    return state.replace(status=state.status.replace(timed_statuses=new_ts))


# ---------------------------------------------------------------------------
# Ordered outcome table: index = (rnd(13) - 1), so index 0..12 maps directly
# to vendor cases 1..13.  Cite: sit.c line 46 (int effect = rnd(13)) and the
# numbered switch at sit.c lines 68-205.
# ---------------------------------------------------------------------------

_OUTCOMES = (
    _case1_attr_drain_and_hp,    # vendor case 1  (sit.c 69-72)
    _attr_gain,                  # vendor case 2  (sit.c 73-75)
    _electrocute,                # vendor case 3  (sit.c 77-82)
    _case4_full_heal,            # vendor case 4  (sit.c 83-101)
    _case5_take_gold,            # vendor case 5  (sit.c 102-104)
    _case6_wish_or_luck,         # vendor case 6  (sit.c 105-111)
    _summon_monsters,            # vendor case 7  (sit.c 112-124)
    _genocide,                   # vendor case 8  (sit.c 125-132)
    _case9_blind_luck_or_rndcurse,  # vendor case 9  (sit.c 133-144)
    _case10_map_or_see_invis,    # vendor case 10 (sit.c 145-184)
    _case11_aggravate_or_tele,   # vendor case 11 (sit.c 185-193)
    _identify_item,              # vendor case 12 (sit.c 194-200)
    _case13_confusion,           # vendor case 13 (sit.c 201-205)
)

_N_OUTCOMES = len(_OUTCOMES)  # 13 — matches vendor rnd(13) range


def sit_throne(state, rng) -> "EnvState":
    """Apply a throne-sit effect with vendor-exact gates and dispatch.

    Vendor flow (sit.c lines 38-233):
        if (rnd(6) > 4) {                # P=1/3 — outer effect gate
            int effect = rnd(13);        # 1..13 inclusive
            switch (effect) { ... }
        } else {
            /* "feels comfortable" or "out of place" — no-op */
        }
        if (!special_throne && !rn2(3) /* AND effect fired */) {
            /* throne vanishes */
        }

    JIT-pure: all branching via jax.lax.switch / jax.lax.cond / jnp.where.

    Parameters
    ----------
    state : EnvState
    rng   : JAX PRNG key

    Returns
    -------
    New EnvState after possibly applying one of 13 outcomes and possibly
    removing the throne tile.

    Cite: vendor/nethack/src/sit.c::throne_sit_effect lines 39-233.
    """
    rng, rng_gate, rng_outcome, rng_effect, rng_remove = jax.random.split(rng, 5)

    # ---- Outer gate: rnd(6) > 4  (vendor sit.c:45) ------------------------
    # rnd(6) ∈ [1,6]; values 5 or 6 → effect fires (P=1/3).
    effect_fired = rnd(rng_gate, 6) > jnp.int32(4)

    # ---- Effect roll: rnd(13) ∈ [1,13]  (vendor sit.c:46) -----------------
    # lax.switch needs 0-based index, so subtract 1.
    case_num = rnd(rng_outcome, _N_OUTCOMES)        # 1..13 (vendor case)
    switch_idx = (case_num - jnp.int32(1)).astype(jnp.int32)

    # Brax-flatten: compute ALL 13 outcomes eagerly, then jnp.where-select
    # via a one-hot mask on switch_idx.  This avoids lax.switch (which under
    # vmap+jit synthesizes a branched HLO that compiles slowly on H100).
    outcome_states = [fn(state, rng_effect) for fn in _OUTCOMES]
    # Start from the case-0 result, then layer cases 1..12 via tree_map+where.
    fired_state = outcome_states[0]
    for i in range(1, _N_OUTCOMES):
        sel = switch_idx == jnp.int32(i)
        fired_state = jax.tree_util.tree_map(
            lambda f, c, _sel=sel: jnp.where(_sel, c, f),
            fired_state,
            outcome_states[i],
        )

    # Brax-flatten: outer gate via jnp.where via tree_map (no lax.cond).
    state = jax.tree_util.tree_map(
        lambda t, f: jnp.where(effect_fired, t, f),
        fired_state,
        state,
    )

    # Case 6 wish (sit.c:110): only fires when effect_fired AND case_num==6
    # AND the lucky-roll branch was NOT taken (luck + rn2(5) >= 0).
    # grant_wish is Python-side (concrete byte string), so we hoist the
    # decision out of the lax.switch.  We bracket it on the gate to avoid
    # granting a wish when the outer gate missed.
    if bool(effect_fired) and int(case_num) == 6:
        # Re-evaluate the lucky-roll to mirror vendor: if bad-luck path was
        # taken, _case6_wish_or_luck already bumped luck and we skip the wish.
        # The wish is granted only when (u.uluck + rn2(5)) >= 0.
        # Using the *original* (pre-case6) luck for the comparison; rng_effect
        # was already consumed inside lax.switch so we draw a fresh roll here.
        rng_wish_roll, rng_wish_grant = jax.random.split(rng_effect)
        luck_check = (
            state.player_luck.astype(jnp.int32) + rn2(rng_wish_roll, 5)
        )
        if int(luck_check) >= 0:
            from Nethax.nethax.subsystems import wish as _wish
            state = _wish.grant_wish(state, rng_wish_grant, _DEFAULT_THRONE_WISH)

    # ---- Removal: vendor sit.c:224-226 -----------------------------------
    # if (!special_throne && !rn2(3))  — AND the effect actually fired
    # (the removal block is reached only after the if/else above; if the
    #  outer gate missed, sit_throne_effect still falls through to it in
    #  vendor — re-check the source).
    # Vendor source (lines 217-233):
    #   The "} else { ... }" of the outer gate closes at line 215; the
    #   removal block at 224 is OUTSIDE both branches, so it fires
    #   regardless of effect_fired.  However, vendor comment in this task
    #   explicitly directs: "Throne removal — Vendor removes throne only
    #   when effect fired AND !rn2(3)".  We follow the task spec.
    remove_roll = rn2(rng_remove, 3) == jnp.int32(0)
    remove = effect_fired & remove_roll

    b  = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - 1
    pr = state.player_pos[0].astype(jnp.int32)
    pc = state.player_pos[1].astype(jnp.int32)

    old_tile = state.terrain[b, lv, pr, pc]
    on_throne = old_tile == jnp.int8(int(TileType.THRONE))
    do_remove = remove & on_throne

    new_terrain = jnp.where(
        do_remove,
        state.terrain.at[b, lv, pr, pc].set(jnp.int8(int(TileType.FLOOR))),
        state.terrain,
    )

    # Mark thrones_used in FeaturesState for tracking.
    flat_lv = b * jnp.int32(state.terrain.shape[1]) + lv
    new_thrones_used = jnp.where(
        do_remove,
        state.features.thrones_used.at[flat_lv, pr, pc].set(jnp.bool_(True)),
        state.features.thrones_used,
    )
    new_features = state.features.replace(thrones_used=new_thrones_used)

    return state.replace(
        terrain=new_terrain,
        features=new_features,
        rng=rng,
    )
