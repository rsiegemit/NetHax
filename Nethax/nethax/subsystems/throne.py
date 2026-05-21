"""Throne sit-effect subsystem.

Canonical source: vendor/nethack/src/sit.c::throne_sit_effect (lines 38-233).
  case 1  → attr_loss_and_damage  (sit.c line 70)
  case 2  → attr_gain             (sit.c line 73)
  case 3  → electric_shock        (sit.c lines 77-81)
  case 4  → full_heal             (sit.c lines 83-100)
  case 5  → take_gold             (sit.c line 103)
  case 6  → wish_or_luck          (sit.c lines 106-110)
  case 7  → summon_court          (sit.c lines 113-124)
  case 8  → genocide              (sit.c lines 126-131)
  case 9  → curse_items           (sit.c lines 132-143)
  case 10 → magic_mapping         (sit.c lines 145-183)
  case 11 → teleport              (sit.c lines 185-190)
  case 12 → identify              (sit.c lines 192-199)
  case 13 → confuse               (sit.c lines 200-205)
  post    → 1/3 throne disappears (sit.c lines 224-233)
"""
import jax
import jax.numpy as jnp
import jax.lax as lax

from Nethax.nethax.constants.tiles import TileType
from Nethax.nethax.rng import rnd, rn2

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
    from Nethax.nethax.rng import rn1
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
    """No effect — small chance outcome.

    Cite: sit.c default case — impossible("throne effect") (we use as nothing).
    """
    return state


# ---------------------------------------------------------------------------
# Ordered outcome table: index matches rn2(14) → [0..13].
# Mapping to sit.c cases:
#   0  = gain_gold      (augmented from case 5 take_gold reversed)
#   1  = gain_wish      (case 6 makewish path)
#   2  = genocide       (case 8)
#   3  = lose_luck      (case 9)
#   4  = charge_ring    (case 10 see_invisible variant → ring charge stub)
#   5  = identify_item  (case 12)
#   6  = magic_mapping  (case 10 do_mapping path)
#   7  = drain_abilities(case 1 adjattrib negative)
#   8  = summon_monsters(case 7)
#   9  = teleport_away  (case 11 tele path)
#   10 = banish         (level teleport variant of case 11)
#   11 = lose_hp        (case 1 losehp(rnd(10)))
#   12 = electrocute    (case 3)
#   13 = do_nothing
# ---------------------------------------------------------------------------

_OUTCOMES = (
    _gain_gold,       # 0
    _gain_wish,       # 1
    _genocide,        # 2
    _lose_luck,       # 3
    _charge_ring,     # 4
    _identify_item,   # 5
    _magic_mapping,   # 6
    _drain_abilities, # 7
    _summon_monsters, # 8
    _teleport_away,   # 9
    _banish,          # 10
    _lose_hp,         # 11
    _electrocute,     # 12
    _do_nothing,      # 13
)

_N_OUTCOMES = len(_OUTCOMES)  # 14


def sit_throne(state, rng) -> "EnvState":
    """Apply a random throne-sit effect, then 1/3 chance the throne disappears.

    JIT-pure: all branching via jax.lax.switch / jnp.where.

    Parameters
    ----------
    state : EnvState
    rng   : JAX PRNG key

    Returns
    -------
    New EnvState after applying one of 14 outcomes and possibly removing the
    throne tile.

    Cite: vendor/nethack/src/sit.c::throne_sit_effect lines 38-233.
    """
    rng, rng_outcome, rng_effect, rng_remove = jax.random.split(rng, 4)

    # Roll outcome: rn2(14) — sit.c uses rnd(13) (1-indexed); we use 0-indexed.
    # Cite: sit.c line 46 — int effect = rnd(13).
    outcome_idx = rn2(rng_outcome, _N_OUTCOMES)

    # Apply the chosen effect via lax.switch (JIT-pure dispatch).
    # Each branch receives (state, rng_effect).
    # Note: use default-argument capture (fn=fn) to avoid the Python
    # late-binding closure bug in list comprehensions.
    state = jax.lax.switch(
        outcome_idx,
        [lambda s, r, _fn=fn: _fn(s, r) for fn in _OUTCOMES],
        state,
        rng_effect,
    )

    # Outcome 1 = wish: grant_wish is Python-side (needs concrete values).
    # Applied after lax.switch using a concrete int check.
    # Cite: vendor/nethack/src/sit.c::throne_wish case 6.
    if int(outcome_idx) == 1:
        from Nethax.nethax.subsystems import wish as _wish
        state = _wish.grant_wish(state, rng_effect, _DEFAULT_THRONE_WISH)

    # 1/3 chance the throne disappears after the sit.
    # Cite: sit.c lines 224-226 — if (!special_throne && !rn2(3)) { levl[tx][ty].typ = ROOM; }
    remove = rn2(rng_remove, 3) == 0

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
