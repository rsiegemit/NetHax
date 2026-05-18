"""Depth-curve monster spawning — Wave 3.

Canonical sources:
    vendor/nethack/src/makemon.c::makemon  — monster creation logic
    vendor/nethack/include/permonst.h::monstr[]  — difficulty rating
    vendor/nethack/src/mondata.c  — monster type queries

Wave 3 status:
    MONSTR_DIFFICULTIES: module-level JAX constant (one int per monster).
    eligible_monsters_for_depth: depth-windowed mask excluding G_NOGEN/G_UNIQ.
    pick_monster_for_level: weighted random selection by gen_freq.
    spawn_initial_monsters: roll HP + place on valid floor tiles.
    populate_level_with_monsters: write spawned monsters into EnvState.

Wave 3 simplifications (explicit):
    - MONSTR_DIFFICULTIES uses entry.level as a proxy for difficulty.
      (NetHack's actual monstr[] applies bonus for speed, breath, petrify, etc.
       Wave 5 can refine this.)
    - No group spawning (G_SGROUP, G_LGROUP — Wave 5).
    - No unique placement (G_UNIQ — Wave 5).
    - HP = level × 1d8 (see makemon.c::newmonhp).
    - No terrain-type distinction beyond FLOOR/CORRIDOR walkable check.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from Nethax.nethax.constants.monsters import (
    MONSTERS,
    NUMMONS,
    G_NOGEN,
    G_UNIQ,
    AttackType,
    DamageType,
    MZ_LARGE,
    MZ_HUGE,
    MZ_GIGANTIC,
    M2_MAGIC,
    M2_NASTY,
    M2_GREEDY,
    MS_SOLDIER,
    MS_PRIEST,
    MS_SPELL,
    MS_SELL,
)
from Nethax.nethax.constants.tiles import TileType
from Nethax.nethax.subsystems.monster_ai import (
    MAX_MONSTERS_PER_LEVEL,
    MAX_MONSTER_INV,
    _MONSTER_MRESISTS,
    _MONSTER_UNDEAD,
    _MONSTER_NONLIVING,
)


# ---------------------------------------------------------------------------
# Module-level constants built at import time from the Python MONSTERS tuple
# ---------------------------------------------------------------------------

def _compute_monstr_full(entry) -> int:
    """Full vendor monstr[] formula per makemon.c::monstr_init.

    monstr = level
           + speed_bonus       (move_speed // 8, capped at 5)
           + attk_count        (number of attacks with dice_n > 0)
           + breath_bonus      (+5 if any attack is AT_BREA)
           + petrify_bonus     (+10 if any attack is AD_STON)

    Canonical source: vendor/nethack/src/mondata.c::mstrength
    (Wave 6 Phase B uses the simplified spec formula above; mstrength's
    full ranged/special-damage tweaks are deferred.)
    """
    level = entry.level
    speed_bonus = min(entry.move_speed // 8, 5)
    attacks = entry.attacks or ()
    attk_count = 0
    has_breath = False
    has_petrify = False
    for atk in attacks:
        atyp = atk[0]
        dtyp = atk[1]
        dice_n = atk[2]
        if dice_n > 0:
            attk_count += 1
        if int(atyp) == int(AttackType.AT_BREA):
            has_breath = True
        if int(dtyp) == int(DamageType.AD_STON):
            has_petrify = True
    breath_bonus = 5 if has_breath else 0
    petrify_bonus = 10 if has_petrify else 0
    return int(level + speed_bonus + attk_count + breath_bonus + petrify_bonus)


def _compute_difficulties() -> jnp.ndarray:
    """Build MONSTR_DIFFICULTIES array from MONSTERS at import time.

    Wave 6 closing audit: prefer the vendor-table ``difficulty`` field
    (mons[i].difficulty, from vendor/nle/src/monst.c MON() macro's trailing
    `d` arg — lines 47-50) when populated. Falls back to the
    speed/breath/petrify formula in ``_compute_monstr_full`` when
    ``entry.difficulty == 0`` (uninitialised sentinel).
    """
    diffs = []
    for m in MONSTERS:
        vendor_d = int(getattr(m, "difficulty", 0))
        if vendor_d > 0:
            diffs.append(vendor_d)
        else:
            diffs.append(_compute_monstr_full(m))
    return jnp.array(diffs, dtype=jnp.int32)


def _compute_gen_freqs() -> jnp.ndarray:
    """Extract the low byte of generation_mask as generation frequency weight."""
    freqs = [m.generation_mask & 0xFF for m in MONSTERS]
    return jnp.array(freqs, dtype=jnp.int32)


def _compute_nogen_mask() -> jnp.ndarray:
    """True where monster has G_NOGEN flag (not spawnable via normal generation)."""
    flags = [(m.generation_mask & G_NOGEN) != 0 for m in MONSTERS]
    return jnp.array(flags, dtype=jnp.bool_)


def _compute_uniq_mask() -> jnp.ndarray:
    """True where monster has G_UNIQ flag."""
    flags = [(m.generation_mask & G_UNIQ) != 0 for m in MONSTERS]
    return jnp.array(flags, dtype=jnp.bool_)


def _compute_is_large() -> jnp.ndarray:
    """True where monster size >= MZ_LARGE."""
    flags = [m.size >= MZ_LARGE for m in MONSTERS]
    return jnp.array(flags, dtype=jnp.bool_)


def _compute_base_ac() -> jnp.ndarray:
    """Base AC for each monster type."""
    acs = [m.ac for m in MONSTERS]
    return jnp.array(acs, dtype=jnp.int8)


def _compute_primary_attack_dice() -> tuple[jnp.ndarray, jnp.ndarray]:
    """(n_dice, sides) for the first non-passive attack of each monster."""
    n_arr = []
    s_arr = []
    for m in MONSTERS:
        n, s = 1, 4  # default: 1d4
        for atk in m.attacks:
            if atk[0] != AttackType.AT_NONE and atk[2] > 0:
                n, s = atk[2], atk[3]
                break
        n_arr.append(n)
        s_arr.append(s)
    # Some vendor entries use values >127 (e.g. 255 sentinel); int16 is safe.
    return jnp.array(n_arr, dtype=jnp.int16), jnp.array(s_arr, dtype=jnp.int16)


# Build all constants once at import time.
MONSTR_DIFFICULTIES: jnp.ndarray = _compute_difficulties()   # [NUMMONS] int32
_GEN_FREQS: jnp.ndarray = _compute_gen_freqs()               # [NUMMONS] int32
_IS_NOGEN: jnp.ndarray = _compute_nogen_mask()               # [NUMMONS] bool
_IS_UNIQ: jnp.ndarray = _compute_uniq_mask()                 # [NUMMONS] bool
_IS_LARGE: jnp.ndarray = _compute_is_large()                 # [NUMMONS] bool
_BASE_AC: jnp.ndarray = _compute_base_ac()                   # [NUMMONS] int8
_ATK_DICE_N, _ATK_DICE_S = _compute_primary_attack_dice()    # [NUMMONS] int8 each


# ---------------------------------------------------------------------------
# Wave 6 Mission: spawn-time inventory kits
# ---------------------------------------------------------------------------
# Vendor reference: src/makemon.c::mongets — per-class initial inventory
# drawn from monster's M2_* flags + class-keyed tables (e.g. weapon for
# soldiers, wand+scroll for mages, gold for shopkeepers).
#
# Wave 6 simplification: hard-coded 5 "class kits" indexed by sound/flags2.
# Each kit fills up to MAX_MONSTER_INV slots with (category, type_id,
# quantity, charges) tuples.  See _MONSTER_INV_KITS below.

# ---- Item category / type IDs (mirror subsystems/inventory.ItemCategory
# and subsystems/items_{potions,scrolls,wands}.<Effect>) ------------------
_CAT_NONE   = 0
_CAT_WEAPON = 2
_CAT_ARMOR  = 3
_CAT_AMULET = 5
_CAT_POTION = 8
_CAT_SCROLL = 9
_CAT_SPBOOK = 10
_CAT_WAND   = 11
_CAT_COIN   = 12

_POT_HEALING      = 10
_SCR_TELEPORT     = 10
_WAN_FIRE         = 16
_SPBOOK_FORCEBOLT = 0      # placeholder type_id within SPBOOK category
_LONG_SWORD       = 37     # weapon type_id (matches objects.py "long sword")
_SMALL_SHIELD     = 129    # armor type_id
_AMULET_REFLECT   = 0      # amulet type within AMULET category (placeholder)
_HOLY_WATER       = 25     # PotionEffect.WATER — blessed-water variant

# Kit IDs.
_KIT_NONE    = 0
_KIT_MAGE    = 1   # MS_SPELL spellcaster or M2_MAGIC carrier
_KIT_PRIEST  = 2   # MS_PRIEST (aligned priest, high priest)
_KIT_SOLDIER = 3   # MS_SOLDIER (soldier, sergeant, captain, ...)
_KIT_GOLD    = 4   # MS_SELL (shopkeeper) or M2_GREEDY
_KIT_NASTY   = 5   # M2_NASTY (demons / nasty creatures)

# Per-kit inventory rows: each row is MAX_MONSTER_INV (category, type_id,
# quantity, charges) tuples.  Empty slot has category = 0.
def _build_kit_table() -> tuple:
    """Build per-kit fixed inventory.  Returns (cat, tid, qty, chg) tables
    of shape [N_KITS, MAX_MONSTER_INV].
    """
    n_kits = 6  # _KIT_NONE .. _KIT_NASTY
    cat = [[0] * MAX_MONSTER_INV for _ in range(n_kits)]
    tid = [[0] * MAX_MONSTER_INV for _ in range(n_kits)]
    qty = [[0] * MAX_MONSTER_INV for _ in range(n_kits)]
    chg = [[0] * MAX_MONSTER_INV for _ in range(n_kits)]

    def _set(k, slot, c, t, q, ch=0):
        cat[k][slot] = c
        tid[k][slot] = t
        qty[k][slot] = q
        chg[k][slot] = ch

    # Mage kit: wand of fire (5 charges) + scroll of teleport + potion of healing.
    _set(_KIT_MAGE, 0, _CAT_WAND,   _WAN_FIRE,        1, 5)
    _set(_KIT_MAGE, 1, _CAT_SCROLL, _SCR_TELEPORT,    2)
    _set(_KIT_MAGE, 2, _CAT_POTION, _POT_HEALING,     1)
    _set(_KIT_MAGE, 3, _CAT_SPBOOK, _SPBOOK_FORCEBOLT, 1)

    # Priest kit: holy water + spellbook (heal) + amulet of reflection.
    _set(_KIT_PRIEST, 0, _CAT_POTION, _HOLY_WATER,        2)
    _set(_KIT_PRIEST, 1, _CAT_SPBOOK, _SPBOOK_FORCEBOLT,  1)
    _set(_KIT_PRIEST, 2, _CAT_AMULET, _AMULET_REFLECT,    1)
    _set(_KIT_PRIEST, 3, _CAT_POTION, _POT_HEALING,       1)

    # Soldier kit: long sword + small shield.
    _set(_KIT_SOLDIER, 0, _CAT_WEAPON, _LONG_SWORD,   1)
    _set(_KIT_SOLDIER, 1, _CAT_ARMOR,  _SMALL_SHIELD, 1)

    # Shopkeeper / greedy kit: stack of gold.
    _set(_KIT_GOLD, 0, _CAT_COIN, 0, 100)

    # Nasty (demons): a healing potion + scroll teleport for muse use.
    _set(_KIT_NASTY, 0, _CAT_POTION, _POT_HEALING,  1)
    _set(_KIT_NASTY, 1, _CAT_SCROLL, _SCR_TELEPORT, 1)

    return (
        jnp.array(cat, dtype=jnp.int8),
        jnp.array(tid, dtype=jnp.int16),
        jnp.array(qty, dtype=jnp.int16),
        jnp.array(chg, dtype=jnp.int8),
    )


_KIT_CATS, _KIT_TIDS, _KIT_QTYS, _KIT_CHGS = _build_kit_table()


def _compute_kit_per_entry() -> jnp.ndarray:
    """Map each MONSTERS[i] → kit id.  Priority order:
        1. MS_SELL  or  M2_GREEDY      → _KIT_GOLD
        2. MS_PRIEST                   → _KIT_PRIEST
        3. MS_SPELL or  M2_MAGIC       → _KIT_MAGE
        4. MS_SOLDIER                  → _KIT_SOLDIER
        5. M2_NASTY                    → _KIT_NASTY
        6. else                        → _KIT_NONE
    """
    kits = []
    for m in MONSTERS:
        snd  = int(m.sound)
        f2   = int(m.flags2) & 0xFFFFFFFF
        if snd == int(MS_SELL) or (f2 & (int(M2_GREEDY) & 0xFFFFFFFF)):
            kits.append(_KIT_GOLD)
        elif snd == int(MS_PRIEST):
            kits.append(_KIT_PRIEST)
        elif snd == int(MS_SPELL) or (f2 & (int(M2_MAGIC) & 0xFFFFFFFF)):
            kits.append(_KIT_MAGE)
        elif snd == int(MS_SOLDIER):
            kits.append(_KIT_SOLDIER)
        elif f2 & (int(M2_NASTY) & 0xFFFFFFFF):
            kits.append(_KIT_NASTY)
        else:
            kits.append(_KIT_NONE)
    return jnp.array(kits, dtype=jnp.int8)


_MONSTER_KIT_BY_ENTRY: jnp.ndarray = _compute_kit_per_entry()   # [NUMMONS] int8


# ---------------------------------------------------------------------------
# Eligible-monster mask
# ---------------------------------------------------------------------------

def eligible_monsters_for_depth(depth: int) -> jnp.ndarray:
    """Return a bool mask [NUMMONS] of monsters that can spawn at ``depth``.

    Eligibility criteria (mirrors vendor makemon.c::pm_gen / rndmonst()):
        mon.gen_freq > 0
        AND mon.diff_lvl <= depth + 5     (vendor depth-cap; lower-bound is
                                            the dynamically-rolled "zlevel
                                            window", which on average opens
                                            at depth - 6)
        AND NOT G_NOGEN
        AND NOT G_UNIQ                    (unique placement is handled
                                            separately by m_initweap /
                                            place_special)
    Citation: vendor/nle/src/makemon.c lines 1185-1244 (rndmonst -- where
    ``mons[i].difficulty > zlevel + 4`` rejects the entry); also
    vendor/nle/src/makemon.c::pm_gen (gen_freq weighting).

    Wave 3 note: G_HELL / G_NOHELL filtering is deferred to Wave 5.
    """
    lo = jnp.int32(depth - 6)
    hi = jnp.int32(depth + 5)
    in_window = (MONSTR_DIFFICULTIES >= lo) & (MONSTR_DIFFICULTIES <= hi)
    # mon.gen_freq > 0 -- entries with a zero generation frequency are
    # never produced by rndmonst (vendor makemon.c pm_gen weighting).
    has_freq = _GEN_FREQS > jnp.int32(0)
    eligible = in_window & has_freq & ~_IS_NOGEN & ~_IS_UNIQ
    return eligible


# ---------------------------------------------------------------------------
# Pick one monster type for a given depth
# ---------------------------------------------------------------------------

def pick_monster_for_level(rng: jax.Array, depth: int) -> jnp.ndarray:
    """Sample one monster type index (int32) for the given dungeon depth.

    Vendor reference: ``makemon.c::rndmonst()`` / ``pm_gen()``.  Weights are
    the monster's ``gen_freq`` (vendor: low byte of ``permonst.geno`` -- the
    set_mons_freq value populated by monst.c::G_FREQ).  Eligibility filters
    out G_NOGEN/G_UNIQ entries plus those whose ``diff_lvl > depth + 5``.

    Returns a scalar jnp.int32 in [0, NUMMONS).
    """
    mask = eligible_monsters_for_depth(depth)
    weights = jnp.where(mask, _GEN_FREQS, jnp.int32(0)).astype(jnp.float32)
    # Guard: if all weights zero (very unusual depth), fall back to uniform over eligible.
    total = jnp.sum(weights)
    weights = jnp.where(total > 0, weights, mask.astype(jnp.float32))
    probs = weights / jnp.sum(weights)
    return jax.random.choice(rng, NUMMONS, p=probs).astype(jnp.int32)


# ---------------------------------------------------------------------------
# Peaceful / hostile classification (vendor makemon.c::peace_minded)
# ---------------------------------------------------------------------------

def peace_minded(type_id: int, player_alignment: int, player_align_record: int) -> bool:
    """Return True if a freshly-spawned monster of ``type_id`` should be
    peaceful to the player at this alignment state.

    Vendor formula (vendor/nle/src/makemon.c::peace_minded lines 2003-2042):
      * always_peaceful(ptr) → True
      * always_hostile(ptr)  → False
      * sgn(monster.maligntyp) != sgn(player.alignment) → False
      * else: chance based on u.ualign.record (peaceful when record > 0).

    Wave 6 audit simplification: we use ``MONSTERS[type_id].maligntyp`` for
    the monster side and the supplied player alignment/record arguments.
    Returns a Python bool — this is used at level-construction time only
    (non-JIT path).
    """
    from Nethax.nethax.constants.monsters import MONSTERS

    if int(type_id) < 0 or int(type_id) >= len(MONSTERS):
        return False
    m = MONSTERS[int(type_id)]
    # Vendor field is permonst.maligntyp; our PermonstEntry mirrors it as
    # ``alignment`` (negative = chaotic, 0 = neutral, positive = lawful).
    mal = int(getattr(m, "alignment", 0))
    ual = int(player_alignment)
    # Same-sign alignment → peaceful candidate; differing → hostile.
    if (mal > 0) != (ual > 0):
        return False
    if (mal < 0) != (ual < 0):
        return False
    # If alignment record is non-negative, give the benefit of the doubt.
    return int(player_align_record) >= 0


# ---------------------------------------------------------------------------
# Monster HP roll  (vendor makemon.c::newmonhp -- d(hd, 8))
# ---------------------------------------------------------------------------

def roll_monster_hp(rng: jax.Array, hit_dice: int) -> jnp.ndarray:
    """Roll ``d(hd, 8)`` HP for a newly-created monster.

    Vendor reference: ``makemon.c::newmonhp`` -- "mon->mhp = mon->mhpmax =
    d((int) mon->m_lev, 8);" (1d8 per hit die, sum, min 1).

    This is a public scalar helper for tests; the in-graph spawning code
    uses ``_roll_hp`` which is the same formula but lax.scan-compiled.
    """
    return _roll_hp(rng, jnp.int32(hit_dice))


# ---------------------------------------------------------------------------
# Roll initial HP for a monster (makemon.c::newmonhp)
# ---------------------------------------------------------------------------

def _roll_hp(rng: jax.Array, level: jnp.ndarray) -> jnp.ndarray:
    """Roll HP = level × 1d8 (newmonhp simplified).

    Each die roll is drawn independently; uses scan over a fixed cap of 20
    levels so this is JIT-safe.
    """
    MAX_LEVEL = 20  # static cap; no monster exceeds this in the window used
    level_i32 = jnp.clip(level.astype(jnp.int32), 1, MAX_LEVEL)

    keys = jax.random.split(rng, MAX_LEVEL)

    def _one_die(carry, args):
        die_idx, key = args
        roll = jax.random.randint(key, (), minval=1, maxval=9, dtype=jnp.int32)
        # Only accumulate for die_idx < level
        roll_masked = jnp.where(die_idx < level_i32, roll, jnp.int32(0))
        return carry + roll_masked, None

    total, _ = jax.lax.scan(
        _one_die,
        jnp.int32(0),
        (jnp.arange(MAX_LEVEL, dtype=jnp.int32), keys),
    )
    return jnp.maximum(total, jnp.int32(1))


# ---------------------------------------------------------------------------
# Pick a valid spawn tile (random FLOOR or CORRIDOR, not on stairs/player)
# ---------------------------------------------------------------------------

def _pick_valid_tile(
    rng: jax.Array,
    valid_tiles_mask: jnp.ndarray,
    map_h: int,
    map_w: int,
) -> jnp.ndarray:
    """Return a (row, col) int16 position sampled uniformly from valid_tiles_mask.

    valid_tiles_mask: bool[map_h, map_w] — True where spawning is allowed.

    Falls back to (0, 0) if the mask is entirely False (should not happen
    on a well-formed level, but guards against JIT shape issues).
    """
    flat_mask = valid_tiles_mask.reshape(-1).astype(jnp.float32)
    total = jnp.sum(flat_mask)
    probs = jnp.where(total > 0, flat_mask / total, jnp.ones(map_h * map_w) / (map_h * map_w))
    flat_idx = jax.random.choice(rng, map_h * map_w, p=probs).astype(jnp.int32)
    row = (flat_idx // map_w).astype(jnp.int16)
    col = (flat_idx % map_w).astype(jnp.int16)
    return jnp.stack([row, col])


# ---------------------------------------------------------------------------
# Spawn initial monsters for a level
# ---------------------------------------------------------------------------

def spawn_initial_monsters(
    rng: jax.Array,
    depth: int,
    n_monsters: int,
    valid_tiles_mask: jnp.ndarray,
    map_h: int,
    map_w: int,
) -> tuple:
    """Spawn ``n_monsters`` monsters for dungeon level ``depth``.

    Returns
    -------
    positions  : int16[n_monsters, 2]
    type_ids   : int32[n_monsters]
    hps        : int32[n_monsters]
    max_hps    : int32[n_monsters]
    count      : int32 scalar  (always == n_monsters in Wave 3)

    Uses jax.lax.fori_loop over n_monsters; JIT-compatible.
    """
    # Split rng into per-monster keys: (type_key, hp_key, pos_key) per slot.
    # We pre-split into n_monsters * 3 keys.
    all_keys = jax.random.split(rng, n_monsters * 3)
    type_keys = all_keys[0 * n_monsters : 1 * n_monsters]
    hp_keys   = all_keys[1 * n_monsters : 2 * n_monsters]
    pos_keys  = all_keys[2 * n_monsters : 3 * n_monsters]

    # Pre-sample all type_ids and positions (vectorized approach).
    # fori_loop carry: (positions, type_ids, hps, max_hps)
    init_positions = jnp.zeros((n_monsters, 2), dtype=jnp.int16)
    init_type_ids  = jnp.zeros((n_monsters,),   dtype=jnp.int32)
    init_hps       = jnp.ones((n_monsters,),    dtype=jnp.int32)
    init_max_hps   = jnp.ones((n_monsters,),    dtype=jnp.int32)

    def _spawn_one(i, carry):
        positions, type_ids, hps, max_hps = carry

        type_id = pick_monster_for_level(type_keys[i], depth)
        level = MONSTR_DIFFICULTIES[type_id]
        hp = _roll_hp(hp_keys[i], level)
        pos = _pick_valid_tile(pos_keys[i], valid_tiles_mask, map_h, map_w)

        positions = positions.at[i].set(pos)
        type_ids  = type_ids.at[i].set(type_id)
        hps       = hps.at[i].set(hp)
        max_hps   = max_hps.at[i].set(hp)

        return positions, type_ids, hps, max_hps

    positions, type_ids, hps, max_hps = jax.lax.fori_loop(
        0, n_monsters, _spawn_one, (init_positions, init_type_ids, init_hps, init_max_hps)
    )

    return positions, type_ids, hps, max_hps, jnp.int32(n_monsters)


# ---------------------------------------------------------------------------
# Populate level in EnvState
# ---------------------------------------------------------------------------

def populate_level_with_monsters(
    state,
    rng: jax.Array,
    n_monsters: int = 5,
) -> object:
    """Spawn monsters into state.monster_ai slots [0, n_monsters).

    Reads terrain from state.terrain[branch=0, level=0] (current level).
    Valid spawn tiles: FLOOR or CORRIDOR, not the player's starting position.

    Writes into the first n_monsters slots of state.monster_ai.
    """
    terrain = state.terrain[0, 0]  # int8[MAP_H, MAP_W]
    map_h, map_w = terrain.shape

    # Valid tiles: FLOOR or CORRIDOR
    walkable = (
        (terrain == jnp.int8(TileType.FLOOR)) |
        (terrain == jnp.int8(TileType.CORRIDOR))
    )
    # Exclude player position
    pr, pc = state.player_pos[0].astype(jnp.int32), state.player_pos[1].astype(jnp.int32)
    player_tile_mask = jnp.ones((map_h, map_w), dtype=jnp.bool_).at[pr, pc].set(False)
    valid_tiles_mask = walkable & player_tile_mask

    positions, type_ids, hps, max_hps, count = spawn_initial_monsters(
        rng, depth=1, n_monsters=n_monsters, valid_tiles_mask=valid_tiles_mask,
        map_h=map_h, map_w=map_w,
    )

    mai = state.monster_ai

    # Write slots [0, n_monsters) from spawn results.
    # Use fori_loop to stay JIT-compatible.
    def _write_slot(i, mai_carry):
        type_id = type_ids[i]
        new_pos       = mai_carry.pos.at[i].set(positions[i])
        new_hp        = mai_carry.hp.at[i].set(hps[i])
        new_hp_max    = mai_carry.hp_max.at[i].set(max_hps[i])
        new_alive     = mai_carry.alive.at[i].set(jnp.bool_(True))
        new_ac        = mai_carry.ac.at[i].set(_BASE_AC[type_id])
        new_is_large  = mai_carry.is_large.at[i].set(_IS_LARGE[type_id])
        new_atk_n     = mai_carry.attack_dice_n.at[i].set(_ATK_DICE_N[type_id])
        new_atk_s     = mai_carry.attack_dice_sides.at[i].set(_ATK_DICE_S[type_id])
        new_strategy  = mai_carry.mstrategy.at[i].set(jnp.int8(0))  # NONE until awakened
        new_entry     = mai_carry.entry_idx.at[i].set(type_id.astype(jnp.int16))
        # Per-monster resist/undead/nonliving from MONSTERS table.
        # Cite: vendor/nethack/src/monst.c MON() mr1 field.
        tid           = type_id.astype(jnp.int32)
        new_resists   = mai_carry.resists.at[i].set(
            jnp.take(_MONSTER_MRESISTS, tid, axis=0).astype(jnp.int32))
        new_undead    = mai_carry.undead.at[i].set(
            jnp.take(_MONSTER_UNDEAD, tid, axis=0).astype(jnp.bool_))
        new_nonliving = mai_carry.nonliving.at[i].set(
            jnp.take(_MONSTER_NONLIVING, tid, axis=0).astype(jnp.bool_))

        # Vendor makemon.c::mongets: assign per-class inventory kit.
        kit_id = _MONSTER_KIT_BY_ENTRY[type_id.astype(jnp.int32)].astype(jnp.int32)
        kit_cats = _KIT_CATS[kit_id]   # [MAX_MONSTER_INV] int8
        kit_tids = _KIT_TIDS[kit_id]   # [MAX_MONSTER_INV] int16
        kit_qtys = _KIT_QTYS[kit_id]   # [MAX_MONSTER_INV] int16
        kit_chgs = _KIT_CHGS[kit_id]   # [MAX_MONSTER_INV] int8

        new_invc = mai_carry.inv_category.at[i].set(kit_cats)
        new_invt = mai_carry.inv_type_id.at[i].set(kit_tids)
        new_invq = mai_carry.inv_quantity.at[i].set(kit_qtys)
        new_invch = mai_carry.inv_charges.at[i].set(kit_chgs)

        return mai_carry.replace(
            pos=new_pos,
            hp=new_hp,
            hp_max=new_hp_max,
            alive=new_alive,
            ac=new_ac,
            is_large=new_is_large,
            attack_dice_n=new_atk_n,
            attack_dice_sides=new_atk_s,
            mstrategy=new_strategy,
            entry_idx=new_entry,
            inv_category=new_invc,
            inv_type_id=new_invt,
            inv_quantity=new_invq,
            inv_charges=new_invch,
            resists=new_resists,
            undead=new_undead,
            nonliving=new_nonliving,
        )

    new_mai = jax.lax.fori_loop(0, n_monsters, _write_slot, mai)
    return state.replace(monster_ai=new_mai)
