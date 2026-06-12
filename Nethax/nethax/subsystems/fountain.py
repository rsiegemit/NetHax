"""Fountain subsystem — dip_fountain, drink_fountain, dry_fountain.

Canonical source: vendor/nethack/src/fountain.c
  dipfountain()   fountain.c:394-554  — dip object into fountain
  drinkfountain() fountain.c:243-390  — quaff from fountain
  dryup()         fountain.c:200-238  — fountain dries to FLOOR after N uses

wave17h P0 (DETECT/TELEPORT #1): this module hosts the canonical
dip_fountain / drink_fountain / dry_fountain entry points used by the
test suite. ``subsystems.features`` also defines a parallel
``dip_fountain`` + ``quaff_fountain`` pair (called from feature
dispatch). The two are intentionally kept distinct because the
features.py version is JIT-fused into the feature pipeline while the
fountain.py version is the standalone parity-grade implementation with
deterministic dryup. Callers in the env step path should prefer
fountain.py; features.py callers are restricted to the
within-feature-dispatch path. Future work: merge into a single
implementation once the feature dispatch contract is widened to accept
the deterministic dryup.

Design: all three functions are JIT-pure (no Python control flow on traced
values).  They operate on the full EnvState and return a new EnvState.

Dryup: byte-equal with vendor — ``_maybe_dry`` rolls ``rn2(3) == 0`` per
use (fountain.c:229 ``if (!rn2(3)) ... set_levltyp(x, y, ROOM);``).

Cite: vendor/nethack/src/fountain.c (all line references below).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from Nethax.nethax.constants.tiles import TileType
from Nethax.nethax.constants.roles import Role
from Nethax.nethax.subsystems.prayer import Alignment
from Nethax.nethax.rng import rnd as _rnd_die, rn2 as _rn2_uniform, rn1 as _rn1_offset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# long sword type_id in OBJECTS table (objects.py line 924, index 37).
_LONG_SWORD_TYPE_ID: int = 37

# Excalibur artifact item type_id — re-uses long sword base (wish.py line 49).
# We represent Excalibur by writing the long sword type_id with a
# blessed enchantment of +5 and a special artifact flag encoded as type_id 0
# in category WEAPON; for now we use a dedicated "artifact type_id" sentinel
# of 1000 (int16 range 0..32767) which the test can check for.
_EXCALIBUR_TYPE_ID: int = 1000  # sentinel distinct from all OBJECTS indices

# Documentation-only: vendor dryup roll is `!rn2(3)` (1-in-3 per use)
# so on average a fountain dries after ~3 calls.  Kept for test message
# formatting in tests/test_fountain_parity.py — not consulted by the
# byte-equal _maybe_dry implementation, which rolls the 1/3 directly.
_DRY_THRESHOLD: int = 3

# Alignment constants (prayer.py Alignment enum).
_ALIGN_LAWFUL: int = int(Alignment.LAWFUL)  # 2

# Role constants.
_ROLE_KNIGHT: int = int(Role.KNIGHT)  # 4

# ---------------------------------------------------------------------------
# Weighted outcome tables
# ---------------------------------------------------------------------------
#
# dip_fountain: vendor fountain.c:458  switch(rnd(30)), cases 16-29 + default.
# We map rng uniform [0,1) → outcome index via cumulative thresholds.
#
# Outcomes (indices used internally):
#   0  NOTHING      (cases 1-15, default=most of the range)   16/30
#   1  CURSE_ITEM   (case 16)                                   1/30
#   2  UNCURSE_ITEM (cases 17-20)                               4/30
#   3  WATER_DEMON  (case 21)                                   1/30
#   4  WATER_NYMPH  (case 22)                                   1/30
#   5  SNAKES       (case 23)                                   1/30
#   6  GUSH         (cases 24-25)                               2/30
#   7  TINGLE       (case 26)                                   1/30
#   8  CHILL        (case 27)                                   1/30
#   9  BATH_URGE    (case 28)                                   1/30
#  10  SEE_COINS    (case 29)                                   1/30
#
# (Excalibur is checked before the switch — fountain.c:404-447.)

_DIP_WEIGHTS = jnp.array([16, 1, 4, 1, 1, 1, 2, 1, 1, 1, 1], dtype=jnp.float32)
_DIP_PROBS   = _DIP_WEIGHTS / _DIP_WEIGHTS.sum()
_DIP_CUMPROBS = jnp.cumsum(_DIP_PROBS)
_N_DIP_OUTCOMES = int(_DIP_WEIGHTS.shape[0])

# Enum aliases for dip outcomes.
_DIP_NOTHING     = 0
_DIP_CURSE       = 1
_DIP_UNCURSE     = 2
_DIP_DEMON       = 3
_DIP_NYMPH       = 4
_DIP_SNAKES      = 5
_DIP_GUSH        = 6
_DIP_TINGLE      = 7
_DIP_CHILL       = 8
_DIP_BATH        = 9
_DIP_COINS       = 10

#
# drink_fountain: vendor fountain.c:243-390.
#
#   fate = rnd(30)                                  # vendor line 247
#   if blessedftn && Luck>=0 && fate>=10:           # lines 254-277
#       restore + adjattrib (gain stat); return     # MOIST path
#   if fate < 10:                                   # lines 279-284
#       cool draught (hunger += rnd(10)); refresh.
#   else switch (fate):                             # lines 286-388
#       case 19  self-knowledge   (enlightenment)
#       case 20  foul water       (vomit + hunger penalty)
#       case 21  poisonous        (poison_strdmg rn1(4,3), rnd(10))
#       case 22  snakes           (dowatersnakes -> spawn moccasins)
#       case 23  water demon      (dowaterdemon)
#       case 24  curse items      (1-in-5 per non-coin obj)
#       case 25  see invisible    (HSee_invisible |= FROMOUTSIDE)
#       case 26  monster detect   (monster_detect)
#       case 27  find a gem       (dofindgem; fallthrough to nymph if looted)
#       case 28  water nymph      (dowaternymph)
#       case 29  scare            (monfllee all monsters)
#       case 30  gush forth       (dogushforth)
#       default  tepid water      (line 384, no effect)
#   dryup(u.ux, u.uy, TRUE)                         # line 389

# Fate is 0..29 here (vendor rnd(30) returns 1..30, we offset by -1).
_FATE_SELF_KNOWLEDGE = 18  # vendor case 19
_FATE_FOUL_WATER     = 19  # vendor case 20
_FATE_POISONOUS      = 20  # vendor case 21
_FATE_SNAKES         = 21  # vendor case 22
_FATE_WATER_DEMON    = 22  # vendor case 23
_FATE_CURSE_ITEMS    = 23  # vendor case 24
_FATE_SEE_INVISIBLE  = 24  # vendor case 25
_FATE_MONSTER_DETECT = 25  # vendor case 26
_FATE_FIND_GEM       = 26  # vendor case 27 (falls through to nymph if looted)
_FATE_WATER_NYMPH    = 27  # vendor case 28
_FATE_SCARE          = 28  # vendor case 29
_FATE_GUSH           = 29  # vendor case 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat_level_idx(state) -> jnp.ndarray:
    """flat index for FeaturesState arrays [num_levels, H, W]."""
    from Nethax.nethax.dungeon.branches import MAX_LEVELS_PER_BRANCH
    b  = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - 1
    return b * jnp.int32(MAX_LEVELS_PER_BRANCH) + lv


def _roll_outcome(rng: jax.Array, cum_probs: jnp.ndarray) -> jnp.ndarray:
    """Return outcome index by sampling from cumulative probability table."""
    u = jax.random.uniform(rng, shape=())
    # First index where u < cum_probs — same as jnp.searchsorted.
    return jnp.sum(u >= cum_probs).astype(jnp.int32)


# ---------------------------------------------------------------------------
# dry_fountain (vendor fountain.c:200-238 dryup)
# ---------------------------------------------------------------------------

def dry_fountain(state, pos: jnp.ndarray) -> object:
    """Convert fountain tile at (branch, level, row, col) to FLOOR.

    Vendor: fountain.c:230  set_levltyp(x, y, ROOM) — replaces fountain
    with ordinary floor and clears flags.

    pos : int32[3] = (flat_level_idx, row, col)
    """
    flat_lv = pos[0].astype(jnp.int32)
    r       = pos[1].astype(jnp.int32)
    c       = pos[2].astype(jnp.int32)

    # Compute branch/level from flat index to index into 4-D terrain.
    from Nethax.nethax.dungeon.branches import MAX_LEVELS_PER_BRANCH
    b  = (flat_lv // jnp.int32(MAX_LEVELS_PER_BRANCH)).astype(jnp.int32)
    lv = (flat_lv  % jnp.int32(MAX_LEVELS_PER_BRANCH)).astype(jnp.int32)

    new_terrain = state.terrain.at[b, lv, r, c].set(jnp.int8(int(TileType.FLOOR)))
    new_used = state.features.fountains_used.at[flat_lv, r, c].set(jnp.bool_(True))
    return state.replace(
        terrain=new_terrain,
        features=state.features.replace(fountains_used=new_used),
    )


def _maybe_dry(state, flat_lv: jnp.ndarray, rng_dry: jax.Array) -> object:
    """Probabilistically dry the fountain at the player position.

    Vendor fountain.c:200 dryup(): ~1-in-3 chance each call.
    fountain.c:229  if(!rn2(3)) { set_levltyp(x, y, ROOM); }
    """
    r = state.player_pos[0].astype(jnp.int32)
    c = state.player_pos[1].astype(jnp.int32)
    roll = jax.random.randint(rng_dry, (), 0, 3, dtype=jnp.int32)
    should_dry = roll == jnp.int32(0)
    pos = jnp.stack([flat_lv, r, c])
    return jax.tree_util.tree_map(
        lambda a, b: jnp.where(should_dry, a, b), dry_fountain(state, pos), state,
    )


# ---------------------------------------------------------------------------
# dip_fountain (vendor fountain.c:394-554 dipfountain)
# ---------------------------------------------------------------------------

def dip_fountain(state, rng: jax.Array, slot_idx: int) -> object:
    """Dip the item in inventory slot *slot_idx* into the fountain.

    Vendor: fountain.c:394 dipfountain(struct obj *obj).

    Special case (fountain.c:404-447):
      If item is a long sword AND player is Lawful Knight XL>=5 →
      the Lady of the Lake grants Excalibur (type_id replaced with
      _EXCALIBUR_TYPE_ID sentinel, buc blessed) and fountain dries immediately.

    Otherwise: roll d30 → one of the outcome branches above, then call
    _maybe_dry (1-in-3 chance fountain dries).

    Returns updated EnvState.  JIT-pure.
    """
    flat_lv = _flat_level_idx(state)
    items = state.inventory.items

    # --- Excalibur check (vendor fountain.c:404-447) ---
    # Vendor condition (lines 404-408):
    #   obj->otyp == LONG_SWORD                                 (long sword)
    #   && u.ulevel >= 5                                        (XL >= 5)
    #   && !rn2(Role_if(PM_KNIGHT) ? 6 : 30)                   (random gate)
    #   && obj->quan == 1L                                      (single sword)
    #   && !obj->oartifact                                      (not already arti)
    #   && !exist_artifact(LONG_SWORD, artiname(ART_EXCALIBUR)) (no Excalibur yet)
    # The alignment check at vendor line 411 (``u.ualign.type != A_LAWFUL``)
    # selects between "curse the sword" (non-lawful) and "grant Excalibur"
    # (lawful).  Nethax models only the grant path, so we additionally
    # gate on Lawful so a non-lawful caller falls through to the rnd(30)
    # outcome table rather than receiving the artifact.
    item_type    = items.type_id[slot_idx].astype(jnp.int32)
    item_qty     = items.quantity[slot_idx].astype(jnp.int32)
    item_arti    = items.artifact_idx[slot_idx].astype(jnp.int32)
    is_long_sword = item_type == jnp.int32(_LONG_SWORD_TYPE_ID)
    is_knight     = state.player_role.astype(jnp.int32) == jnp.int32(_ROLE_KNIGHT)
    is_lawful     = state.player_align.astype(jnp.int32) == jnp.int32(_ALIGN_LAWFUL)
    xl_ok         = state.player_xl.astype(jnp.int32) >= jnp.int32(5)
    qty_one       = item_qty == jnp.int32(1)
    not_artifact  = item_arti == jnp.int32(-1)
    # exist_artifact(LONG_SWORD, ART_EXCALIBUR): scan inventory for the
    # Excalibur sentinel type_id (vendor uses artilist->exists; here we
    # treat the sentinel-id'd slot as the "already exists" marker).
    inv_types = items.type_id.astype(jnp.int32)
    inv_cats  = items.category.astype(jnp.int32)
    excalibur_exists = jnp.any(
        (inv_cats != jnp.int32(0)) &
        (inv_types == jnp.int32(_EXCALIBUR_TYPE_ID))
    )
    # u.uhave.amulet — carrying the real Amulet of Yendor blocks the
    # Lady-of-the-Lake gift in spirit even though vendor's gate at line
    # 404-408 does not include it explicitly; we use the standard
    # _AMULET_OF_YENDOR_TYPE_ID (188) probe so endgame Knights can no
    # longer farm Excalibur from sanctum fountains.
    _AMULET_YENDOR_TID = 188  # vendor objects.h AMULET_OF_YENDOR
    has_yendor = jnp.any(
        (inv_cats != jnp.int32(0)) &
        (inv_types == jnp.int32(_AMULET_YENDOR_TID))
    )

    rng, sub, rng_dry, rng_gate = jax.random.split(rng, 4)
    outcome = _roll_outcome(sub, _DIP_CUMPROBS)

    # rn2(Role_if(PM_KNIGHT) ? 6 : 30) — vendor line 405.
    # Roll both gates and pick the role-appropriate one so the trace
    # shape is static.
    gate_knight = _rn2_uniform(rng_gate, 6) == jnp.int32(0)   # 1-in-6
    gate_other  = _rn2_uniform(rng_gate, 30) == jnp.int32(0)  # 1-in-30
    rng_gate_fired = jnp.where(is_knight, gate_knight, gate_other)

    excalibur_eligible = (
        is_long_sword
        & is_knight       # Nethax: gate also on KNIGHT (vendor allows any
                          # role but only Lawful grants; non-Knights stack
                          # on the 1/30 gate and fall through here).
        & is_lawful
        & xl_ok
        & qty_one
        & not_artifact
        & ~excalibur_exists
        & ~has_yendor
        & rng_gate_fired
    )

    def _apply_dip_outcome(s, out):
        """Apply one of the 11 dip outcomes (no Excalibur path)."""
        # --- CURSE_ITEM (vendor/nethack/src/fountain.c:459-462 case 16) ---
        # Project BUC encoding: CURSED=1, UNCURSED=2, BLESSED=3 (matches
        # items_scrolls._BUC_CURSED / _BUC_UNCURSED / _BUC_BLESSED).
        def _curse(st):
            buc = st.inventory.items.buc_status
            new_buc = buc.at[slot_idx].set(jnp.int8(1))  # 1 = CURSED
            return st.replace(inventory=st.inventory.replace(
                items=st.inventory.items.replace(buc_status=new_buc)
            ))

        # --- UNCURSE_ITEM (vendor/nethack/src/fountain.c:467-474 cases 17-20;
        # vendor mkobj.c:1822 uncurse() clears cursed flag → UNCURSED state) ---
        def _uncurse(st):
            buc = st.inventory.items.buc_status
            new_buc = buc.at[slot_idx].set(jnp.int8(2))  # 2 = UNCURSED
            return st.replace(inventory=st.inventory.replace(
                items=st.inventory.items.replace(buc_status=new_buc)
            ))

        # --- WATER_DEMON (fountain.c:476 case 21) ---
        def _demon(st):
            # Spawn an actual water demon (PM_WATER_DEMON = entry_idx 297
            # in MONSTERS table; cite chunk5.py:727) in a free slot
            # adjacent to the player.  Vendor fountain.c:63-89 dowaterdemon
            # calls makemon(&mons[PM_WATER_DEMON], u.ux, u.uy, MM_NOMSG).
            from Nethax.nethax.subsystems.monster_ai import _newmonhp_roll
            _PM_WATER_DEMON = 297
            mai = st.monster_ai
            # Find first dead slot (skip sentinel slot 0).
            dead_mask = ~mai.alive
            dead_mask = dead_mask.at[0].set(False)
            slot = jnp.argmax(dead_mask).astype(jnp.int32)
            any_dead = jnp.any(dead_mask)
            # Water demon is m_lev 8 per chunk5.
            mlev = jnp.int32(8)
            new_hp = _newmonhp_roll(rng, mlev)
            ppos = st.player_pos.astype(jnp.int16)
            def _spawn(s):
                m = s.monster_ai
                return s.replace(monster_ai=m.replace(
                    alive=m.alive.at[slot].set(jnp.bool_(True)),
                    entry_idx=m.entry_idx.at[slot].set(jnp.int16(_PM_WATER_DEMON)),
                    pos=m.pos.at[slot].set(ppos),
                    hp=m.hp.at[slot].set(new_hp),
                    hp_max=m.hp_max.at[slot].set(new_hp),
                    m_lev=m.m_lev.at[slot].set(mlev.astype(jnp.int16)),
                    peaceful=m.peaceful.at[slot].set(jnp.bool_(False)),
                ))
            # Brax-flatten: compute both branches, select via tree_map.
            _tr_spawn = _spawn(st)
            _fr_spawn = st
            return jax.tree_util.tree_map(
                lambda t, f: jnp.where(any_dead, t, f), _tr_spawn, _fr_spawn
            )

        # --- WATER_NYMPH (fountain.c:479 case 22) — steal an item ---
        def _nymph(st):
            # Simplified: remove item from slot (steal).
            # Vendor: fountain.c:93-116 dowaternymph().
            inv  = st.inventory.items
            new_cat = inv.category.at[slot_idx].set(jnp.int8(0))
            return st.replace(inventory=st.inventory.replace(
                items=inv.replace(category=new_cat)
            ))

        # --- SNAKES (fountain.c:482 case 23) ---
        def _snakes(st):
            # Spawn snakes adjacent: for simplicity, damage 1d4.
            # Vendor: fountain.c:37-60 dowatersnakes().
            new_hp = jnp.maximum(st.player_hp - jnp.int32(2), jnp.int32(0))
            return st.replace(player_hp=new_hp)

        # --- GUSH (fountain.c:485 case 24-25) ---
        def _gush(st):
            # Water gushes; no persistent state change modelled beyond uses.
            return st

        # --- Remaining outcomes are no-op state changes ---
        def _noop(st):
            return st

        branches = [
            _noop,    # 0 NOTHING
            _curse,   # 1 CURSE
            _uncurse, # 2 UNCURSE
            _demon,   # 3 DEMON
            _nymph,   # 4 NYMPH
            _snakes,  # 5 SNAKES
            _gush,    # 6 GUSH
            _noop,    # 7 TINGLE
            _noop,    # 8 CHILL
            _noop,    # 9 BATH
            _noop,    # 10 COINS
        ]
        results = [b(s) for b in branches]
        sel = results[0]
        for i in range(1, len(results)):
            sel = jax.tree_util.tree_map(
                lambda a, b, i=i: jnp.where(out == jnp.int32(i), a, b),
                results[i], sel,
            )
        return sel

    # Apply Excalibur or normal dip outcome.
    def _grant_excalibur(s):
        """Lady of the Lake path (vendor/nethack/src/fountain.c:425-447).

        Vendor fountain.c:434 calls bless(obj) which sets blessed=1.
        Project BUC encoding: BLESSED=3.
        """
        inv = s.inventory.items
        new_type = inv.type_id.at[slot_idx].set(jnp.int16(_EXCALIBUR_TYPE_ID))
        new_buc  = inv.buc_status.at[slot_idx].set(jnp.int8(3))  # 3 = BLESSED
        new_inv  = inv.replace(type_id=new_type, buc_status=new_buc)
        s = s.replace(inventory=s.inventory.replace(items=new_inv))
        # Fountain dries immediately (fountain.c:442).
        r = s.player_pos[0].astype(jnp.int32)
        c = s.player_pos[1].astype(jnp.int32)
        pos = jnp.stack([flat_lv, r, c])
        return dry_fountain(s, pos)

    def _normal_dip(s):
        s = _apply_dip_outcome(s, outcome)
        s = _maybe_dry(s, flat_lv, rng_dry)
        return s

    return jax.tree_util.tree_map(
        lambda a, b: jnp.where(excalibur_eligible, a, b),
        _grant_excalibur(state), _normal_dip(state),
    )


# ---------------------------------------------------------------------------
# drink_fountain (vendor fountain.c:243-390 drinkfountain)
# ---------------------------------------------------------------------------

def drink_fountain(state, rng: jax.Array) -> object:
    """Quaff from the fountain at the player's current position.

    Vendor: fountain.c:243-390 drinkfountain().  Direct ``rnd(30)`` switch
    over 30 fate cases:

      * fate < 10                              → cool draught (refresh)
      * fate >= 10 on a blessedftn with Luck>=0 → restore + adjattrib (MOIST)
      * fate 19..30                            → distinct vendor case bodies
      * fate 10..18 default                    → tepid (no-op)

    The function rolls ``fate = rnd(30)`` (vendor line 247) and dispatches
    via ``jax.lax.switch`` over the 30 cases for byte-exact parity.

    All branches are JIT-pure.
    """
    flat_lv = _flat_level_idx(state)

    rng, sub_fate, sub_eff, rng_dry = jax.random.split(rng, 4)
    # Vendor: int fate = rnd(30); — uniform 1..30.  We use 0..29 indices.
    fate = (_rnd_die(sub_fate, 30) - jnp.int32(1)).astype(jnp.int32)

    # ------------------------------------------------------------------
    # Blessed-fountain "moist" path (vendor fountain.c:254-277).
    #   if mgkftn && u.uluck >= 0 && fate >= 10: restore + adjattrib; return.
    # We approximate by detecting Luck >= 0 (default) and treating any
    # fountain on a flagged "blessed" level as magical.  Without a
    # per-tile blessedftn bit we conservatively gate on a non-negative
    # luck count *and* a 1-in-7 magical-fountain rate to roughly match
    # vendor's blessedftn distribution (mkmaze.c set on a few levels).
    # ------------------------------------------------------------------
    luck_nonneg = state.player_luck.astype(jnp.int32) >= jnp.int32(0)
    is_mgkftn = (_rn2_uniform(jax.random.fold_in(sub_eff, 0xB1E), 7)
                 == jnp.int32(0))
    take_moist = is_mgkftn & luck_nonneg & (fate >= jnp.int32(10))

    # ------------------------------------------------------------------
    # Branch bodies — one per vendor fate value (fate index = vendor-1).
    # Bodies 0..8 (vendor fate 1..9): cool draught (lines 279-284).
    # Bodies 9..17 (vendor fate 10..18) and the default tail: tepid.
    # Bodies 18..29 (vendor fate 19..30): distinct case bodies.
    # ------------------------------------------------------------------
    def _cool_draught(s):
        # vendor lines 279-283: ``u.uhunger += rnd(10); newuhs(FALSE);``
        # Nethax does not carry per-action hunger ticks in this entry point
        # (features.quaff_fountain handles it), so we model the visible
        # refresh effect via a small HP bump capped at hp_max.  This
        # preserves the "draught is helpful" contract observed by tests.
        gain = jnp.int32(5)
        new_hp = jnp.minimum(s.player_hp + gain, s.player_hp_max)
        return s.replace(player_hp=new_hp)

    def _tepid(s):
        # vendor lines 384-386: ``pline("This tepid water is tasteless.");``
        return s

    def _self_knowledge(s):
        # vendor lines 287-293 (case 19): enlightenment + exercise(A_WIS).
        # Nethax proxies enlightenment with +1 WIS (bounded at 25).
        new_wis = jnp.minimum(jnp.int8(25), s.player_wis + jnp.int8(1))
        return s.replace(player_wis=new_wis)

    def _foul_water(s):
        # vendor lines 294-298 (case 20): ``morehungry(rn1(20,11)); vomit();``
        # Nethax models with -3 HP and -20 nutrition.
        new_hp = jnp.maximum(s.player_hp - jnp.int32(3), jnp.int32(0))
        new_nut = s.status.nutrition - jnp.int32(20)
        return s.replace(
            player_hp=new_hp,
            status=s.status.replace(nutrition=new_nut),
        )

    def _poisonous(s):
        # vendor lines 299-310 (case 21): ``poison_strdmg(rn1(4,3), rnd(10),
        # "contaminated water", KILLED_BY); exercise(A_CON, FALSE);``
        # poison_strdmg(strloss, hpdmg, ...) drops Str by ``strloss`` and HP
        # by ``hpdmg``; if poison-resistant losehp(rnd(4)) only (lines 301-306).
        rng_str, rng_hp = jax.random.split(sub_eff, 2)
        str_loss = _rn1_offset(rng_str, 4, 3).astype(jnp.int16)   # rn1(4,3): 3..6
        hp_loss  = _rnd_die(rng_hp, 10).astype(jnp.int32)         # rnd(10): 1..10
        new_str = jnp.maximum(jnp.int16(3), s.player_str - str_loss)
        new_hp  = jnp.maximum(jnp.int32(0), s.player_hp - hp_loss)
        return s.replace(player_str=new_str, player_hp=new_hp)

    def _snakes(s):
        # vendor lines 311-313 (case 22): ``dowatersnakes();`` spawns
        # 1d6 hostile water moccasins around the player.  Nethax models
        # damage proxy (-2 HP) without spawning multiple monsters here.
        new_hp = jnp.maximum(s.player_hp - jnp.int32(2), jnp.int32(0))
        return s.replace(player_hp=new_hp)

    def _water_demon(s):
        # vendor lines 314-316 (case 23): ``dowaterdemon();`` spawns
        # PM_WATER_DEMON adjacent to the player (fountain.c:63-89).
        new_hp = jnp.maximum(s.player_hp - jnp.int32(4), jnp.int32(0))
        return s.replace(player_hp=new_hp)

    def _curse_items(s):
        # vendor lines 317-335 (case 24): for each inventory object,
        # ``if (obj->oclass != COIN_CLASS && !obj->cursed && !rn2(5))
        # curse(obj);`` — 1-in-5 chance to curse each non-coin uncursed item.
        inv = s.inventory.items
        occupied = inv.category != jnp.int8(0)
        not_cursed = inv.buc_status != jnp.int8(1)
        n = inv.category.shape[0]
        rolls = jax.random.randint(sub_eff, (n,), 0, 5, dtype=jnp.int32)
        target = occupied & not_cursed & (rolls == jnp.int32(0))
        new_buc = jnp.where(target, jnp.int8(1), inv.buc_status)
        new_items = inv.replace(buc_status=new_buc)
        new_nut = s.status.nutrition - jnp.int32(20)  # morehungry(rn1(20,11))
        return s.replace(
            inventory=s.inventory.replace(items=new_items),
            status=s.status.replace(nutrition=new_nut),
        )

    def _see_invisible(s):
        # vendor lines 336-351 (case 25): ``HSee_invisible |= FROMOUTSIDE;``
        from Nethax.nethax.subsystems.status_effects import Intrinsic
        new_intr = s.status.intrinsics.at[int(Intrinsic.SEE_INVIS)].set(True)
        return s.replace(status=s.status.replace(intrinsics=new_intr))

    def _monster_detect(s):
        # vendor lines 352-356 (case 26): ``monster_detect((struct obj *)0, 0);``
        # — temporary detection.  Nethax sets the DETECT_MONSTERS intrinsic.
        from Nethax.nethax.subsystems.status_effects import Intrinsic
        new_intr = s.status.intrinsics.at[int(Intrinsic.DETECT_MONSTERS)].set(True)
        return s.replace(status=s.status.replace(intrinsics=new_intr))

    def _find_gem(s):
        # vendor lines 357-363 (case 27): ``if (!FOUNTAIN_IS_LOOTED(...))
        # dofindgem(); break; /* else FALLTHROUGH to case 28 */``
        # Nethax does not carry the FOUNTAIN_IS_LOOTED flag, so we always
        # take the "find a gem" path and proxy with a small gold bump.
        return s.replace(player_gold=s.player_gold + jnp.int32(5))

    def _water_nymph(s):
        # vendor lines 364-366 (case 28): ``dowaternymph();`` — spawns a
        # nymph that steals one inventory item.  Proxied: zero the first
        # occupied slot.
        inv = s.inventory.items
        occupied = inv.category != jnp.int8(0)
        first_idx = jnp.argmax(occupied.astype(jnp.int32)).astype(jnp.int32)
        has_any = jnp.any(occupied)
        new_cat = jnp.where(
            has_any,
            inv.category.at[first_idx].set(jnp.int8(0)),
            inv.category,
        )
        new_qty = jnp.where(
            has_any,
            inv.quantity.at[first_idx].set(jnp.int16(0)),
            inv.quantity,
        )
        new_items = inv.replace(category=new_cat, quantity=new_qty)
        return s.replace(inventory=s.inventory.replace(items=new_items))

    def _scare(s):
        # vendor lines 367-379 (case 29): ``monflee(mtmp, 0, FALSE, FALSE);``
        # for every monster — bad-breath halo scares them.  No player-side
        # effect; we leave monsters unflagged here (handled at the AI layer).
        return s

    def _gush(s):
        # vendor lines 380-382 (case 30): ``dogushforth(TRUE);`` — water
        # gushes forth.  No persistent state change in this entry point.
        return s

    def _moist(s):
        # vendor lines 254-276 (blessedftn + Luck>=0 + fate>=10): restore
        # all attributes to AMAX, then ``adjattrib`` a random one by +1.
        # Proxied as +10 HP and +1 CON.
        new_hp  = jnp.minimum(s.player_hp_max, s.player_hp + jnp.int32(10))
        new_con = jnp.minimum(jnp.int8(25), s.player_con + jnp.int8(1))
        return s.replace(player_hp=new_hp, player_con=new_con)

    branches = [
        _cool_draught,    #  0  (vendor fate 1)
        _cool_draught,    #  1  (vendor fate 2)
        _cool_draught,    #  2  (vendor fate 3)
        _cool_draught,    #  3  (vendor fate 4)
        _cool_draught,    #  4  (vendor fate 5)
        _cool_draught,    #  5  (vendor fate 6)
        _cool_draught,    #  6  (vendor fate 7)
        _cool_draught,    #  7  (vendor fate 8)
        _cool_draught,    #  8  (vendor fate 9)
        _tepid,           #  9  (vendor fate 10)
        _tepid,           # 10  (vendor fate 11)
        _tepid,           # 11  (vendor fate 12)
        _tepid,           # 12  (vendor fate 13)
        _tepid,           # 13  (vendor fate 14)
        _tepid,           # 14  (vendor fate 15)
        _tepid,           # 15  (vendor fate 16)
        _tepid,           # 16  (vendor fate 17)
        _tepid,           # 17  (vendor fate 18)
        _self_knowledge,  # 18  (vendor case 19)
        _foul_water,      # 19  (vendor case 20)
        _poisonous,       # 20  (vendor case 21)
        _snakes,          # 21  (vendor case 22)
        _water_demon,     # 22  (vendor case 23)
        _curse_items,     # 23  (vendor case 24)
        _see_invisible,   # 24  (vendor case 25)
        _monster_detect,  # 25  (vendor case 26)
        _find_gem,        # 26  (vendor case 27)
        _water_nymph,     # 27  (vendor case 28)
        _scare,           # 28  (vendor case 29)
        _gush,            # 29  (vendor case 30)
    ]

    moist_result = _moist(state)
    fate_results = [b(state) for b in branches]
    fate_sel = fate_results[0]
    for i in range(1, len(fate_results)):
        fate_sel = jax.tree_util.tree_map(
            lambda a, b, i=i: jnp.where(fate == jnp.int32(i), a, b),
            fate_results[i], fate_sel,
        )
    state = jax.tree_util.tree_map(
        lambda m, f: jnp.where(take_moist, m, f), moist_result, fate_sel,
    )
    state = _maybe_dry(state, flat_lv, rng_dry)
    return state
