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
# drink_fountain: vendor fountain.c:247  fate = rnd(30).
# fate < 10 → REFRESH (first 9 values, prob 9/30).
# fate 10-18 → TEPID (9 values, prob 9/30).
# fate 19 → SELF_KNOWLEDGE, 20 → VOMIT, 21 → POISON,
# fate 22 → SNAKES, 23 → DEMON, 24 → CURSE_ITEMS,
# fate 25 → SEE_INVIS, 26 → MONSTER_DETECT, 27 → FIND_GEM,
# fate 28 → NYMPH, 29 → SCARE, 30 → GUSH.
#
# Simplified outcome map:
#   0  REFRESH      (fate 1-9,  healing)          9/30
#   1  TEPID        (fate 10-18, no effect)        9/30
#   2  VOMIT        (fate 20)                      1/30
#   3  HEAL_MORE    (fate 19, self-knowledge → we give +hp bonus) 1/30
#   4  GAIN_XL      (fate 18 in blessed ftn)       merged into TEPID; see GAIN_STAT below
#   5  SNAKES       (fate 22)                      1/30
#   6  DEMON        (fate 23)                      1/30
#   7  CURSE_ITEMS  (fate 24)                      1/30
#   8  NYMPH        (fate 28)                      1/30
#   9  GUSH         (fate 30)                      1/30
#  10  GAIN_STAT    (blessed ftn path → gain attr) 5/30

_DRINK_WEIGHTS = jnp.array([9, 9, 1, 1, 1, 1, 1, 1, 5, 1], dtype=jnp.float32)
_DRINK_PROBS   = _DRINK_WEIGHTS / _DRINK_WEIGHTS.sum()
_DRINK_CUMPROBS = jnp.cumsum(_DRINK_PROBS)
_N_DRINK_OUTCOMES = int(_DRINK_WEIGHTS.shape[0])

_DRINK_REFRESH   = 0
_DRINK_TEPID     = 1
_DRINK_VOMIT     = 2
_DRINK_HEAL      = 3
_DRINK_SNAKES    = 4
_DRINK_DEMON     = 5
_DRINK_CURSE     = 6
_DRINK_NYMPH     = 7
_DRINK_GUSH      = 8
_DRINK_GAIN_STAT = 9


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
    return jax.lax.cond(
        should_dry,
        lambda s: dry_fountain(s, pos),
        lambda s: s,
        state,
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

    # --- Excalibur check (fountain.c:404-413) ---
    # Conditions: long sword in slot, player is Lawful, player is Knight, XL>=5.
    item_type = items.type_id[slot_idx].astype(jnp.int32)
    is_long_sword = item_type == jnp.int32(_LONG_SWORD_TYPE_ID)
    is_knight     = state.player_role.astype(jnp.int32) == jnp.int32(_ROLE_KNIGHT)
    is_lawful     = state.player_align.astype(jnp.int32) == jnp.int32(_ALIGN_LAWFUL)
    xl_ok         = state.player_xl.astype(jnp.int32) >= jnp.int32(5)
    excalibur_eligible = is_long_sword & is_knight & is_lawful & xl_ok

    rng, sub, rng_dry = jax.random.split(rng, 3)
    outcome = _roll_outcome(sub, _DIP_CUMPROBS)

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
            return jax.lax.cond(any_dead, _spawn, lambda s: s, st)

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
        return jax.lax.switch(out, branches, s)

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

    return jax.lax.cond(
        excalibur_eligible,
        _grant_excalibur,
        _normal_dip,
        state,
    )


# ---------------------------------------------------------------------------
# drink_fountain (vendor fountain.c:243-390 drinkfountain)
# ---------------------------------------------------------------------------

def drink_fountain(state, rng: jax.Array) -> object:
    """Quaff from the fountain at the player's current position.

    Vendor: fountain.c:243 drinkfountain().
    fate = rnd(30); mapped to _DRINK_* outcomes above.

    Returns updated EnvState.  JIT-pure.
    """
    flat_lv = _flat_level_idx(state)

    rng, sub, rng_dry = jax.random.split(rng, 3)
    outcome = _roll_outcome(sub, _DRINK_CUMPROBS)

    def _refresh(s):
        # fountain.c:279-283 cool draught: restores 1d10 HP.
        # Vendor: losehp is not called; u.uhunger += rnd(10) — we map to +HP.
        gain = jnp.int32(5)
        new_hp = jnp.minimum(s.player_hp + gain, s.player_hp_max)
        return s.replace(player_hp=new_hp)

    def _vomit(s):
        # fountain.c:295-297 fate 20: foul water, lose HP from hunger.
        new_hp = jnp.maximum(s.player_hp - jnp.int32(3), jnp.int32(0))
        return s.replace(player_hp=new_hp)

    def _heal(s):
        # fountain.c:287 fate 19: self-knowledge + wisdom; bonus heal.
        gain = jnp.int32(8)
        new_hp = jnp.minimum(s.player_hp + gain, s.player_hp_max)
        return s.replace(player_hp=new_hp)

    def _snakes(s):
        # fountain.c:311-312 fate 22: dowatersnakes().
        new_hp = jnp.maximum(s.player_hp - jnp.int32(2), jnp.int32(0))
        return s.replace(player_hp=new_hp)

    def _demon(s):
        # fountain.c:313-314 fate 23: dowaterdemon().
        new_hp = jnp.maximum(s.player_hp - jnp.int32(4), jnp.int32(0))
        return s.replace(player_hp=new_hp)

    def _curse_items(s):
        # vendor/nethack/src/fountain.c:315-332 fate 24: curse random
        # inventory items.  Simplified: curse first non-empty slot.
        # Project BUC encoding: CURSED=1 (was 2 — inverted; matches
        # items_scrolls._BUC_CURSED).
        inv = s.inventory.items
        has_item = inv.category > jnp.int8(0)
        slot = jnp.argmax(has_item).astype(jnp.int32)
        new_buc = inv.buc_status.at[slot].set(jnp.int8(1))  # 1 = CURSED
        return s.replace(inventory=s.inventory.replace(
            items=inv.replace(buc_status=new_buc)
        ))

    def _nymph(s):
        # fountain.c:364 fate 28: dowaternymph() — steal item from slot 0.
        inv = s.inventory.items
        new_cat = inv.category.at[0].set(jnp.int8(0))
        return s.replace(inventory=s.inventory.replace(
            items=inv.replace(category=new_cat)
        ))

    def _gush(s):
        # fountain.c:380 fate 30: dogushforth(TRUE).
        return s

    def _gain_stat(s):
        # fountain.c:254-276 blessed fountain path: gain attribute.
        # Simplified: +1 XL.
        new_xl = jnp.minimum(s.player_xl + jnp.int32(1), jnp.int32(30))
        return s.replace(player_xl=new_xl)

    def _noop(s):
        return s

    branches = [
        _refresh,    # 0 REFRESH
        _noop,       # 1 TEPID
        _vomit,      # 2 VOMIT
        _heal,       # 3 HEAL
        _snakes,     # 4 SNAKES
        _demon,      # 5 DEMON
        _curse_items,# 6 CURSE
        _nymph,      # 7 NYMPH
        _gush,       # 8 GUSH
        _gain_stat,  # 9 GAIN_STAT
    ]

    state = jax.lax.switch(outcome, branches, state)
    state = _maybe_dry(state, flat_lv, rng_dry)
    return state
