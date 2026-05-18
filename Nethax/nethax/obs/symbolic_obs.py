"""Symbolic (flat-vector) observation builder for nethax.

Produces a compact fixed-size vector suitable for MLP-based RL baselines.
Encodes structured game state directly without rendering glyphs or ASCII.

Feature layout (total = 2171):
  [0:10]       Player features (10)
  [10:1669]    Map features — glyph idx per tile, 21*79=1659 cells (int16 → float32)
  [1669:1919]  Monster features — MAX_MONSTERS_PER_LEVEL slots * 5 = 50*5=250
               Each slot: (alive, hp, type, distance, peaceful)
  [1919:2139]  Inventory features — 55 slots * 4 = 220
               Each slot: (category, type_id, quantity, identified)
  [2139:2171]  Status features — 32: timed_statuses[:25] + intrinsics[:7] (bitmask)

Returns a flat float32 array of shape (2171,).
JIT-pure.
"""

import jax.numpy as jnp

# ------------------------------------------------------------------
# Symbolic RL agent limits — intentionally smaller than engine maxima
# so that the feature vector stays compact.
# ------------------------------------------------------------------
_SYM_MAX_MONSTERS: int = 50    # first 50 slots of MonsterAIState
_SYM_INV_SLOTS: int = 55       # NLE_INVENTORY_SIZE (a-z, A-Z, + 3 extras)
_SYM_STATUS_FEATURES: int = 32 # timed_statuses[:25] + first 7 intrinsic bits

SYMBOLIC_OBS_DIM: int = (
    10                                    # player
    + 21 * 79                             # map glyphs (1659)
    + _SYM_MAX_MONSTERS * 5               # monsters (250)
    + _SYM_INV_SLOTS * 4                  # inventory (220)
    + _SYM_STATUS_FEATURES                # status (32)
)  # = 2171


def build_symbolic_observation(env_state) -> jnp.ndarray:
    """Build a flat symbolic observation vector from nethax EnvState.

    Args:
        env_state: nethax EnvState.

    Returns:
        jnp.ndarray of shape (SYMBOLIC_OBS_DIM,) float32.
    """
    parts = []

    # ------------------------------------------------------------------
    # 1. Player features (10)
    #    hp, hp_max, pw, pw_max, hunger, ac, level, encumbrance,
    #    alignment, role
    # ------------------------------------------------------------------
    player = jnp.array([
        env_state.player_hp,
        env_state.player_hp_max,
        env_state.player_pw,
        env_state.player_pw_max,
        env_state.status.hunger_state,
        env_state.player_ac,
        env_state.player_xl,
        env_state.status.encumbrance,
        env_state.player_align,
        env_state.player_role,
    ], dtype=jnp.float32)
    parts.append(player)

    # ------------------------------------------------------------------
    # 2. Map features (21*79 = 1659)
    #    Glyph index at each tile of the current level, clipped to int16.
    # ------------------------------------------------------------------
    from Nethax.nethax.obs.nle_obs import build_glyphs
    glyphs = build_glyphs(env_state)          # int16[21, 79]
    map_feats = glyphs.reshape(-1).astype(jnp.float32)   # float32[1659]
    parts.append(map_feats)

    # ------------------------------------------------------------------
    # 3. Monster features (_SYM_MAX_MONSTERS * 5 = 250)
    #    Per slot: (alive, hp, type, distance, peaceful)
    #    distance = Chebyshev distance from player position.
    # ------------------------------------------------------------------
    mai = env_state.monster_ai
    n = _SYM_MAX_MONSTERS

    alive    = mai.alive[:n].astype(jnp.float32)           # [n]
    mon_hp   = mai.hp[:n].astype(jnp.float32)              # [n]
    mon_type = mai.entry_idx[:n].astype(jnp.float32)       # [n]
    peaceful = mai.peaceful[:n].astype(jnp.float32)        # [n]

    pr = jnp.int32(env_state.player_pos[0])
    pc = jnp.int32(env_state.player_pos[1])
    drow = jnp.abs(mai.pos[:n, 0].astype(jnp.int32) - pr)
    dcol = jnp.abs(mai.pos[:n, 1].astype(jnp.int32) - pc)
    dist = jnp.maximum(drow, dcol).astype(jnp.float32)    # Chebyshev

    # Stack in (n, 5) order then flatten.
    mon_feats = jnp.stack([alive, mon_hp, mon_type, dist, peaceful], axis=1)  # [n, 5]
    parts.append(mon_feats.reshape(-1))

    # ------------------------------------------------------------------
    # 4. Inventory features (_SYM_INV_SLOTS * 4 = 220)
    #    Per slot: (category, type_id, quantity, identified)
    #    InventoryState.items has MAX_INVENTORY_SLOTS=52 entries; pad to 55
    #    with zeros to match NLE_INVENTORY_SIZE.
    # ------------------------------------------------------------------
    inv = env_state.inventory.items
    inv_slots = inv.category.shape[0]          # typically 52

    def _pad(arr, target, dtype):
        pad = jnp.zeros((target - inv_slots,), dtype=dtype)
        return jnp.concatenate([arr[:inv_slots], pad])

    cat  = _pad(inv.category,    _SYM_INV_SLOTS, jnp.int8).astype(jnp.float32)
    tid  = _pad(inv.type_id,     _SYM_INV_SLOTS, jnp.int16).astype(jnp.float32)
    qty  = _pad(inv.quantity,    _SYM_INV_SLOTS, jnp.int16).astype(jnp.float32)
    idf  = _pad(inv.identified,  _SYM_INV_SLOTS, jnp.bool_).astype(jnp.float32)

    inv_feats = jnp.stack([cat, tid, qty, idf], axis=1)    # [55, 4]
    parts.append(inv_feats.reshape(-1))

    # ------------------------------------------------------------------
    # 5. Status features (32)
    #    timed_statuses[:25] (turns remaining per condition)
    #    + intrinsics[:7] (permanent intrinsic flags, booleans as float)
    # ------------------------------------------------------------------
    ts = env_state.status.timed_statuses[:25].astype(jnp.float32)   # [25]
    intr = env_state.status.intrinsics[:7].astype(jnp.float32)      # [7]
    parts.append(ts)
    parts.append(intr)

    return jnp.concatenate(parts, axis=0)
