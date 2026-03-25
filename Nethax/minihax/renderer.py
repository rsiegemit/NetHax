"""Symbolic renderer for minihax environments."""
import jax
import jax.numpy as jnp

from Nethax.minihax.constants import NUM_TILE_TYPES, TileType, MONSTER_SYMBOLS
from Nethax.minihax.minihax_state import EnvState, StaticEnvParams


def render_minihax_symbolic(state, static_params):
    """Render game state as a flat symbolic observation vector.

    Encodes:
    1. Map one-hot (fog-of-war masked: unseen=VOID)
    2. Player position (one-hot on map grid)
    3. Monster positions (only on visible tiles)
    4. Visibility channels: seen_map + visible_map
    5. Player stats (normalized)
    """
    map_h = static_params.map_height
    map_w = static_params.map_width

    # Fog of war: unseen tiles appear as VOID, previously seen tiles show normally
    fog_map = jnp.where(state.seen_map, state.map, TileType.VOID)

    # Map one-hot
    tile_one_hot = jax.nn.one_hot(fog_map, NUM_TILE_TYPES)
    tile_flat = tile_one_hot.reshape(-1)

    # Player position
    player_map = jnp.zeros((map_h, map_w), dtype=jnp.float32)
    player_map = player_map.at[state.player_position[0], state.player_position[1]].set(1.0)
    player_flat = player_map.reshape(-1)

    # Monster presence (only on currently visible tiles)
    monster_map = jnp.zeros((map_h, map_w), dtype=jnp.float32)
    max_m = static_params.max_monsters

    def place_monster(m_map, i):
        pos = state.monsters.position[i]
        alive = state.monsters.mask[i]
        visible = state.visible_map[pos[0], pos[1]]
        m_map = m_map.at[pos[0], pos[1]].set(
            jnp.where(alive & visible, 1.0, m_map[pos[0], pos[1]])
        )
        return m_map, None

    monster_map, _ = jax.lax.scan(place_monster, monster_map, jnp.arange(max_m))
    monster_flat = monster_map.reshape(-1)

    # Visibility channels
    seen_flat = state.seen_map.astype(jnp.float32).reshape(-1)
    visible_flat = state.visible_map.astype(jnp.float32).reshape(-1)

    # Player stats
    stats = jnp.array([
        state.player_hp / jnp.maximum(state.player_max_hp, 1),
        state.player_max_hp / 50.0,
        state.player_xp_level / 30.0,
        state.player_ac / 20.0,
        state.player_strength / 25.0,
        state.score / 100.0,
        state.monsters_killed / 16.0,
        state.timestep / 1500.0,
    ], dtype=jnp.float32)

    obs = jnp.concatenate([tile_flat, player_flat, monster_flat, seen_flat, visible_flat, stats])
    return obs


def render_navigation_symbolic(state, static_params):
    """Render Tier 1 navigation state as a flat symbolic observation vector.

    Encodes:
    1. Map one-hot (fog-of-war masked: unseen=VOID)
    2. Player position (one-hot on map grid) [map_h * map_w]
    3. Stair position (one-hot, only if visible) [map_h * map_w]
    4. Ground item presence (visible items only) [map_h * map_w]
    5. Visibility channels: seen_map + visible_map [2 * map_h * map_w]
    6. Stats: [timestep_normalized]

    Args:
        state: NavigationState
        static_params: NavigationStaticParams (map_height, map_width)

    Returns:
        Flat float32 observation vector.
    """
    map_h = static_params.map_height
    map_w = static_params.map_width

    # Fog of war: unseen tiles appear as VOID
    fog_map = jnp.where(state.seen_map, state.map, TileType.VOID)

    # Map one-hot
    tile_one_hot = jax.nn.one_hot(fog_map, NUM_TILE_TYPES)
    map_flat = tile_one_hot.reshape(-1)

    # Player position one-hot
    player_idx = state.player_position[0] * map_w + state.player_position[1]
    player_onehot = jax.nn.one_hot(player_idx, map_h * map_w)

    # Stair position one-hot (only if seen)
    stair_idx = state.downstair_position[0] * map_w + state.downstair_position[1]
    stair_seen = state.seen_map[state.downstair_position[0], state.downstair_position[1]]
    stair_onehot = jnp.where(stair_seen, jax.nn.one_hot(stair_idx, map_h * map_w),
                             jnp.zeros(map_h * map_w, dtype=jnp.float32))

    # Ground item presence (only on visible tiles)
    gi = state.ground_items
    max_gi = static_params.max_ground_items
    item_map = jnp.zeros((map_h, map_w), dtype=jnp.float32)

    def place_item(m, i):
        pos = gi.position[i]
        exists = gi.mask[i]
        vis = state.visible_map[pos[0], pos[1]]
        m = m.at[pos[0], pos[1]].set(
            jnp.where(exists & vis, 1.0, m[pos[0], pos[1]])
        )
        return m, None

    item_map, _ = jax.lax.scan(place_item, item_map, jnp.arange(max_gi))
    item_flat = item_map.reshape(-1)

    # Visibility channels
    seen_flat = state.seen_map.astype(jnp.float32).reshape(-1)
    visible_flat = state.visible_map.astype(jnp.float32).reshape(-1)

    # Stats
    stats = jnp.array([state.timestep / 1500.0], dtype=jnp.float32)

    return jnp.concatenate([map_flat, player_onehot, stair_onehot, item_flat,
                            seen_flat, visible_flat, stats])


def render_minihax_text(state, static_params):
    """Render game state as human-readable ASCII text (not JIT-compatible).

    Applies fog of war:
    - Unseen tiles: space (blank)
    - Previously seen but not visible: lowercase/dimmed representation
    - Currently visible: normal rendering
    - Monsters only shown on visible tiles
    """
    TILE_CHARS = {
        int(TileType.VOID): ' ',
        int(TileType.FLOOR): '.',
        int(TileType.VWALL): '|',
        int(TileType.HWALL): '-',
        int(TileType.TLCORN): '-',
        int(TileType.TRCORN): '-',
        int(TileType.BLCORN): '-',
        int(TileType.BRCORN): '-',
        int(TileType.ALTAR): '_',
        int(TileType.UPSTAIR): '<',
    }

    # Dimmed chars for previously seen but not currently visible tiles
    TILE_CHARS_DIM = {
        int(TileType.VOID): ' ',
        int(TileType.FLOOR): ':',
        int(TileType.VWALL): '|',
        int(TileType.HWALL): '-',
        int(TileType.TLCORN): '-',
        int(TileType.TRCORN): '-',
        int(TileType.BLCORN): '-',
        int(TileType.BRCORN): '-',
        int(TileType.ALTAR): '_',
        int(TileType.UPSTAIR): '<',
    }

    pos = (int(state.player_position[0]), int(state.player_position[1]))

    # Build monster lookup (only on visible tiles)
    monster_positions = {}
    for i in range(static_params.max_monsters):
        if bool(state.monsters.mask[i]):
            r = int(state.monsters.position[i, 0])
            c = int(state.monsters.position[i, 1])
            if bool(state.visible_map[r, c]):
                mon_type = int(state.monsters.type_id[i])
                monster_positions[(r, c)] = MONSTER_SYMBOLS.get(mon_type, '?')

    lines = []
    for r in range(static_params.map_height):
        row = ""
        for c in range(static_params.map_width):
            seen = bool(state.seen_map[r, c])
            visible = bool(state.visible_map[r, c])
            if (r, c) == pos:
                row += "@"
            elif not seen:
                row += " "
            elif (r, c) in monster_positions:
                row += monster_positions[(r, c)]
            elif visible:
                tile = int(state.map[r, c])
                row += TILE_CHARS.get(tile, '?')
            else:
                # Previously seen but not currently visible
                tile = int(state.map[r, c])
                row += TILE_CHARS_DIM.get(tile, '?')
        lines.append(row)

    hp_str = f"HP:{int(state.player_hp)}({int(state.player_max_hp)})"
    score_str = f"S:{int(state.score)}"
    kills_str = f"K:{int(state.monsters_killed)}/16"
    turns_str = f"T:{int(state.timestep)}"
    status_line = f"{hp_str}  {score_str}  {kills_str}  {turns_str}"
    lines.append("")
    lines.append(status_line)

    return "\n".join(lines)


def render_hazard_symbolic(state, static_params):
    """Render Tier 2 hazard state as a flat symbolic observation vector.

    Encodes:
    1. Map one-hot (fog-of-war masked: unseen=VOID)
    2. Player position (one-hot on map grid) [map_h * map_w]
    3. Stair position (one-hot, only if seen) [map_h * map_w]
    4. Monster presence (only on visible tiles) [map_h * map_w]
    5. Visibility channels: seen_map + visible_map [2 * map_h * map_w]
    6. Inventory encoding [max_items] (item_id / NUM_ITEM_TYPES, 0 if empty)
    7. Stats: [hp_norm, max_hp_norm, xp_level, ac, str, dex, con, int, wis, cha,
              energy, max_energy, levitating, timestep_norm]

    Args:
        state: HazardState
        static_params: HazardStaticParams

    Returns:
        Flat float32 observation vector.
    """
    from Nethax.minihax.constants import NUM_ITEM_TYPES

    map_h = static_params.map_height
    map_w = static_params.map_width
    max_m = static_params.max_monsters

    # Fog of war: unseen tiles appear as VOID
    fog_map = jnp.where(state.seen_map, state.map, TileType.VOID)

    # Map one-hot
    tile_one_hot = jax.nn.one_hot(fog_map, NUM_TILE_TYPES)
    map_flat = tile_one_hot.reshape(-1)

    # Player position one-hot
    player_idx = state.player_position[0] * map_w + state.player_position[1]
    player_onehot = jax.nn.one_hot(player_idx, map_h * map_w)

    # Stair position one-hot (only if seen)
    stair_idx = state.downstair_position[0] * map_w + state.downstair_position[1]
    stair_seen = state.seen_map[state.downstair_position[0], state.downstair_position[1]]
    stair_onehot = jnp.where(stair_seen, jax.nn.one_hot(stair_idx, map_h * map_w),
                             jnp.zeros(map_h * map_w, dtype=jnp.float32))

    # Monster presence map (only on currently visible tiles)
    monster_map = jnp.zeros((map_h, map_w), dtype=jnp.float32)

    def place_monster(m_map, i):
        pos = state.monsters.position[i]
        alive = state.monsters.mask[i]
        visible = state.visible_map[pos[0], pos[1]]
        m_map = m_map.at[pos[0], pos[1]].set(
            jnp.where(alive & visible, 1.0, m_map[pos[0], pos[1]])
        )
        return m_map, None

    monster_map, _ = jax.lax.scan(place_monster, monster_map, jnp.arange(max_m))
    monster_flat = monster_map.reshape(-1)

    # Visibility channels
    seen_flat = state.seen_map.astype(jnp.float32).reshape(-1)
    visible_flat = state.visible_map.astype(jnp.float32).reshape(-1)

    # Inventory encoding: normalized item IDs
    inv_encoded = jnp.where(
        state.inventory.item_mask,
        state.inventory.item_ids.astype(jnp.float32) / jnp.float32(NUM_ITEM_TYPES),
        0.0,
    )

    # Stats
    ps = state.player_stats
    stats = jnp.array([
        ps.hp / jnp.maximum(ps.max_hp, 1),
        ps.max_hp / 50.0,
        ps.xp_level / 30.0,
        ps.ac / 20.0,
        ps.strength / 25.0,
        ps.dexterity / 25.0,
        ps.constitution / 25.0,
        ps.intelligence / 25.0,
        ps.wisdom / 25.0,
        ps.charisma / 25.0,
        ps.energy / jnp.maximum(ps.max_energy, 1),
        ps.max_energy / 50.0,
        jnp.float32(state.player_levitating),
        state.timestep / 1500.0,
    ], dtype=jnp.float32)

    return jnp.concatenate([map_flat, player_onehot, stair_onehot, monster_flat,
                            seen_flat, visible_flat, inv_encoded, stats])


def render_combat_symbolic(state, static_params):
    """Render Tier 3 combat state as a flat symbolic observation vector.

    Encodes:
    1. Map one-hot (fog-of-war masked: unseen=VOID)
    2. Player position (one-hot on map grid) [map_h * map_w]
    3. Stair position (one-hot, only if seen) [map_h * map_w]
    4. Monster presence (only on visible tiles) [map_h * map_w]
    5. Trap presence (only on seen tiles) [map_h * map_w]
    6. Visibility channels: seen_map + visible_map [2 * map_h * map_w]
    7. Inventory encoding [max_items] (item_id / NUM_ITEM_TYPES, 0 if empty)
    8. Stats: [hp_norm, max_hp_norm, xp_level, ac, str, dex, con, int, wis, cha,
              energy, max_energy, levitating, has_key, score, kills, timestep_norm]

    Args:
        state: CombatState
        static_params: CombatStaticParams

    Returns:
        Flat float32 observation vector.
    """
    from Nethax.minihax.constants import NUM_ITEM_TYPES

    map_h = static_params.map_height
    map_w = static_params.map_width
    max_m = static_params.max_monsters
    max_traps = static_params.max_traps

    # Fog of war: unseen tiles appear as VOID
    fog_map = jnp.where(state.seen_map, state.map, TileType.VOID)

    # Map one-hot
    tile_one_hot = jax.nn.one_hot(fog_map, NUM_TILE_TYPES)
    map_flat = tile_one_hot.reshape(-1)

    # Player position one-hot
    player_idx = state.player_position[0] * map_w + state.player_position[1]
    player_onehot = jax.nn.one_hot(player_idx, map_h * map_w)

    # Stair position one-hot (only if seen)
    stair_idx = state.downstair_position[0] * map_w + state.downstair_position[1]
    stair_seen = state.seen_map[state.downstair_position[0], state.downstair_position[1]]
    stair_onehot = jnp.where(stair_seen, jax.nn.one_hot(stair_idx, map_h * map_w),
                             jnp.zeros(map_h * map_w, dtype=jnp.float32))

    # Monster presence map (only on currently visible tiles)
    monster_map = jnp.zeros((map_h, map_w), dtype=jnp.float32)

    def place_monster(m_map, i):
        pos = state.monsters.position[i]
        alive = state.monsters.mask[i]
        visible = state.visible_map[pos[0], pos[1]]
        m_map = m_map.at[pos[0], pos[1]].set(
            jnp.where(alive & visible, 1.0, m_map[pos[0], pos[1]])
        )
        return m_map, None

    monster_map, _ = jax.lax.scan(place_monster, monster_map, jnp.arange(max_m))
    monster_flat = monster_map.reshape(-1)

    # Trap presence map (only on seen tiles - traps are remembered once seen)
    trap_map = jnp.zeros((map_h, map_w), dtype=jnp.float32)

    def place_trap(t_map, i):
        pos = state.traps.position[i]
        active = state.traps.mask[i] & jnp.logical_not(state.traps.triggered[i])
        seen = state.seen_map[pos[0], pos[1]]
        t_map = t_map.at[pos[0], pos[1]].set(
            jnp.where(active & seen, 1.0, t_map[pos[0], pos[1]])
        )
        return t_map, None

    trap_map, _ = jax.lax.scan(place_trap, trap_map, jnp.arange(max_traps))
    trap_flat = trap_map.reshape(-1)

    # Visibility channels
    seen_flat = state.seen_map.astype(jnp.float32).reshape(-1)
    visible_flat = state.visible_map.astype(jnp.float32).reshape(-1)

    # Inventory encoding: normalized item IDs
    inv_encoded = jnp.where(
        state.inventory.item_mask,
        state.inventory.item_ids.astype(jnp.float32) / jnp.float32(NUM_ITEM_TYPES),
        0.0,
    )

    # Stats
    ps = state.player_stats
    stats = jnp.array([
        ps.hp / jnp.maximum(ps.max_hp, 1),
        ps.max_hp / 50.0,
        ps.xp_level / 30.0,
        ps.ac / 20.0,
        ps.strength / 25.0,
        ps.dexterity / 25.0,
        ps.constitution / 25.0,
        ps.intelligence / 25.0,
        ps.wisdom / 25.0,
        ps.charisma / 25.0,
        ps.energy / jnp.maximum(ps.max_energy, 1),
        ps.max_energy / 50.0,
        jnp.float32(state.player_levitating),
        jnp.float32(state.player_has_key),
        state.zap_phase / 2.0,
        state.pending_zap_slot / 2.0,
        ps.score / 100.0,
        ps.monsters_killed / 16.0,
        state.timestep / 1500.0,
    ], dtype=jnp.float32)

    return jnp.concatenate([map_flat, player_onehot, stair_onehot, monster_flat,
                            trap_flat, seen_flat, visible_flat, inv_encoded, stats])
