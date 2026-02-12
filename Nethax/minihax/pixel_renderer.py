"""Generic pixel rendering functions for minihax environments.

Shared rendering logic for both monster-based and simple navigation/sokoban envs.
"""
import jax
import jax.numpy as jnp
from jax import lax

from Nethax.minihax.constants import (
    TILE_TYPE_SPRITES, MONSTER_SPRITES, ITEM_SPRITES, SPRITE_PLAYER,
    SPRITE_BOULDER, TILE_SIZE, TileType,
)

# Wall auto-tiling: 4-bit index U(8) D(4) L(2) R(1) -> sprite index
# Uses plain Python tuple to avoid JAX tracer leaks from module-level jnp arrays
_WALL_AUTOTILE = (
    851,  # 0000 isolated   -> vwall
    852,  # 0001 R          -> hwall
    852,  # 0010 L          -> hwall
    852,  # 0011 L+R        -> hwall
    851,  # 0100 D          -> vwall
    853,  # 0101 D+R        -> tlcorn
    854,  # 0110 D+L        -> trcorn
    859,  # 0111 D+L+R      -> tdwall
    851,  # 1000 U          -> vwall
    855,  # 1001 U+R        -> blcorn
    856,  # 1010 U+L        -> brcorn
    858,  # 1011 U+L+R      -> tuwall
    851,  # 1100 U+D        -> vwall
    861,  # 1101 U+D+R      -> trwall
    860,  # 1110 U+D+L      -> tlwall
    857,  # 1111 U+D+L+R    -> crwall
)


def _autotile_walls(game_map, tile_indices):
    """Replace wall sprite indices with auto-tiled variants based on neighbor connectivity."""
    # Which cells count as wall-connected (walls + doors)
    gm = game_map
    wc = ((gm == TileType.VWALL) | (gm == TileType.HWALL) |
          (gm == TileType.TLCORN) | (gm == TileType.TRCORN) |
          (gm == TileType.BLCORN) | (gm == TileType.BRCORN) |
          (gm == TileType.DOOR_CLOSED) | (gm == TileType.DOOR_OPEN) |
          (gm == TileType.DOOR_LOCKED))

    # Which cells are walls (auto-tiling targets, not doors)
    is_wall = ((gm == TileType.VWALL) | (gm == TileType.HWALL) |
               (gm == TileType.TLCORN) | (gm == TileType.TRCORN) |
               (gm == TileType.BLCORN) | (gm == TileType.BRCORN))

    # Shifted neighbor arrays (False at boundary = no connection)
    up    = jnp.pad(wc[:-1, :], ((1, 0), (0, 0)))
    down  = jnp.pad(wc[1:, :],  ((0, 1), (0, 0)))
    left  = jnp.pad(wc[:, :-1], ((0, 0), (1, 0)))
    right = jnp.pad(wc[:, 1:],  ((0, 0), (0, 1)))

    # 4-bit connectivity -> lookup sprite
    conn = (up.astype(jnp.int32) * 8 +
            down.astype(jnp.int32) * 4 +
            left.astype(jnp.int32) * 2 +
            right.astype(jnp.int32))

    autotile_lut = jnp.array(_WALL_AUTOTILE, dtype=jnp.int32)
    auto_sprite = autotile_lut[conn]
    return jnp.where(is_wall, auto_sprite, tile_indices)


def _autotile_doors(game_map, tile_indices):
    """Auto-detect door orientation (vertical vs horizontal) based on wall neighbors.

    NetHack uses different sprites for vertical doors (in vertical walls, |) vs
    horizontal doors (in horizontal walls, -).  We detect orientation by checking
    whether the door has walls above+below (vertical) or left+right (horizontal).

    Sprite mapping:
        Vertical closed = 865 (S_vcdoor), Horizontal closed = 866 (S_hcdoor)
        Vertical open   = 863 (S_vodoor), Horizontal open   = 864 (S_hodoor)
    """
    gm = game_map

    is_door_closed = ((gm == TileType.DOOR_CLOSED) | (gm == TileType.DOOR_LOCKED))
    is_door_open = (gm == TileType.DOOR_OPEN)
    is_door = is_door_closed | is_door_open

    # Wall mask for neighbor checks
    is_wall = ((gm == TileType.VWALL) | (gm == TileType.HWALL) |
               (gm == TileType.TLCORN) | (gm == TileType.TRCORN) |
               (gm == TileType.BLCORN) | (gm == TileType.BRCORN))

    up    = jnp.pad(is_wall[:-1, :], ((1, 0), (0, 0)))
    down  = jnp.pad(is_wall[1:, :],  ((0, 1), (0, 0)))
    left  = jnp.pad(is_wall[:, :-1], ((0, 0), (1, 0)))
    right = jnp.pad(is_wall[:, 1:],  ((0, 0), (0, 1)))

    # Walls above AND below -> vertical wall -> vertical door sprite
    # Walls left AND right  -> horizontal wall -> horizontal door sprite
    # Default to vertical if ambiguous
    is_vert = (up & down) | ~(left & right)

    closed_sprite = jnp.where(is_vert, 865, 866)  # vcdoor / hcdoor
    open_sprite   = jnp.where(is_vert, 863, 864)  # vodoor / hodoor

    result = jnp.where(is_door_closed, closed_sprite, tile_indices)
    result = jnp.where(is_door_open, open_sprite, result)
    return result


def _overlay_boulders(game_map, sprites, tiles_array):
    """Overlay boulder object sprites onto floor tiles where map has BOULDER.

    Boulder tiles are rendered as floor in the base terrain pass (TILE_TYPE_SPRITES
    maps BOULDER -> floor 869).  This function alpha-blends the boulder object
    sprite (844) on top so they appear as orange circles on the floor.
    """
    is_boulder = (game_map == TileType.BOULDER)  # [map_h, map_w]

    floor_s   = tiles_array[TILE_TYPE_SPRITES[TileType.FLOOR]].astype(jnp.float32)
    boulder_s = tiles_array[SPRITE_BOULDER].astype(jnp.float32)
    alpha = (jnp.sum(boulder_s, axis=-1, keepdims=True) > 0).astype(jnp.float32)
    boulder_on_floor = (boulder_s * alpha + floor_s * (1.0 - alpha)).astype(jnp.uint8)

    # Replace sprites at boulder positions  [map_h, map_w, 16, 16, 3]
    return jnp.where(is_boulder[:, :, None, None, None],
                     boulder_on_floor[None, None], sprites)


def render_pixels_with_monsters(state, static_params, tiles_array, max_monsters,
                                wizard_mode=False, traps=None):
    """Render pixel obs for envs WITH monsters (Tier 2, 3, ZombieHorde).

    State must have: map, player_position, monsters.position, monsters.mask, monsters.type_id

    Args:
        state: Environment state with map, player_position, monsters
        static_params: Static params with map_height, map_width
        tiles_array: jnp.ndarray [num_tiles, 16, 16, 3] uint8 tile sprites
        max_monsters: Maximum number of monsters to render
        wizard_mode: If True, render traps visibly (static Python bool, not traced)
        traps: Traps struct with position, type_id, mask (required if wizard_mode=True)

    Returns:
        jnp.ndarray [map_h * 16, map_w * 16, 3] uint8 RGB image
    """
    map_h = static_params.map_height
    map_w = static_params.map_width

    # --- Terrain ---
    tile_indices = TILE_TYPE_SPRITES[state.map]        # [map_h, map_w]
    tile_indices = _autotile_walls(state.map, tile_indices)
    tile_indices = _autotile_doors(state.map, tile_indices)
    sprites = tiles_array[tile_indices]                 # [map_h, map_w, 16, 16, 3]
    sprites = _overlay_boulders(state.map, sprites, tiles_array)
    # Reshape to pixel image: [map_h, tile_h, map_w, tile_w, 3] -> [H, W, 3]
    pixels = sprites.transpose(0, 2, 1, 3, 4).reshape(
        map_h * TILE_SIZE, map_w * TILE_SIZE, 3
    )
    # Work in float32 for blending
    pixels = pixels.astype(jnp.float32)

    # --- Monster overlay via lax.scan ---
    def _add_monster(pixels, i):
        pos = state.monsters.position[i]
        alive = state.monsters.mask[i]
        # Only show monster if on a currently visible tile
        mon_visible = state.visible_map[pos[0], pos[1]]
        sprite = tiles_array[MONSTER_SPRITES[state.monsters.type_id[i]]]
        sprite_f = sprite.astype(jnp.float32)

        r = pos[0] * TILE_SIZE
        c = pos[1] * TILE_SIZE

        bg = jax.lax.dynamic_slice(pixels, (r, c, 0), (TILE_SIZE, TILE_SIZE, 3))

        # Alpha: non-black pixels are opaque
        alpha = (jnp.sum(sprite_f, axis=-1, keepdims=True) > 0).astype(jnp.float32)
        blended = sprite_f * alpha + bg * (1.0 - alpha)

        # Only apply if monster is alive AND on a visible tile
        result = jnp.where(alive & mon_visible, blended, bg)
        pixels = jax.lax.dynamic_update_slice(pixels, result, (r, c, 0))
        return pixels, None

    pixels, _ = jax.lax.scan(_add_monster, pixels, jnp.arange(max_monsters))

    # --- Ground items overlay ---
    if hasattr(state, 'ground_items') and state.ground_items is not None:
        max_gi = state.ground_items.position.shape[0]

        def _add_ground_item(pixels, i):
            pos = state.ground_items.position[i]
            exists = state.ground_items.mask[i]
            item_visible = state.visible_map[pos[0], pos[1]]
            sprite_idx = ITEM_SPRITES[state.ground_items.type_id[i]]
            sprite_f = tiles_array[sprite_idx].astype(jnp.float32)

            r = pos[0] * TILE_SIZE
            c = pos[1] * TILE_SIZE

            bg = jax.lax.dynamic_slice(pixels, (r, c, 0), (TILE_SIZE, TILE_SIZE, 3))
            alpha = (jnp.sum(sprite_f, axis=-1, keepdims=True) > 0).astype(jnp.float32)
            blended = sprite_f * alpha + bg * (1.0 - alpha)

            result = jnp.where(exists & item_visible, blended, bg)
            pixels = jax.lax.dynamic_update_slice(pixels, result, (r, c, 0))
            return pixels, None

        pixels, _ = jax.lax.scan(_add_ground_item, pixels, jnp.arange(max_gi))

    # --- Trap overlay (wizard mode only) ---
    if wizard_mode and traps is not None:
        # Map trap type_id to tile sprite: 1=TRAP_BOARD, 2=PIT
        trap_sprite_map = jnp.array([
            TILE_TYPE_SPRITES[0],              # 0 = none (unused)
            TILE_TYPE_SPRITES[TileType.TRAP_BOARD],  # 1 = board trap
            TILE_TYPE_SPRITES[TileType.PIT],          # 2 = pit trap
        ], dtype=jnp.int32)

        def _add_trap(pixels, i):
            pos = traps.position[i]
            active = traps.mask[i]
            sprite_idx = trap_sprite_map[traps.type_id[i]]
            sprite = tiles_array[sprite_idx].astype(jnp.float32)

            r = pos[0] * TILE_SIZE
            c = pos[1] * TILE_SIZE

            bg = jax.lax.dynamic_slice(pixels, (r, c, 0), (TILE_SIZE, TILE_SIZE, 3))
            alpha = (jnp.sum(sprite, axis=-1, keepdims=True) > 0).astype(jnp.float32)
            blended = sprite * alpha + bg * (1.0 - alpha)

            result = jnp.where(active, blended, bg)
            pixels = jax.lax.dynamic_update_slice(pixels, result, (r, c, 0))
            return pixels, None

        max_traps = traps.position.shape[0]
        pixels, _ = jax.lax.scan(_add_trap, pixels, jnp.arange(max_traps))

    # --- Player overlay ---
    # Use floor tile as background so staircase/altar don't show through player
    floor_sprite = tiles_array[TILE_TYPE_SPRITES[1]].astype(jnp.float32)  # FLOOR
    player_sprite = tiles_array[SPRITE_PLAYER].astype(jnp.float32)
    pr = state.player_position[0] * TILE_SIZE
    pc = state.player_position[1] * TILE_SIZE
    player_alpha = (jnp.sum(player_sprite, axis=-1, keepdims=True) > 0).astype(jnp.float32)
    player_blended = player_sprite * player_alpha + floor_sprite * (1.0 - player_alpha)
    pixels = jax.lax.dynamic_update_slice(pixels, player_blended, (pr, pc, 0))

    # --- Fog of war ---
    # Upscale visibility maps to pixel resolution
    visible_px = state.visible_map.astype(jnp.float32)
    visible_px = jnp.repeat(visible_px, TILE_SIZE, axis=0)
    visible_px = jnp.repeat(visible_px, TILE_SIZE, axis=1)
    visible_px = visible_px[:, :, None]

    seen_px = state.seen_map.astype(jnp.float32)
    seen_px = jnp.repeat(seen_px, TILE_SIZE, axis=0)
    seen_px = jnp.repeat(seen_px, TILE_SIZE, axis=1)
    seen_px = seen_px[:, :, None]

    # visible=1.0, seen-but-not-visible=0.3, unseen=0.0
    brightness = jnp.where(visible_px > 0, 1.0,
                 jnp.where(seen_px > 0, 0.3, 0.0))
    pixels = pixels * brightness

    return pixels.astype(jnp.uint8)


def render_pixels_no_monsters(state, static_params, tiles_array):
    """Render pixel obs for envs WITHOUT monsters (Tier 1 Navigation, Tier 4 Sokoban).

    State must have: map, player_position.
    Optionally renders ground_items if present (ExploreMaze apples).

    Args:
        state: Environment state with map, player_position
        static_params: Static params with map_height, map_width
        tiles_array: jnp.ndarray [num_tiles, 16, 16, 3] uint8 tile sprites

    Returns:
        jnp.ndarray [map_h * 16, map_w * 16, 3] uint8 RGB image
    """
    map_h = static_params.map_height
    map_w = static_params.map_width

    # --- Terrain ---
    tile_indices = TILE_TYPE_SPRITES[state.map]        # [map_h, map_w]
    tile_indices = _autotile_walls(state.map, tile_indices)
    tile_indices = _autotile_doors(state.map, tile_indices)
    sprites = tiles_array[tile_indices]                 # [map_h, map_w, 16, 16, 3]
    sprites = _overlay_boulders(state.map, sprites, tiles_array)
    # Reshape to pixel image: [map_h, tile_h, map_w, tile_w, 3] -> [H, W, 3]
    pixels = sprites.transpose(0, 2, 1, 3, 4).reshape(
        map_h * TILE_SIZE, map_w * TILE_SIZE, 3
    )
    # Work in float32 for blending
    pixels = pixels.astype(jnp.float32)

    # --- Ground items overlay (ExploreMaze apples) ---
    if hasattr(state, 'ground_items') and state.ground_items is not None:
        max_gi = state.ground_items.position.shape[0]
        if max_gi > 0:
            def _add_ground_item(pixels, i):
                pos = state.ground_items.position[i]
                exists = state.ground_items.mask[i]
                item_visible = state.visible_map[pos[0], pos[1]]
                sprite_idx = ITEM_SPRITES[state.ground_items.type_id[i]]
                sprite_f = tiles_array[sprite_idx].astype(jnp.float32)

                r = pos[0] * TILE_SIZE
                c = pos[1] * TILE_SIZE

                bg = jax.lax.dynamic_slice(pixels, (r, c, 0), (TILE_SIZE, TILE_SIZE, 3))
                alpha = (jnp.sum(sprite_f, axis=-1, keepdims=True) > 0).astype(jnp.float32)
                blended = sprite_f * alpha + bg * (1.0 - alpha)

                result = jnp.where(exists & item_visible, blended, bg)
                pixels = jax.lax.dynamic_update_slice(pixels, result, (r, c, 0))
                return pixels, None

            pixels, _ = jax.lax.scan(_add_ground_item, pixels, jnp.arange(max_gi))

    # --- Player overlay ---
    # Use floor tile as background so staircase/altar don't show through player
    floor_sprite = tiles_array[TILE_TYPE_SPRITES[1]].astype(jnp.float32)  # FLOOR
    player_sprite = tiles_array[SPRITE_PLAYER].astype(jnp.float32)
    pr = state.player_position[0] * TILE_SIZE
    pc = state.player_position[1] * TILE_SIZE
    player_alpha = (jnp.sum(player_sprite, axis=-1, keepdims=True) > 0).astype(jnp.float32)
    player_blended = player_sprite * player_alpha + floor_sprite * (1.0 - player_alpha)
    pixels = jax.lax.dynamic_update_slice(pixels, player_blended, (pr, pc, 0))

    # --- Fog of war ---
    # Upscale visibility maps to pixel resolution
    visible_px = state.visible_map.astype(jnp.float32)
    visible_px = jnp.repeat(visible_px, TILE_SIZE, axis=0)
    visible_px = jnp.repeat(visible_px, TILE_SIZE, axis=1)
    visible_px = visible_px[:, :, None]

    seen_px = state.seen_map.astype(jnp.float32)
    seen_px = jnp.repeat(seen_px, TILE_SIZE, axis=0)
    seen_px = jnp.repeat(seen_px, TILE_SIZE, axis=1)
    seen_px = seen_px[:, :, None]

    # visible=1.0, seen-but-not-visible=0.3, unseen=0.0
    brightness = jnp.where(visible_px > 0, 1.0,
                 jnp.where(seen_px > 0, 0.3, 0.0))
    pixels = pixels * brightness

    return pixels.astype(jnp.uint8)
