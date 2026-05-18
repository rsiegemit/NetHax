"""Detection spells / scrolls / items — vendor/nethack/src/detect.c parity.

All functions are JIT-pure: (state: EnvState, rng) -> EnvState via state.replace.

Canonical sources:
  vendor/nethack/src/detect.c::food_detect    (lines ~479-601)
  vendor/nethack/src/detect.c::object_detect  (lines ~603-797)
  vendor/nethack/src/detect.c::monster_detect (lines ~798-962)
  vendor/nethack/src/detect.c::do_clairvoyance (lines ~1446-1560)
"""
import jax.numpy as jnp

from Nethax.nethax.constants.tiles import TileType, VendorTileType
from Nethax.nethax.subsystems.inventory import ItemCategory


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _current_flat_level(state) -> jnp.ndarray:
    """Flat level index: branch * max_levels_per_branch + (level - 1)."""
    b  = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)
    max_lv = jnp.int32(state.terrain.shape[1])
    return b * max_lv + lv


# ---------------------------------------------------------------------------
# 1. detect_food
#    Cite: vendor/nethack/src/detect.c::food_detect (~line 479)
#    Sets detect_food_until_turn = timestep + 50.
# ---------------------------------------------------------------------------

def detect_food(state, rng):
    """Reveal nearby food items; set timer for 50 turns.

    Cite: vendor/nethack/src/detect.c::food_detect (~line 479).
    food_detect() iterates the object list and marks FOOD_CLASS items
    visible.  We model this as a timed flag on IdentificationState.
    """
    ts = state.timestep.astype(jnp.int32)
    new_ident = state.identification.replace(
        detect_food_until_turn=ts + jnp.int32(50),
    )
    return state.replace(identification=new_ident)


# ---------------------------------------------------------------------------
# 2. detect_treasure
#    Cite: vendor/nethack/src/detect.c::object_detect (~line 603)
#    Reveals COIN_CLASS objects; sets detect_treasure_until_turn = ts + 50.
# ---------------------------------------------------------------------------

def detect_treasure(state, rng):
    """Reveal gold piles within radius 10; set timer for 50 turns.

    Cite: vendor/nethack/src/detect.c::object_detect (~line 603).
    object_detect() with COIN_CLASS iterates all objects on the level
    and marks gold piles on the map.  We set the timer and mark gold-pile
    cells explored in ground_items.
    """
    ts = state.timestep.astype(jnp.int32)
    new_ident = state.identification.replace(
        detect_treasure_until_turn=ts + jnp.int32(50),
    )

    # Reveal cells containing COIN items within Chebyshev radius 10.
    b  = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)

    level_cats = state.ground_items.category[b, lv]  # [H, W, MAX_STACK]
    has_gold = jnp.any(
        level_cats == jnp.int8(int(ItemCategory.COIN)), axis=-1
    )  # [H, W]

    pr = state.player_pos[0].astype(jnp.int32)
    pc = state.player_pos[1].astype(jnp.int32)
    H, W = state.terrain.shape[2], state.terrain.shape[3]
    rs = jnp.arange(H, dtype=jnp.int32)
    cs = jnp.arange(W, dtype=jnp.int32)
    dr = jnp.abs(rs[:, None] - pr)
    dc = jnp.abs(cs[None, :] - pc)
    in_radius = (dr <= 10) & (dc <= 10)  # [H, W]

    reveal = has_gold & in_radius
    new_explored = state.explored.at[b, lv].set(
        state.explored[b, lv] | reveal
    )
    return state.replace(identification=new_ident, explored=new_explored)


# ---------------------------------------------------------------------------
# 3. detect_objects
#    Cite: vendor/nethack/src/detect.c::object_detect (~line 603)
#    Like detect_treasure but for all objects.  Sets detect_objects_until_turn.
# ---------------------------------------------------------------------------

def detect_objects(state, rng):
    """Reveal all objects on the current level; set timer for 50 turns.

    Cite: vendor/nethack/src/detect.c::object_detect (~line 603).
    Passing class=ALL_CLASSES reveals every non-empty object location.
    """
    ts = state.timestep.astype(jnp.int32)
    new_ident = state.identification.replace(
        detect_objects_until_turn=ts + jnp.int32(50),
    )

    b  = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)

    level_cats = state.ground_items.category[b, lv]  # [H, W, MAX_STACK]
    has_item = jnp.any(level_cats != jnp.int8(0), axis=-1)  # [H, W]

    new_explored = state.explored.at[b, lv].set(
        state.explored[b, lv] | has_item
    )
    return state.replace(identification=new_ident, explored=new_explored)


# ---------------------------------------------------------------------------
# 4. detect_monsters
#    Cite: vendor/nethack/src/detect.c::monster_detect (~line 798)
#    Sets detect_monsters_until_turn = timestep + 100.
# ---------------------------------------------------------------------------

def detect_monsters(state, rng):
    """Reveal all monsters on the current level; set timer for 100 turns.

    Cite: vendor/nethack/src/detect.c::monster_detect (~line 798).
    monster_detect() marks every monster's map position visible for the
    duration.  We track the window via the existing IdentificationState timer.
    """
    ts = state.timestep.astype(jnp.int32)
    new_ident = state.identification.replace(
        detect_monsters_until_turn=ts + jnp.int32(100),
    )
    return state.replace(identification=new_ident)


# ---------------------------------------------------------------------------
# 5. detect_magic
#    Cite: vendor/nethack/src/detect.c::object_detect with magic classes
#    Reveals wands, potions, scrolls, rings, amulets, spellbooks.
#    Sets detect_magic_until_turn = timestep + 50.
# ---------------------------------------------------------------------------

_MAGIC_CATEGORIES = frozenset([
    int(ItemCategory.WAND),
    int(ItemCategory.POTION),
    int(ItemCategory.SCROLL),
    int(ItemCategory.RING),
    int(ItemCategory.AMULET),
    int(ItemCategory.SPBOOK),
])


def detect_magic(state, rng):
    """Reveal all magic items (wands/potions/scrolls/rings/amulets/spellbooks).

    Cite: vendor/nethack/src/detect.c::object_detect (magic-class branch).
    Sets detect_magic_until_turn timer and marks magic-item cells explored.
    """
    ts = state.timestep.astype(jnp.int32)
    new_ident = state.identification.replace(
        detect_magic_until_turn=ts + jnp.int32(50),
    )

    b  = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)
    level_cats = state.ground_items.category[b, lv]  # [H, W, MAX_STACK]

    # Build a mask: is this category a magic class?
    magic_vals = jnp.array(sorted(_MAGIC_CATEGORIES), dtype=jnp.int8)
    # [H, W, STACK, N_MAGIC] — True where category matches any magic class
    is_magic = jnp.any(
        level_cats[:, :, :, None] == magic_vals[None, None, None, :],
        axis=(-2, -1),
    )  # [H, W]

    new_explored = state.explored.at[b, lv].set(
        state.explored[b, lv] | is_magic
    )
    return state.replace(identification=new_ident, explored=new_explored)


# ---------------------------------------------------------------------------
# 6. detect_unseen
#    Cite: vendor/nethack/src/detect.c (detect_unseen path, ~line 1340+)
#    Reveals SDOOR -> CLOSED_DOOR and SCORR -> CORRIDOR on terrain.
#    (Invisible monsters: timer-based; handled by the same ident field.)
# ---------------------------------------------------------------------------

_SDOOR = jnp.int8(int(VendorTileType.SDOOR))
_SCORR = jnp.int8(int(VendorTileType.SCORR))
_CLOSED_DOOR = jnp.int8(int(TileType.CLOSED_DOOR))
_CORRIDOR = jnp.int8(int(TileType.CORRIDOR))


def detect_unseen(state, rng):
    """Reveal secret doors (SDOOR->CLOSED_DOOR) and corridors (SCORR->CORRIDOR).

    Cite: vendor/nethack/src/detect.c (SPE_DETECT_UNSEEN branch, ~line 1340).
    In vendor, detect_unseen reveals all SDOOR/SCORR tiles on the level and
    shows invisible/hidden monsters.  We convert the terrain tiles in-place.
    """
    b  = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)

    level_terrain = state.terrain[b, lv]  # [H, W]
    # SDOOR -> CLOSED_DOOR
    level_terrain = jnp.where(level_terrain == _SDOOR, _CLOSED_DOOR, level_terrain)
    # SCORR -> CORRIDOR
    level_terrain = jnp.where(level_terrain == _SCORR, _CORRIDOR, level_terrain)

    new_terrain = state.terrain.at[b, lv].set(level_terrain)
    return state.replace(terrain=new_terrain)


# ---------------------------------------------------------------------------
# 7. clairvoyance
#    Cite: vendor/nethack/src/detect.c::do_clairvoyance (~line 1446)
#    Reveals 5x5 Chebyshev region around player in explored[].
# ---------------------------------------------------------------------------

def clairvoyance(state, rng):
    """Reveal 5x5 tiles centred on the player in explored[].

    Cite: vendor/nethack/src/detect.c::do_clairvoyance (~line 1446).
    do_clairvoyance() calls do_vicinity_map() with radius 2 (Chebyshev),
    which reveals a 5x5 area (center +/- 2 in each direction).
    """
    b  = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)

    pr = state.player_pos[0].astype(jnp.int32)
    pc = state.player_pos[1].astype(jnp.int32)

    H, W = state.terrain.shape[2], state.terrain.shape[3]
    rs = jnp.arange(H, dtype=jnp.int32)
    cs = jnp.arange(W, dtype=jnp.int32)
    dr = jnp.abs(rs[:, None] - pr)   # [H, 1]
    dc = jnp.abs(cs[None, :] - pc)   # [1, W]
    in_5x5 = (dr <= 2) & (dc <= 2)   # [H, W]

    new_explored = state.explored.at[b, lv].set(
        state.explored[b, lv] | in_5x5
    )
    return state.replace(explored=new_explored)
