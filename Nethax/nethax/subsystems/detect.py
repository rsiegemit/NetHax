"""Detection spells / scrolls / items — vendor/nethack/src/detect.c parity.

All functions are JIT-pure: (state: EnvState, rng) -> EnvState via state.replace.

Canonical sources:
  vendor/nethack/src/detect.c::gold_detect    (lines 335-475)
  vendor/nethack/src/detect.c::food_detect    (lines 479-594)
  vendor/nethack/src/detect.c::object_detect  (lines 603-789)
  vendor/nethack/src/detect.c::monster_detect (lines 798-862)
  vendor/nethack/src/detect.c::trap_detect    (lines 1011-1088)
  vendor/nethack/src/detect.c::do_vicinity_map (lines 1448-1585)
  vendor/nethack/src/detect.c::cvt_sdoor_to_door (lines 1589-1604)
  vendor/nethack/src/teleport.c::safe_teleds  (lines 716-770)

Line-of-sight helper:
  Callers that need a vendor-parity LoS check (``cansee`` / ``couldsee`` /
  ``clear_path`` from vendor vision.h/vision.c) should import the helpers
  exposed in ``Nethax.nethax.subsystems.vision``.  The detect routines below
  all reveal the entire current level (matching vendor's level-wide
  ``map_object`` / ``map_monst`` sweeps with NO distance limit), so they do
  not perform per-tile LoS gating themselves.
"""
import jax
import jax.numpy as jnp

from Nethax.nethax.constants.tiles import TileType, VendorTileType
from Nethax.nethax.subsystems.inventory import ItemCategory
# Re-export the LoS primitives so that callers preferring the historical
# detect.cansee / detect.couldsee names see them surfaced from this module.
from Nethax.nethax.subsystems.vision import (  # noqa: F401  (public re-export)
    cansee,
    cansee_with_blind,
    clear_path,
    couldsee,
)


# ---------------------------------------------------------------------------
# wave17h P0 (DETECT/TELEPORT #2): unified _teleds helper.
# Cite: vendor/nethack/src/teleport.c::safe_teleds (lines 716-770),
#       vendor/nethack/src/teleport.c::teleds       (lines 448-...),
#       vendor/nethack/src/teleport.c::scrolltele   (line 914 — calls safe_teleds).
#
# The helper rejection-samples a walkable destination on the current level
# and writes player_pos. Vendor safe_teleds tries 40 random uniform draws
# (lines 736-743) accepting any tile that passes teleok().  We approximate
# teleok by accepting FLOOR / CORRIDOR / OPEN_DOOR (the JAX TileTypes that
# correspond to vendor's walkable non-trap goodpos tiles).  Used by
# potion/scroll/wand teleport so the three subsystems do not diverge.
# ---------------------------------------------------------------------------

def _teleds(state, rng):
    """Teleport the player to a uniformly-sampled walkable tile.

    Cite: vendor/nethack/src/teleport.c::safe_teleds (lines 716-770).
    Vendor loops `for (tcnt = 0; tcnt < 40; ++tcnt)` (line 736), each draw
    via `rnd(COLNO-1)` / `rn2(ROWNO)` (lines 737-738), accepting via
    teleok() (lines 420-445).  JIT-pure: uses bounded rejection (40 tries)
    via lax.scan to avoid lax.while_loop.
    """
    b  = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)
    terrain_2d = state.terrain[b, lv]
    h, w = terrain_2d.shape

    MAX_TRIES = 40
    rng_r, rng_c = jax.random.split(rng, 2)
    rrows = jax.random.randint(rng_r, (MAX_TRIES,), 0, h, dtype=jnp.int32)
    rcols = jax.random.randint(rng_c, (MAX_TRIES,), 0, w, dtype=jnp.int32)

    def _walkable(r, c):
        t = terrain_2d[r, c]
        return (
            (t == jnp.int8(TileType.FLOOR))
            | (t == jnp.int8(TileType.CORRIDOR))
            | (t == jnp.int8(TileType.OPEN_DOOR))
        )

    def _pick(carry, i):
        chosen_r, chosen_c, found = carry
        r = rrows[i].astype(jnp.int32); c = rcols[i].astype(jnp.int32)
        ok = _walkable(r, c) & ~found
        new_r = jnp.where(ok, r, chosen_r).astype(jnp.int32)
        new_c = jnp.where(ok, c, chosen_c).astype(jnp.int32)
        new_found = found | _walkable(r, c)
        return (new_r, new_c, new_found), None

    (final_r, final_c, _), _ = jax.lax.scan(
        _pick,
        (jnp.int32(state.player_pos[0]), jnp.int32(state.player_pos[1]), jnp.bool_(False)),
        jnp.arange(MAX_TRIES),
    )
    new_pos = jnp.array([final_r, final_c], dtype=jnp.int16)
    return state.replace(player_pos=new_pos)


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
#    Cite: vendor/nethack/src/detect.c::food_detect (lines 479-594).
#    Vendor iterates fobj (line 492) and every monster's minvent
#    (lines 499-510) with `o_in(obj, FOOD_CLASS)` (or POTION_CLASS when
#    confused/cursed at line 485); on success calls `map_object(temp, 1)`
#    for each match (lines 555-572) — level-wide, no distance limit.
# ---------------------------------------------------------------------------

def detect_food(state, rng):
    """Reveal nearby food items; set timer for 50 turns.

    Cite: vendor/nethack/src/detect.c::food_detect (lines 479-594).
    Vendor reveals every FOOD_CLASS object location on the current level
    via `map_object(temp, 1)` (line 561). We model this as a timed flag
    on IdentificationState.
    """
    ts = state.timestep.astype(jnp.int32)
    new_ident = state.identification.replace(
        detect_food_until_turn=ts + jnp.int32(50),
    )
    return state.replace(identification=new_ident)


# ---------------------------------------------------------------------------
# 2. detect_treasure (gold detection)
#    Cite: vendor/nethack/src/detect.c::gold_detect (lines 335-475).
#    Vendor iterates *every* object on the level (lines 372-382:
#       for (obj = fobj; obj; obj = obj->nobj) ... map_object(...))
#    and every monster's inventory (lines 347-369, 435-463) — there is
#    NO distance limit.  The whole level is revealed.
#    Sets detect_treasure_until_turn = ts + 50 as the JAX model window.
# ---------------------------------------------------------------------------

def detect_treasure(state, rng):
    """Reveal all gold piles on the current level; set timer for 50 turns.

    Cite: vendor/nethack/src/detect.c::gold_detect (lines 335-475).
    Vendor loops `for (obj = fobj; obj; obj = obj->nobj)` (line 372) and
    `for (mtmp = fmon; ...)` (line 435) without any distance check, then
    calls `map_object(temp, 1)` per gold-bearing location (lines 424, 430,
    445, 452, 457) — so the entire level's gold is exposed in one shot.
    """
    ts = state.timestep.astype(jnp.int32)
    new_ident = state.identification.replace(
        detect_treasure_until_turn=ts + jnp.int32(50),
    )

    # Reveal every cell on the current level that contains a COIN_CLASS
    # item, with no distance limit (vendor gold_detect:372-382).
    b  = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)

    level_cats = state.ground_items.category[b, lv]  # [H, W, MAX_STACK]
    has_gold = jnp.any(
        level_cats == jnp.int8(int(ItemCategory.COIN)), axis=-1
    )  # [H, W]

    new_explored = state.explored.at[b, lv].set(
        state.explored[b, lv] | has_gold
    )
    return state.replace(identification=new_ident, explored=new_explored)


# ---------------------------------------------------------------------------
# 3. detect_objects
#    Cite: vendor/nethack/src/detect.c::object_detect (lines 603-789).
#    Vendor with class==0 (ALL_CLASSES) iterates fobj (line 643), the
#    buried-object list (line 654), each monster's inventory (line 671),
#    then maps every found object via `map_object` (lines 709, 711, 731,
#    733, 748, 763, 773) across the whole level — no distance limit.
#    Sets detect_objects_until_turn = ts + 50 as the JAX model window.
# ---------------------------------------------------------------------------

def detect_objects(state, rng):
    """Reveal all objects on the current level; set timer for 50 turns.

    Cite: vendor/nethack/src/detect.c::object_detect (lines 603-789).
    With class==0 (ALL_CLASSES) vendor reveals every non-empty object
    location via the per-tile scan at lines 721-735.
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
#    Cite: vendor/nethack/src/detect.c::monster_detect (lines 798-862).
#    Vendor iterates fmon (line 828) and calls map_monst(mtmp, TRUE)
#    (line 834) for each non-dead monster on the level — no distance
#    limit.  100-turn window models the blessed-detector persistence
#    branch at line 848 (`if ((otmp && otmp->blessed) && !unconstrained)`).
# ---------------------------------------------------------------------------

def detect_monsters(state, rng):
    """Reveal all monsters on the current level; set timer for 100 turns.

    Cite: vendor/nethack/src/detect.c::monster_detect (lines 798-862).
    Vendor calls `map_monst(mtmp, TRUE)` (line 834) on every live monster.
    We track the window via the existing IdentificationState timer.
    """
    ts = state.timestep.astype(jnp.int32)
    new_ident = state.identification.replace(
        detect_monsters_until_turn=ts + jnp.int32(100),
    )
    return state.replace(identification=new_ident)


# ---------------------------------------------------------------------------
# 5. detect_magic
#    Cite: vendor/nethack/src/detect.c::object_detect (lines 603-789) invoked
#    via the SPE_DETECT_MAGIC / SCR_MAGIC_MAPPING-adjacent dispatch with the
#    wand/potion/scroll/ring/amulet/spellbook classes.  Internally the same
#    object_detect machinery iterates fobj / buried / minvent and calls
#    map_object(otmp, 1) for every match — level-wide.
#    Sets detect_magic_until_turn = timestep + 50 as the JAX model window.
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

    Cite: vendor/nethack/src/detect.c::object_detect (lines 603-789) — the
    same routine is invoked once per magic object class. Sets
    detect_magic_until_turn timer and marks magic-item cells explored.
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
#    Cite: vendor/nethack/src/detect.c::cvt_sdoor_to_door (lines 1589-1604).
#    Vendor cvt_sdoor_to_door does `lev->typ = DOOR; lev->doormask |= D_CLOSED`
#    (line 1599/1601-1602) — i.e. SDOOR becomes a closed door.  The matching
#    SCORR->CORR conversion lives in findone() (detect.c line 1639+) via
#    `lev->typ = CORR` once the secret corridor is detected.
#    Note: VendorTileType.DOOR maps to TileType.CLOSED_DOOR in the JAX
#    state's local enum (see constants/tiles.py).
# ---------------------------------------------------------------------------

_SDOOR = jnp.int8(int(VendorTileType.SDOOR))
_SCORR = jnp.int8(int(VendorTileType.SCORR))
_CLOSED_DOOR = jnp.int8(int(TileType.CLOSED_DOOR))
_CORRIDOR = jnp.int8(int(TileType.CORRIDOR))


def detect_unseen(state, rng):
    """Reveal secret doors (SDOOR->CLOSED_DOOR) and corridors (SCORR->CORRIDOR).

    Cite: vendor/nethack/src/detect.c::cvt_sdoor_to_door (lines 1589-1604);
          vendor/nethack/src/detect.c::findone (line 1639+) for the SCORR
          -> CORR conversion path.
    We convert the terrain tiles in-place on the current level.
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
#    Cite: vendor/nethack/src/detect.c::do_vicinity_map (lines 1448-1585).
#    Vendor scans an asymmetric rectangle centred (approximately) on the
#    player; lines 1464-1467:
#        lo_y = ((u.uy - 5 < 0) ? 0 : u.uy - 5)
#        hi_y = ((u.uy + 6 >= ROWNO) ? ROWNO - 1 : u.uy + 6)
#        lo_x = ((u.ux - 9 < 1) ? 1 : u.ux - 9)
#        hi_x = ((u.ux + 10 >= COLNO) ? COLNO - 1 : u.ux + 10)
#    and the loop (line 1500) is `for (zx = lo_x; zx <= hi_x; zx++)
#    for (zy = lo_y; zy <= hi_y; zy++)`, i.e. zy ∈ [u.uy-5, u.uy+6] and
#    zx ∈ [u.ux-9, u.ux+10] (inclusive on both ends).
#
#    Note: the row axis in JAX state corresponds to vendor 'y' and the
#    column axis to vendor 'x', so we mirror the asymmetric half-widths
#    accordingly (rows -5/+6, cols -9/+10).
# ---------------------------------------------------------------------------

def trap_detect(state, rng):
    """One-shot reveal of every trap on the current level.

    Cite: vendor/nethack/src/detect.c::trap_detect (lines 1011-1088) and
          vendor/nethack/src/detect.c::display_trap_map.
    Vendor walks gf.ftrap (line 1025), buried/inventory/door traps and
    calls display_trap_map which marks each trap's seenv on the level.
    There is NO timer — the reveal is immediate and persistent.

    Implementation: sets state.traps.revealed[flat_lv, :, :] = True so
    that every trap location on the current level is permanently visible.
    """
    b  = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)
    max_lv = jnp.int32(state.terrain.shape[1])
    flat_lv = b * max_lv + lv

    old_revealed = state.traps.revealed
    full_row = jnp.ones_like(old_revealed[flat_lv])
    new_revealed = old_revealed.at[flat_lv].set(full_row)
    return state.replace(traps=state.traps.replace(revealed=new_revealed))


def clairvoyance(state, rng):
    """Reveal the do_vicinity_map rectangle centred on the player.

    Cite: vendor/nethack/src/detect.c::do_vicinity_map (lines 1448-1585),
    specifically the lo_y/hi_y/lo_x/hi_x box at lines 1464-1467 and the
    nested scan at lines 1500-1537.
    """
    b  = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)

    pr = state.player_pos[0].astype(jnp.int32)
    pc = state.player_pos[1].astype(jnp.int32)

    H, W = state.terrain.shape[2], state.terrain.shape[3]
    rs = jnp.arange(H, dtype=jnp.int32)
    cs = jnp.arange(W, dtype=jnp.int32)
    # Vendor box (inclusive): rows ∈ [pr-5, pr+6], cols ∈ [pc-9, pc+10].
    row_in = (rs[:, None] >= pr - jnp.int32(5)) & (rs[:, None] <= pr + jnp.int32(6))
    col_in = (cs[None, :] >= pc - jnp.int32(9)) & (cs[None, :] <= pc + jnp.int32(10))
    in_box = row_in & col_in  # [H, W]

    new_explored = state.explored.at[b, lv].set(
        state.explored[b, lv] | in_box
    )
    return state.replace(explored=new_explored)
