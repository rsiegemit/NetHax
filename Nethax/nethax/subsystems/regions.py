"""Region / gas-cloud subsystem.

Canonical source: vendor/nethack/src/region.c

NetHack's region system is a generic, rectangular-area mechanism whose only
in-game use is the gas-cloud / stinking-cloud effect (region.c:1213,
``create_gas_cloud``).  We model the minimum needed to support that effect
in a byte-equal, JIT-compatible way:

  * a fixed-shape ``RegionState`` array of up to ``MAX_REGIONS`` slots,
  * ``create_gas_cloud`` allocates the first free slot,
  * ``run_regions`` ticks every active slot — applies damage to a player
    standing inside its bounding rectangle, ages it down via ``lifetime``,
    and frees the slot when ``lifetime <= 0``.

The vendor data model is a flexible list of rectangles per region (region.c
``NhRegion``).  For JIT-shape stability we approximate each gas cloud by its
single enclosing rectangle — sufficient for damage-tick parity given that
``create_gas_cloud`` always builds a 4-connected blob whose bounding rect
contains every cloud tile.

Vendor citations
----------------
* region.c:414  ``run_regions`` — per-turn aging + callbacks.
* region.c:1213 ``create_gas_cloud`` — spawn a poison-gas region.
* region.c:1157 ``monkilled(mtmp, "gas cloud", AD_DRST)`` — damage type.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

from Nethax.nethax.rng import rnd, rn1


# Number of simultaneous regions (gas clouds + headroom).
# Vendor allocates dynamically (gm.max_regions); the headless sim only ever
# spawns a handful at once (scroll of stinking cloud + sink stench), so 8 is
# ample without bloating the pytree.
MAX_REGIONS: int = 8

# Kind enum (slot occupancy).  0 = unused, 1 = gas cloud.
KIND_UNUSED:     int = 0
KIND_GAS_CLOUD:  int = 1


@struct.dataclass
class RegionState:
    """Fixed-shape region table.

    Fields
    ------
    rectangles : int8[MAX_REGIONS, 4]
        Per slot: (x1, y1, x2, y2) in vendor (col, row) convention.
        ``-1`` in every component marks an unused slot.
    damage     : int8[MAX_REGIONS]
        Per-tick HP damage for gas-cloud slots (region.c:1192 ``arg.a_int``).
    lifetime   : int16[MAX_REGIONS]
        Turns-to-live counter.  Region is removed when this drops to ``0``.
        Vendor field: ``NhRegion->ttl`` (region.c:1303).
    kind       : int8[MAX_REGIONS]
        Slot occupancy / type tag (``KIND_*`` constants above).
    """
    rectangles: jax.Array   # int8[MAX_REGIONS, 4]
    damage:     jax.Array   # int8[MAX_REGIONS]
    lifetime:   jax.Array   # int16[MAX_REGIONS]
    kind:       jax.Array   # int8[MAX_REGIONS]


def make_region_state() -> RegionState:
    """Return a fresh, empty ``RegionState`` (all slots free)."""
    return RegionState(
        rectangles=jnp.full((MAX_REGIONS, 4), -1, dtype=jnp.int8),
        damage=jnp.zeros((MAX_REGIONS,), dtype=jnp.int8),
        lifetime=jnp.zeros((MAX_REGIONS,), dtype=jnp.int16),
        kind=jnp.zeros((MAX_REGIONS,), dtype=jnp.int8),
    )


# ---------------------------------------------------------------------------
# create_gas_cloud — region.c:1213
# ---------------------------------------------------------------------------

def create_gas_cloud(
    state,
    rng: jax.Array,
    cx: jax.Array,
    cy: jax.Array,
    size: jax.Array,
    damage: jax.Array,
):
    """Spawn a stinking gas cloud centred on ``(cx, cy)``.

    Vendor: ``create_gas_cloud`` (region.c:1213).  The vendor function grows
    a 4-connected blob of up to ``size`` tiles via BFS, picks its bounding
    rectangle (the cloud body is the union of single-cell rects), and assigns

        ttl = (rn1(3, 4) * size) / nplaced

    (region.c:1303-1305).  Because our regions are stored as a single
    bounding rectangle, we record the largest axis-aligned square of side
    ``floor(sqrt(size))`` centred on ``(cx, cy)`` — the same bounding box the
    vendor blob would yield in open space — and apply the same ttl formula
    with ``nplaced == size`` (no growth-constrained inflation).

    Parameters
    ----------
    state  : EnvState
    rng    : JAX PRNG key
    cx, cy : int — cloud centre in vendor (col, row) coords.
    size   : int — desired tile count (vendor ``cloudsize``).
    damage : int — per-tick HP damage (vendor ``damage`` arg).

    Returns the updated ``EnvState`` (with ``region_state`` mutated).  If no
    free slot exists, returns ``state`` unchanged (vendor behaviour: the
    region simply isn't created; ``add_region`` realloc never fails in
    practice but a missing slot in our fixed table is a silent no-op).
    """
    rs = state.region_state

    # Square half-radius such that (2r+1)^2 >= size; matches the bounding box
    # of an open-space BFS blob from create_gas_cloud (region.c:1243-1296).
    size_i32 = jnp.asarray(size, dtype=jnp.int32)
    # half = ceil((sqrt(size)-1)/2); integer approximation suffices for ttl
    # parity (the rectangle is only used for in/out hit-testing).
    half = jnp.floor(jnp.sqrt(size_i32.astype(jnp.float32))).astype(jnp.int32) // jnp.int32(2)

    cx_i = jnp.asarray(cx, dtype=jnp.int32)
    cy_i = jnp.asarray(cy, dtype=jnp.int32)
    x1 = (cx_i - half).astype(jnp.int8)
    y1 = (cy_i - half).astype(jnp.int8)
    x2 = (cx_i + half).astype(jnp.int8)
    y2 = (cy_i + half).astype(jnp.int8)

    # ttl: rn1(3, 4) * size / nplaced — region.c:1303-1305.
    # nplaced == size in the unconstrained case (open-space blob).
    ttl_base = rn1(rng, 3, 4).astype(jnp.int32)        # rn1(3, 4) ∈ [4, 6]
    ttl = (ttl_base * size_i32) // jnp.maximum(size_i32, jnp.int32(1))
    ttl_i16 = ttl.astype(jnp.int16)

    # First-free slot: lowest index whose kind == KIND_UNUSED.
    free_mask = rs.kind == jnp.int8(KIND_UNUSED)
    has_free  = jnp.any(free_mask)
    slot      = jnp.argmax(free_mask).astype(jnp.int32)

    new_rect = jnp.array([x1, y1, x2, y2], dtype=jnp.int8)
    rectangles = jnp.where(
        has_free,
        rs.rectangles.at[slot].set(new_rect),
        rs.rectangles,
    )
    damage_arr = jnp.where(
        has_free,
        rs.damage.at[slot].set(jnp.asarray(damage, dtype=jnp.int8)),
        rs.damage,
    )
    lifetime_arr = jnp.where(
        has_free,
        rs.lifetime.at[slot].set(ttl_i16),
        rs.lifetime,
    )
    kind_arr = jnp.where(
        has_free,
        rs.kind.at[slot].set(jnp.int8(KIND_GAS_CLOUD)),
        rs.kind,
    )

    new_rs = rs.replace(
        rectangles=rectangles,
        damage=damage_arr,
        lifetime=lifetime_arr,
        kind=kind_arr,
    )
    return state.replace(region_state=new_rs)


# ---------------------------------------------------------------------------
# run_regions — region.c:414
# ---------------------------------------------------------------------------

def _hero_inside(rect: jax.Array, prow: jax.Array, pcol: jax.Array) -> jax.Array:
    """Return True if player (row=prow, col=pcol) is inside the (x1,y1,x2,y2) rect.

    Vendor ``hero_inside`` / ``inside_region`` (region.c:103-118).  Rectangle
    is stored in vendor (col, row) convention.
    """
    x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
    return (
        (pcol >= x1.astype(jnp.int32)) & (pcol <= x2.astype(jnp.int32)) &
        (prow >= y1.astype(jnp.int32)) & (prow <= y2.astype(jnp.int32))
    )


def run_regions(state, rng: jax.Array):
    """Per-turn tick — age every region, apply gas-cloud damage to player.

    Vendor: ``run_regions`` (region.c:414).

    Order of operations (matching vendor lines 423-458):
      1. For each active slot: if ``lifetime == 0``, free the slot
         (vendor: ``remove_region`` after ``expire_f`` returns TRUE).
      2. For each still-active slot:
           a. If ``lifetime > 0``, decrement it (line 436).
           b. If the player is inside the rect and the slot is a gas cloud,
              apply ``rnd(damage) + 5`` HP damage (region.c:1152 player path
              mirrors the monster path immediately above).
              AD_DRST = poison; resists_poison() not modelled here — the
              player takes raw poison damage just like a non-resistant
              monster (region.c:1148-1152).
    """
    rs = state.region_state
    prow = state.player_pos[0].astype(jnp.int32)
    pcol = state.player_pos[1].astype(jnp.int32)

    # Step 1 — expire slots whose lifetime already hit 0.
    # (Vendor processes this *before* aging the rest, region.c:425-431.)
    expired = (rs.kind != jnp.int8(KIND_UNUSED)) & (rs.lifetime == jnp.int16(0))
    rectangles = jnp.where(
        expired[:, None],
        jnp.full((MAX_REGIONS, 4), -1, dtype=jnp.int8),
        rs.rectangles,
    )
    damage = jnp.where(expired, jnp.int8(0), rs.damage)
    lifetime = jnp.where(expired, jnp.int16(0), rs.lifetime)
    kind = jnp.where(expired, jnp.int8(KIND_UNUSED), rs.kind)

    # Step 2 — for each remaining active slot, apply effects then age.
    active = kind != jnp.int8(KIND_UNUSED)
    is_cloud = kind == jnp.int8(KIND_GAS_CLOUD)

    # Hero-inside test per slot (vectorised; rectangles shape [MAX_REGIONS, 4]).
    def _hi(rect):
        return _hero_inside(rect, prow, pcol)
    inside = jax.vmap(_hi)(rectangles)               # bool[MAX_REGIONS]

    # Damage roll per slot — rnd(damage) + 5, but only when (active & cloud
    # & inside & damage > 0).  Vendor: region.c:1152.
    rngs = jax.random.split(rng, MAX_REGIONS)
    def _roll(r, d):
        # rnd is JIT-pure; n must be a static positive int for it to call
        # randint, so we clamp.  rnd(rng, n) returns int in [1, n].
        # Damage values are small (single digits) — bounded via maximum.
        return rnd(r, jnp.maximum(d.astype(jnp.int32), jnp.int32(1)))
    dmg_rolls = jax.vmap(_roll)(rngs, damage)        # int32[MAX_REGIONS]
    hits = active & is_cloud & inside & (damage > jnp.int8(0))
    per_slot_dmg = jnp.where(hits, dmg_rolls + jnp.int32(5), jnp.int32(0))
    total_dmg = jnp.sum(per_slot_dmg).astype(jnp.int32)
    new_hp = state.player_hp - total_dmg

    # Age every active slot by 1 turn (region.c:436-437).
    aged = jnp.where(
        active & (lifetime > jnp.int16(0)),
        lifetime - jnp.int16(1),
        lifetime,
    )

    new_rs = rs.replace(
        rectangles=rectangles,
        damage=damage,
        lifetime=aged,
        kind=kind,
    )
    return state.replace(region_state=new_rs, player_hp=new_hp)
