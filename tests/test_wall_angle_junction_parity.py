"""Wall-angle junction parity: _apply_wall_angle vs vendor fix_wall_spines.

Nethax stores a single TileType.WALL and re-derives the display wall variant
(hwall / vwall / corners / T-junctions / crosswall) at RENDER time from a
4-neighbour wall-presence bitmask (_WALL_ANGLE_TABLE in nle_obs.py).

Vendor NetHack instead bakes the wall typ at GENERATION in
vendor/nethack/src/mkmaze.c::fix_wall_spines via ``spine_array`` indexed by
the NSEW spine bitmask (bits = N<<3 | S<<2 | E<<1 | W), then wall_angle()
maps that typ to the S_* display glyph.  For a fully-seen wall the typ->glyph
map is the identity family member, so the render-time table MUST reproduce
``spine_array`` exactly.

Room-family envs only ever hit isolated stub / straight-run / L-corner
patterns, so a bug in the four T-junction entries (which occur at multi-room
and maze junctions, e.g. Sokoban) is invisible to the Room byte-parity gate.
This test pins every one of the 16 neighbour combinations directly against
the vendor spine_array, catching exactly that class of regression.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax.numpy as jnp

from Nethax.nethax.obs.nle_obs import _apply_wall_angle
from Nethax.nethax.constants.tiles import TileType


# Vendor cmap indices for the wall family (vendor/nethack/include/defsym.h,
# offset so S_vwall==1 matching _build_wall_angle_table's local numbering).
_S = {
    "VWALL": 1, "HWALL": 2,
    "TLCORNER": 3, "TRCORNER": 4, "BLCORNER": 5, "BRCORNER": 6,
    "CROSSWALL": 7,
    "TUWALL": 8, "TDWALL": 9, "TLWALL": 10, "TRWALL": 11,
}

# Verbatim from vendor/nethack/src/mkmaze.c::fix_wall_spines spine_array,
# indexed by bits = (N<<3) | (S<<2) | (E<<1) | W.
_VENDOR_SPINE = [
    "VWALL", "HWALL", "HWALL", "HWALL",
    "VWALL", "TRCORNER", "TLCORNER", "TDWALL",
    "VWALL", "BRCORNER", "BLCORNER", "TUWALL",
    "VWALL", "TLWALL", "TRWALL", "CROSSWALL",
]


def _render_center_wall(N, S, E, W):
    """Return the cmap variant _apply_wall_angle picks for a WALL cell whose
    N/S/E/W neighbours are walls per the given flags (all else open)."""
    WALL = int(TileType.WALL)
    FLOOR = int(TileType.FLOOR)
    # 3x3 patch, center at (1,1); pad the used neighbours only.
    terrain = np.full((3, 3), FLOOR, dtype=np.int16)
    terrain[1, 1] = WALL
    if N:
        terrain[0, 1] = WALL
    if S:
        terrain[2, 1] = WALL
    if W:
        terrain[1, 0] = WALL
    if E:
        terrain[1, 2] = WALL
    cmap = np.zeros((3, 3), dtype=np.int16)
    out = _apply_wall_angle(jnp.asarray(terrain), jnp.asarray(cmap))
    return int(np.asarray(out)[1, 1])


def test_all_16_neighbour_patterns_match_vendor_spine():
    """_apply_wall_angle must reproduce vendor spine_array for every NSEW combo.

    bits==0 is the free-standing wall: vendor leaves typ alone (VWALL default),
    and our table also yields S_vwall, so it's included.
    """
    mismatches = []
    for N in (0, 1):
        for S in (0, 1):
            for E in (0, 1):
                for W in (0, 1):
                    bits = (N << 3) | (S << 2) | (E << 1) | W
                    expected_name = _VENDOR_SPINE[bits]
                    expected = _S[expected_name]
                    got = _render_center_wall(N, S, E, W)
                    if got != expected:
                        got_name = next((k for k, v in _S.items() if v == got), got)
                        mismatches.append(
                            f"N{N}S{S}E{E}W{W}: expected {expected_name}"
                            f"({expected}) got {got_name}({got})"
                        )
    assert not mismatches, "wall-angle mismatches vs vendor spine_array:\n" + "\n".join(mismatches)


def test_t_junctions_specifically():
    """Pin the four T-junctions that Room parity never exercises (the ones
    that were previously swapped and broke Sokoban junctions)."""
    # walls N+S+E, W open  -> TRWALL
    assert _render_center_wall(1, 1, 1, 0) == _S["TRWALL"]
    # walls N+S+W, E open  -> TLWALL
    assert _render_center_wall(1, 1, 0, 1) == _S["TLWALL"]
    # walls N+E+W, S open  -> TUWALL
    assert _render_center_wall(1, 0, 1, 1) == _S["TUWALL"]
    # walls S+E+W, N open  -> TDWALL
    assert _render_center_wall(0, 1, 1, 1) == _S["TDWALL"]
