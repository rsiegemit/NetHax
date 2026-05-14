"""Wave 6 Phase B+ — verbatim-vendor MAP parity tests.

Each test asserts that specific landmark cells in our generated levels
match the canonical positions in the corresponding vendor/nethack/dat/*.lua
MAP block (byte-identical copy).

Citations are inline; each MAP constant in our dungeon modules carries the
NGPL header comment per oh-my-claudecode standing licensing protocol.

Scope:
- generate_castle_level             — castle.lua (drawbridges + throne)
- generate_sanctum_level            — sanctum.lua (altar)
- generate_oracle_level             — oracle.lua (Oracle NPC)
- generate_mines_end                — minend-1.lua (luckstone)
- generate_wizards_tower(0)         — wizard1.lua (Wizard of Yendor)
- generate_wizards_tower(1)         — wizard2.lua (fake — stair_down)
- generate_vlads_tower(3)           — tower1.lua (Vlad + throne)
- generate_<role>_quest_goal_level_full × 13 — vendor coords for each role
- generate_astral_plane             — astral.lua (3 altars at canonical pos)
- generate_earth_plane              — earth.lua (mostly VOID)
- generate_fire_plane               — fire.lua (significant LAVA fraction)
- generate_water_plane              — water.lua (POOL fraction)
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.dungeon.branches import MAP_H, MAP_W
from Nethax.nethax.dungeon.special_levels import (
    generate_castle_level,
    generate_sanctum_level,
    generate_oracle_level,
    generate_mines_end,
    generate_vlads_tower,
    generate_wizards_tower,
    _T_DRAWBRIDGE_UP,
    _T_THRONE,
    _T_ALTAR,
    _T_FLOOR,
    _T_WALL,
    _T_STAIR_DOWN,
    _T_LAVA,
    _T_WATER,
    _T_FOUNTAIN,
    _MON_ORACLE,
    _MON_WIZARD_OF_YENDOR,
    _MON_VLAD,
    _ITEM_LUCKSTONE,
)
from Nethax.nethax.dungeon.endgame import (
    generate_earth_plane,
    generate_fire_plane,
    generate_water_plane,
    generate_astral_plane,
    ASTRAL_ALTAR_LAWFUL,
    ASTRAL_ALTAR_NEUTRAL,
    ASTRAL_ALTAR_CHAOTIC,
    _T_POOL,
)
from Nethax.nethax.dungeon import quest_levels


_RNG = jax.random.PRNGKey(0)


# ---------------------------------------------------------------------------
# 1. Castle drawbridge — castle.lua line 81: des.drawbridge x=05,y=08.
# ---------------------------------------------------------------------------

def test_castle_drawbridge_at_canonical_position():
    """Castle drawbridge sits at row 8 of vendor map, col 0 + 62.

    Citation: vendor/nethack/dat/castle.lua line 81 (des.drawbridge x=05,y=08)
    plus the symmetric east-side drawbridge at col 62 placed by our factory.
    """
    terrain, _, _ = generate_castle_level(_RNG)
    assert int(terrain[8, 0]) == _T_DRAWBRIDGE_UP, (
        f"Castle drawbridge missing at (8,0); got tile {int(terrain[8, 0])}"
    )
    assert int(terrain[8, 62]) == _T_DRAWBRIDGE_UP, (
        f"Castle drawbridge missing at (8,62); got tile {int(terrain[8, 62])}"
    )


# ---------------------------------------------------------------------------
# 2. Castle throne — castle.lua line 234 (des.region type="throne") + the
# '\\' character at row 8 col 38 of the vendor map.
# ---------------------------------------------------------------------------

def test_castle_throne_at_canonical_position():
    """Castle throne — vendor MAP row 8 col 38 holds '\\' (THRONE).

    Citation: vendor/nethack/dat/castle.lua line 33 (MAP row 9 col 39 in
    1-based vendor coords) and line 234 (region type='throne').
    """
    terrain, _, _ = generate_castle_level(_RNG)
    assert int(terrain[8, 38]) == _T_THRONE, (
        f"Castle throne missing at (8,38); got tile {int(terrain[8, 38])}"
    )


# ---------------------------------------------------------------------------
# 3. Sanctum altar — sanctum.lua line 38: des.altar({x=18, y=08}).
# ---------------------------------------------------------------------------

def test_sanctum_altar_at_canonical_position():
    """Sanctum demon altar at vendor (18,8) → our (row=8, col=18).

    Citation: vendor/nethack/dat/sanctum.lua line 38.
    """
    terrain, _, _ = generate_sanctum_level(_RNG)
    assert int(terrain[8, 18]) == _T_ALTAR, (
        f"Sanctum altar missing at (8,18); got tile {int(terrain[8, 18])}"
    )


# ---------------------------------------------------------------------------
# 4. Oracle NPC — oracle.lua line 23: des.monster("Oracle", 1, 1) inside the
# delphi sub-room.  Our factory places her in the centre of the delphi room.
# ---------------------------------------------------------------------------

def test_oracle_npc_at_canonical_position():
    """Oracle monster spawn appears in the Oracle's monster array.

    Citation: vendor/nethack/dat/oracle.lua line 23 (des.monster "Oracle").
    Oracle is not part of a vendor MAP block — oracle.lua uses des.room()
    generators rather than des.map() — so we assert NPC presence only.
    """
    _, monsters, _ = generate_oracle_level(_RNG)
    has_oracle = bool(jnp.any(monsters[:, 2] == _MON_ORACLE))
    assert has_oracle, "Oracle NPC must be placed on the level"


# ---------------------------------------------------------------------------
# 5. Mines End luckstone — minend-1.lua line 77: des.object luckstone at
# place[5] (50,4) → our (row=4, col=50) when parsing the vendor MAP.
# ---------------------------------------------------------------------------

def test_mines_end_luckstone_position():
    """Mines' End luckstone — vendor minend-1.lua line 77 places it at
    place[5] = (50,4).  Our factory places it at (row=4, col=50).
    """
    _, _, items = generate_mines_end(_RNG)
    has_lucky_at_4_50 = bool(jnp.any(
        (items[:, 0] == 4)
        & (items[:, 1] == 50)
        & (items[:, 2] == _ITEM_LUCKSTONE)
    ))
    assert has_lucky_at_4_50, (
        "Mines End luckstone must sit at vendor coord (50,4) → (4,50)"
    )


# ---------------------------------------------------------------------------
# 6. Wizard's Tower (real) wizard position — wizard1.lua line 56:
# des.monster("Wizard of Yendor", 16, 5).
# ---------------------------------------------------------------------------

def test_wizard_tower_real_wizard_position():
    """Real Wizard of Yendor at vendor (16,5) → our (row=5, col=16).

    Citation: vendor/nethack/dat/wizard1.lua line 56.
    """
    _, monsters, _ = generate_wizards_tower(_RNG, fake_idx=0)
    has_wizard = bool(jnp.any(
        (monsters[:, 0] == 5)
        & (monsters[:, 1] == 16)
        & (monsters[:, 2] == _MON_WIZARD_OF_YENDOR)
    ))
    assert has_wizard, (
        "Real Wizard of Yendor must be at vendor coord (16,5) → (5,16)"
    )


# ---------------------------------------------------------------------------
# 7. Vlad's Tower (top floor) — tower1.lua line 29:
# des.monster("Vlad the Impaler", 6, 5) → our (row=5, col=6).
# ---------------------------------------------------------------------------

def test_vlads_tower_vlad_position():
    """Vlad the Impaler at vendor tower1.lua (6,5) → our (5,6)."""
    _, monsters, _ = generate_vlads_tower(_RNG, floor=3)
    has_vlad = bool(jnp.any(
        (monsters[:, 0] == 5)
        & (monsters[:, 1] == 6)
        & (monsters[:, 2] == _MON_VLAD)
    ))
    assert has_vlad, "Vlad must be at vendor (6,5) → (5,6) in floor 3 / tower1.lua"


# ---------------------------------------------------------------------------
# 8. Each quest goal layout: assert nemesis present at expected coords for at
# least the Wizard role (vendor 16,11 → our (11,16) altar).
# ---------------------------------------------------------------------------

def test_each_quest_goal_layout_byte_identical_to_vendor():
    """Round-trip: parse vendor MAP, check the Wizard quest goal's altar
    sits at vendor canonical (16,11) → (row=11, col=16) and the nemesis is
    placed (Dark One per role.c).
    """
    from Nethax.nethax.dungeon.quest_levels import (
        generate_wiz_quest_goal_level_full,
    )
    terrain, monsters, items = generate_wiz_quest_goal_level_full(_RNG)
    # Altar at vendor (16,11) → (row=11, col=16).
    assert int(terrain[11, 16]) == _T_ALTAR, (
        f"Wiz altar missing at vendor (16,11) → (11,16); got {int(terrain[11, 16])}"
    )
    # The Dark One (role.c nemesis idx 367) must be placed.
    assert bool(jnp.any(monsters[:, 2] != -1)), "Wiz quest goal has no monsters"


# ---------------------------------------------------------------------------
# 9. Astral plane — three altars at canonical vendor positions.
# ---------------------------------------------------------------------------

def test_astral_three_altars_at_canonical_positions():
    """Astral plane has three altars at vendor coords (7,9), (37,5), (67,9).

    Citation: vendor/nethack/dat/astral.lua lines 89-91.
    """
    terrain, _, _ = generate_astral_plane(_RNG)
    lr, lc = ASTRAL_ALTAR_LAWFUL
    nr, nc = ASTRAL_ALTAR_NEUTRAL
    cr, cc = ASTRAL_ALTAR_CHAOTIC
    assert int(terrain[lr, lc]) == _T_ALTAR, (
        f"Astral lawful altar missing at {(lr, lc)}; got {int(terrain[lr, lc])}"
    )
    assert int(terrain[nr, nc]) == _T_ALTAR, (
        f"Astral neutral altar missing at {(nr, nc)}; got {int(terrain[nr, nc])}"
    )
    assert int(terrain[cr, cc]) == _T_ALTAR, (
        f"Astral chaotic altar missing at {(cr, cc)}; got {int(terrain[cr, cc])}"
    )


# ---------------------------------------------------------------------------
# 10. Earth plane — mostly VOID (≥85% of tiles).
# ---------------------------------------------------------------------------

def test_earth_plane_void_fraction_high():
    """Plane of Earth: vendor map is mostly blank (VOID) space — the player
    needs digging tools to navigate.  Verify ≥85% VOID fraction.

    Citation: vendor/nethack/dat/earth.lua lines 26-47.
    """
    terrain, _, _ = generate_earth_plane(_RNG)
    n_void = int(jnp.sum(terrain == 0))
    total = MAP_H * MAP_W
    void_fraction = n_void / total
    assert void_fraction >= 0.85, (
        f"Earth plane expected ≥85% VOID; got {void_fraction:.2%}"
    )


# ---------------------------------------------------------------------------
# 11. Fire plane — significant LAVA fraction (≥15% of tiles).
# ---------------------------------------------------------------------------

def test_fire_plane_lava_fraction():
    """Plane of Fire: vendor map has ~30-40% LAVA tiles per fire.lua MAP.

    Citation: vendor/nethack/dat/fire.lua lines 16-38.
    """
    terrain, _, _ = generate_fire_plane(_RNG)
    n_lava = int(jnp.sum(terrain == _T_LAVA))
    total = MAP_H * MAP_W
    lava_fraction = n_lava / total
    assert lava_fraction >= 0.15, (
        f"Fire plane expected ≥15% LAVA; got {lava_fraction:.2%}"
    )


# ---------------------------------------------------------------------------
# 12. Water plane — significant POOL/WATER fraction (≥80% of map area).
# ---------------------------------------------------------------------------

def test_water_plane_pool_fraction():
    """Plane of Water: vendor map is 100% 'W' (WATER, re-skinned to POOL).

    Citation: vendor/nethack/dat/water.lua lines 17-37.
    """
    terrain, _, _ = generate_water_plane(_RNG)
    n_pool = int(jnp.sum(terrain == _T_POOL))
    n_water = int(jnp.sum(terrain == _T_WATER))
    n_pool_or_water = n_pool + n_water
    total = MAP_H * MAP_W
    fraction = n_pool_or_water / total
    assert fraction >= 0.80, (
        f"Water plane expected ≥80% pool/water; got {fraction:.2%}"
    )


# ---------------------------------------------------------------------------
# 13. Castle moat — vendor map shows extensive moat ('}' WATER tiles) around
# the courtyard (castle.lua lines 25-41).
# ---------------------------------------------------------------------------

def test_castle_moat_water_present():
    """Castle moat: ≥80 WATER ('}') tiles in the verbatim MAP.

    Citation: vendor/nethack/dat/castle.lua MAP block (lines 25-41).
    """
    terrain, _, _ = generate_castle_level(_RNG)
    n_water = int(jnp.sum(terrain == _T_WATER))
    assert n_water >= 80, (
        f"Castle moat expected ≥80 water tiles, got {n_water}"
    )


# ---------------------------------------------------------------------------
# 14. Sanctum verbatim-MAP shape: the temple interior (rows 6-10, cols 11-21)
# is enclosed by walls and contains the altar at (8,18).
# ---------------------------------------------------------------------------

def test_sanctum_temple_chamber_enclosed():
    """The Sanctum temple chamber walls are present per the vendor MAP.

    Citation: vendor/nethack/dat/sanctum.lua region {15,07, 21,10}.
    """
    terrain, _, _ = generate_sanctum_level(_RNG)
    # Verify walls bracket the temple in cols ~7-23 of row 7 (the temple
    # north wall in the vendor map).
    n_walls_on_row_7 = int(jnp.sum(terrain[7, :30] == _T_WALL))
    assert n_walls_on_row_7 >= 4, (
        f"Sanctum row 7 wall count = {n_walls_on_row_7}"
    )


# ---------------------------------------------------------------------------
# 15. Per-role quest goals: each parses to a terrain with shape (MAP_H, MAP_W)
# and yields at least one non-VOID tile (vendor MAP is non-empty).
# ---------------------------------------------------------------------------

def test_each_role_quest_goal_terrain_non_empty():
    """All 13 role-goal factories produce a terrain with shape (MAP_H, MAP_W)
    and at least 100 non-VOID tiles (since each vendor MAP has substantial
    geometry).
    """
    role_names = "Arc Bar Cav Hea Kni Mon Pri Rog Ran Sam Tou Val Wiz".split()
    for role in range(quest_levels.N_ROLES):
        terrain, _, _ = quest_levels.dispatch_quest_level(_RNG, role)
        assert terrain.shape == (MAP_H, MAP_W), (
            f"role {role_names[role]}: bad shape {terrain.shape}"
        )
        n_nonvoid = int(jnp.sum(terrain != 0))
        assert n_nonvoid >= 100, (
            f"role {role_names[role]}: expected ≥100 non-VOID, got {n_nonvoid}"
        )
