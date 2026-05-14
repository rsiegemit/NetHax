"""Tests for the MiniHack des-file parser.

These tests exercise the parser and emitter against synthetic minimal
des sources and the real canonical files under
``vendor/minihack/minihack/dat/``.
"""
from __future__ import annotations

import os

import pytest

import jax

from Nethax.minihax import des_parser as dp


DAT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "vendor", "minihack", "minihack", "dat",
)


def _dat_path(name: str) -> str:
    return os.path.join(DAT_DIR, name)


def _read(name: str) -> str:
    with open(_dat_path(name), "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# 1. Minimal ROOM source.
# ---------------------------------------------------------------------------


def test_parse_simple_room():
    src = """\
LEVEL: "tiny"

ROOM: "ordinary", lit, (3,3), (center,center), (5,5) {
    STAIR: random, down
}
"""
    ast = dp.parse_des(src)
    # The room is captured as a Room statement.
    rooms = [s for s in ast.statements if isinstance(s, dp.Room)]
    assert len(rooms) == 1
    room = rooms[0]
    assert room.kind == "ordinary"
    assert room.lit is True
    # Body should contain at least one statement (the STAIR).
    assert any(isinstance(b, dp.StairStmt) for b in room.body)


# ---------------------------------------------------------------------------
# 2. Minimal MAP + REGION → factory produces a 5x5 lit ordinary region.
# ---------------------------------------------------------------------------


def test_parse_room_5x5_des():
    """Synthetic 5x5 room (room-5x5.des is not in MiniHack)."""
    src = """\
MAZE: "mylevel", ' '
INIT_MAP: solidfill, ' '
GEOMETRY: center, center
MAP
.....
.....
.....
.....
.....
ENDMAP
REGION: (0,0,4,4), lit, "ordinary"
"""
    ast = dp.parse_des(src)
    assert ast.map_block is not None
    rows = ast.map_block.grid.split("\n")
    assert len(rows) == 5
    assert all(row == "....." for row in rows)
    # Compile and check that the region/map are emitted on the mock.
    fn = dp.compile_des(ast)
    lg = dp._MockLevelGenerator()
    fn(lg, jax.random.PRNGKey(0))
    call_names = [c[0] for c in lg.calls]
    assert "set_map" in call_names
    assert "add_region" in call_names


# ---------------------------------------------------------------------------
# 3. MONSTER with name.
# ---------------------------------------------------------------------------


def test_parse_monster_with_name():
    src = """\
LEVEL: "x"
MONSTER: ('d', "dog"), (3,3)
"""
    ast = dp.parse_des(src)
    monsters = [s for s in ast.statements if isinstance(s, dp.Monster)]
    assert len(monsters) == 1
    m = monsters[0]
    assert m.sym == "d"
    assert m.name == "dog"
    assert isinstance(m.pos, dp.Coord)
    assert m.pos.x == 3 and m.pos.y == 3


# ---------------------------------------------------------------------------
# 4. TERRAIN rect → fill_terrain call.
# ---------------------------------------------------------------------------


def test_parse_terrain_rect():
    src = """\
LEVEL: "x"
TERRAIN: (1,1,5,5), 'L'
"""
    ast = dp.parse_des(src)
    terrains = [s for s in ast.statements if isinstance(s, dp.Terrain)]
    assert len(terrains) == 1
    t = terrains[0]
    assert t.glyph == "L"
    fn = dp.compile_des(ast)
    lg = dp._MockLevelGenerator()
    fn(lg, 0)
    fills = [c for c in lg.calls if c[0] == "fill_terrain"]
    # At least one fill_terrain call with glyph 'L'.
    assert any(c[2].get("glyph") == "L" for c in fills)


# ---------------------------------------------------------------------------
# 5. CHOICE branch picks one with seeded rng.
# ---------------------------------------------------------------------------


def test_parse_choice_branch_picks_one():
    src = """\
LEVEL: "x"
CHOICE {
    MONSTER: ('d',"dog"),(1,1)
} | {
    MONSTER: ('f',"cat"),(2,2)
}
"""
    ast = dp.parse_des(src)
    choices = [s for s in ast.statements if isinstance(s, dp.Choice)]
    assert len(choices) == 1
    assert len(choices[0].branches) == 2
    # Compile with two different seeds and confirm at most one monster
    # is emitted per run (i.e. exactly one branch fires).
    f = dp.compile_des(ast)
    for seed in (0, 1, 2, 3):
        lg = dp._MockLevelGenerator()
        f(lg, seed)
        monsters = [c for c in lg.calls if c[0] == "add_monster"]
        assert len(monsters) == 1


# ---------------------------------------------------------------------------
# 6. corridor2.des → factory produces at least one corridor directive.
# ---------------------------------------------------------------------------


def test_compile_corridor_des():
    src = _read("corridor2.des")
    ast = dp.parse_des(src)
    fn = dp.compile_des(ast)
    lg = dp._MockLevelGenerator()
    fn(lg, jax.random.PRNGKey(7))
    # At least one room added, plus a RANDOM_CORRIDORS directive.
    names = [c[0] for c in lg.calls]
    assert names.count("add_room") >= 2
    assert "add_random_corridors" in names


# ---------------------------------------------------------------------------
# 7. Comments and blank lines tolerated.
# ---------------------------------------------------------------------------


def test_parser_handles_comments_and_blanks():
    src = """\
# top-of-file comment
LEVEL: "x"

# stand-alone comment line
ROOM: "ordinary", lit, (3,3), (center,center), (5,5) {
    # inner comment
    STAIR: random, down
}


# trailing blank lines and comment
"""
    ast = dp.parse_des(src)
    rooms = [s for s in ast.statements if isinstance(s, dp.Room)]
    assert len(rooms) == 1


# ---------------------------------------------------------------------------
# 8. lavacross / lava_crossing → produces lava terrain glyph.
# ---------------------------------------------------------------------------


def test_round_trip_lavacross_easy():
    src = _read("lava_crossing.des")
    ast = dp.parse_des(src)
    fn = dp.compile_des(ast)
    lg = dp._MockLevelGenerator()
    fn(lg, jax.random.PRNGKey(0))
    # MAP block contains the 'L' glyph; set_map should preserve it.
    set_map = [c for c in lg.calls if c[0] == "set_map"]
    assert set_map, "expected set_map to be called for lava_crossing.des"
    rows = set_map[0][1][0]
    assert any("L" in r for r in rows)


# ---------------------------------------------------------------------------
# 9. key_and_door.des → factory produces room + key + door.
# ---------------------------------------------------------------------------


def test_parse_keyroom_fixed_s_des():
    """key_and_door.des: ROOM with SUBROOM containing STAIR + DOOR + key OBJECT."""
    src = _read("key_and_door.des")
    ast = dp.parse_des(src)
    rooms = [s for s in ast.statements if isinstance(s, dp.Room)]
    assert rooms, "expected at least one ROOM"
    # Walk into the room body to find object/door.
    flat: list = []

    def _walk(items):
        for it in items:
            flat.append(it)
            body = getattr(it, "body", None)
            if body:
                _walk(body)

    _walk(ast.statements)
    objects = [s for s in flat if isinstance(s, dp.ObjectStmt)]
    doors = [s for s in flat if isinstance(s, (dp.Door, dp.RoomDoor))]
    assert objects, "expected at least one OBJECT in key_and_door.des"
    assert doors, "expected at least one DOOR in key_and_door.des"
    # And a stair somewhere.
    stairs = [s for s in flat if isinstance(s, dp.StairStmt)]
    assert stairs


# ---------------------------------------------------------------------------
# 10. Smoke-parse the entire corpus.
# ---------------------------------------------------------------------------


def test_smoke_parse_all_canonical_des_files():
    ok, total, failures = dp.smoke_parse_dir(DAT_DIR)
    # Helpful failure message listing what failed.
    if failures:  # pragma: no cover — diagnostic path
        msg = "\n".join(f"  {name}: {err}" for name, err in failures)
        pytest.fail(f"{len(failures)} of {total} des files failed to parse:\n{msg}")
    assert ok == total
    assert total >= 30  # MiniHack ships 36 canonical files


# ---------------------------------------------------------------------------
# 11. Container with body.
# ---------------------------------------------------------------------------


def test_parse_container_with_body():
    src = _read("chest.des")
    ast = dp.parse_des(src)
    containers = [s for s in ast.statements if isinstance(s, dp.Container)]
    assert containers, "expected a CONTAINER statement"
    c = containers[0]
    assert c.sym == "("
    assert c.name == "chest"
    assert any(isinstance(b, dp.ObjectStmt) for b in c.body)


# ---------------------------------------------------------------------------
# 12. IF/ELSE evaluation.
# ---------------------------------------------------------------------------


def test_des_to_factory_integration():
    """End-to-end: parsing + emitting against the real LevelGenerator (A1)."""
    pytest.importorskip("Nethax.minihax.level_generator")
    src = """\
MAZE: "mylevel", ' '
INIT_MAP: solidfill, ' '
GEOMETRY: center, center
MAP
.....
.....
.....
.....
.....
ENDMAP
REGION: (0,0,4,4), lit, "ordinary"
STAIR: (4,4), down
"""
    factory = dp.des_to_factory(src, w=20, h=10)
    out = factory(jax.random.PRNGKey(0))
    # Either an EnvState (real LG) or the adapter (fallback) — both
    # acceptable.  We just want no crash.
    assert out is not None


def test_parse_if_else_eval():
    src = """\
LEVEL: "x"
IF [100%] {
    MONSTER: ('d',"dog"),(1,1)
} ELSE {
    MONSTER: ('f',"cat"),(2,2)
}
"""
    ast = dp.parse_des(src)
    f = dp.compile_des(ast)
    lg = dp._MockLevelGenerator()
    f(lg, 42)
    monsters = [c for c in lg.calls if c[0] == "add_monster"]
    assert len(monsters) == 1
    # 100% chance → then branch always fires (dog).
    assert monsters[0][2].get("name") == "dog"

    src2 = src.replace("100%", "0%")
    ast2 = dp.parse_des(src2)
    f2 = dp.compile_des(ast2)
    lg2 = dp._MockLevelGenerator()
    f2(lg2, 42)
    monsters2 = [c for c in lg2.calls if c[0] == "add_monster"]
    assert len(monsters2) == 1
    assert monsters2[0][2].get("name") == "cat"
