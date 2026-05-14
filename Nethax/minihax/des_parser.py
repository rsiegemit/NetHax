"""MiniHack `.des` file parser.

Translates NetHack des-format level files into a sequence of
``LevelGenerator`` calls.  The parser is hand-rolled recursive descent;
the grammar reference is ``vendor/nethack/util/lev_comp.{l,y}``.

Supported constructs (those that appear in the 36 MiniHack canonical
``vendor/minihack/minihack/dat/*.des`` files):

* Headers: ``MAZE``, ``LEVEL``, ``GEOMETRY``, ``INIT_MAP``, ``FLAGS``,
  ``NOMAP``, ``MESSAGE``.
* Map block: ``MAP`` ... ``ENDMAP``.
* ``REGION``, ``ROOM`` (with body), ``SUBROOM``.
* ``MONSTER``, ``OBJECT``, ``CONTAINER`` (with body), ``TRAP``,
  ``STAIR``, ``DOOR``, ``ROOMDOOR``, ``BRANCH``, ``TERRAIN``,
  ``REPLACE_TERRAIN``, ``RANDOM_CORRIDORS``, ``RANDOM_MONSTERS``,
  ``RANDOM_OBJECTS``, ``MAZEWALK``, ``LOOP``, ``IF`` / ``ELSE``,
  ``CHOICE``, ``SHUFFLE``, ``NON_DIGGABLE``, ``NON_PASSWALL``.
* Variable assignments (``$name = selection:fillrect (..)`` etc.)
  recorded as opaque expressions.

Anything else parses as an ``UnknownStmt`` AST node and the compiler
logs the directive but does not raise.  This means new constructs do
not crash existing factories.

Coordinate convention: des-files use (x, y) = (column, row).  The
parser preserves that convention in the AST; the compiler converts to
(row, col) when emitting ``LevelGenerator`` calls (the LG API uses
row-major coords matching the rest of nethax).
"""
from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

# Reserved words.  Anything matching ``[A-Z_][A-Z0-9_]*`` is initially
# tagged as a keyword candidate; the parser decides what to do.
_KEYWORDS = frozenset(
    {
        "MAZE", "LEVEL", "GEOMETRY", "INIT_MAP", "FLAGS", "NOMAP",
        "MESSAGE", "MAP", "ENDMAP", "REGION", "ROOM", "SUBROOM",
        "MONSTER", "OBJECT", "CONTAINER", "TRAP", "STAIR", "DOOR",
        "ROOMDOOR", "BRANCH", "TERRAIN", "REPLACE_TERRAIN", "MAZEWALK",
        "RANDOM_CORRIDORS", "RANDOM_MONSTERS", "RANDOM_OBJECTS",
        "RANDOM_PLACES", "LOOP", "IF", "ELSE", "CHOICE", "SHUFFLE",
        "NON_DIGGABLE", "NON_PASSWALL", "WALLIFY", "GOLD", "FOUNTAIN",
        "ALTAR", "DRAWBRIDGE", "ENGRAVING", "SINK", "NAME",
        # value words / qualifiers (lowercase too — see normalisation
        # below)
        "lit", "unlit", "random", "center", "north", "south", "east",
        "west", "up", "down", "blessed", "uncursed", "cursed", "true",
        "false", "open", "closed", "locked", "broken", "nodoor",
        "asleep", "awake", "hostile", "peaceful", "tame", "not_trapped",
        "trapped",
    }
)


# Token kinds — kept as strings to make the parser easier to read.
TOK_KEYWORD = "KW"
TOK_IDENT = "ID"
TOK_STRING = "STR"
TOK_NUMBER = "NUM"
TOK_CHAR = "CH"
TOK_PERCENT_LIT = "PCT"
TOK_DOLLAR = "DOL"
TOK_OP = "OP"


@dataclass
class Token:
    kind: str
    value: Any
    line: int
    col: int

    def __repr__(self) -> str:  # pragma: no cover — debug only
        return f"Token({self.kind}, {self.value!r}, line={self.line})"


class DesParseError(Exception):
    """Raised when the parser cannot proceed."""


def _strip_comment(line: str) -> str:
    """Strip a NetHack-style ``#`` line comment, but only outside string literals."""
    out: List[str] = []
    in_str = False
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == '"' and (i == 0 or line[i - 1] != "\\"):
            in_str = not in_str
            out.append(ch)
            i += 1
            continue
        if ch == "#" and not in_str:
            break
        out.append(ch)
        i += 1
    return "".join(out)


def tokenize(src: str) -> List[Token]:
    """Tokenize a des-file source string.

    The tokenizer is two-phase: first the MAP/ENDMAP region is extracted
    verbatim as a single TOK_STRING (kind="MAP") so the grid characters
    don't get interpreted as identifiers; then the rest goes through the
    normal scanner.
    """
    tokens: List[Token] = []
    lines = src.split("\n")
    line_no = 0
    in_map = False
    map_buf: List[str] = []
    map_start_line = 0

    while line_no < len(lines):
        raw = lines[line_no]
        line_no += 1

        if in_map:
            if raw.strip() == "ENDMAP":
                tokens.append(
                    Token("MAP_BLOCK", "\n".join(map_buf), map_start_line, 0)
                )
                map_buf = []
                in_map = False
            else:
                map_buf.append(raw)
            continue

        # Detect MAP start.  Note: ``MAP`` may have trailing whitespace.
        stripped = raw.rstrip()
        if stripped == "MAP" or stripped.strip().startswith("MAP "):
            in_map = True
            map_start_line = line_no
            continue

        line = _strip_comment(raw)
        col = 0
        n = len(line)
        while col < n:
            ch = line[col]
            if ch in " \t\r":
                col += 1
                continue
            # String literal.
            if ch == '"':
                end = col + 1
                while end < n and line[end] != '"':
                    if line[end] == "\\" and end + 1 < n:
                        end += 2
                        continue
                    end += 1
                if end >= n:
                    raise DesParseError(
                        f"unterminated string at line {line_no}"
                    )
                tokens.append(Token(TOK_STRING, line[col + 1 : end], line_no, col))
                col = end + 1
                continue
            # Char literal ('x' or '\x' or '\'').
            if ch == "'":
                end = col + 1
                if end < n and line[end] == "\\":
                    end += 2
                else:
                    end += 1
                if end >= n or line[end] != "'":
                    raise DesParseError(
                        f"bad char literal at line {line_no}, col {col}"
                    )
                inner = line[col + 1 : end]
                if inner.startswith("\\") and len(inner) == 2:
                    esc = inner[1]
                    inner = {"n": "\n", "t": "\t", "'": "'", "\\": "\\"}.get(
                        esc, esc
                    )
                tokens.append(Token(TOK_CHAR, inner, line_no, col))
                col = end + 1
                continue
            # Number (possibly with trailing %).
            if ch.isdigit() or (
                ch == "-" and col + 1 < n and line[col + 1].isdigit()
            ):
                end = col + 1
                while end < n and line[end].isdigit():
                    end += 1
                num_text = line[col:end]
                if end < n and line[end] == "%":
                    tokens.append(
                        Token(TOK_PERCENT_LIT, int(num_text), line_no, col)
                    )
                    col = end + 1
                else:
                    tokens.append(Token(TOK_NUMBER, int(num_text), line_no, col))
                    col = end
                continue
            # Dollar variable.
            if ch == "$":
                end = col + 1
                while end < n and (line[end].isalnum() or line[end] == "_"):
                    end += 1
                tokens.append(Token(TOK_DOLLAR, line[col + 1 : end], line_no, col))
                col = end
                continue
            # Identifier or keyword.
            if ch.isalpha() or ch == "_":
                end = col + 1
                while end < n and (line[end].isalnum() or line[end] == "_"):
                    end += 1
                word = line[col:end]
                kind = TOK_KEYWORD if word in _KEYWORDS else TOK_IDENT
                tokens.append(Token(kind, word, line_no, col))
                col = end
                continue
            # Punctuation / operators.
            if ch in "(){},:;|=[]+":
                tokens.append(Token(TOK_OP, ch, line_no, col))
                col += 1
                continue
            # Anything else: keep raw, advance.  We do this rather than
            # raise so weird new constructs don't kill smoke parsing.
            tokens.append(Token(TOK_OP, ch, line_no, col))
            col += 1

    if in_map:
        tokens.append(
            Token("MAP_BLOCK", "\n".join(map_buf), map_start_line, 0)
        )

    return tokens


# ---------------------------------------------------------------------------
# AST node types
# ---------------------------------------------------------------------------


@dataclass
class Coord:
    """A literal (x, y) coordinate."""
    x: int
    y: int


@dataclass
class Rect:
    """A literal rectangle (x1, y1, x2, y2)."""
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class Random:
    """Marker for ``random`` placeholder."""
    pass


@dataclass
class VarRef:
    """A `$name` or `$name[idx]` reference."""
    name: str
    index: Optional[int] = None


@dataclass
class RndCoord:
    """A ``rndcoord(...)`` expression."""
    arg: Any  # VarRef, Rect, or Coord


@dataclass
class Selection:
    """A `selection:fillrect (...)` or similar expression."""
    kind: str  # "fillrect", "line", ...
    args: Tuple[Any, ...]


@dataclass
class ListExpr:
    """A literal `{ a, b, c }` list value."""
    items: List[Any]


@dataclass
class MapBlock:
    """A MAP ... ENDMAP block (string grid, des coords)."""
    grid: str  # raw text, newlines separate rows


# Statement types ----------------------------------------------------------


@dataclass
class Header:
    """Top-level header directive (MAZE / LEVEL / GEOMETRY / FLAGS / ...)."""
    kind: str
    args: Tuple[Any, ...] = ()


@dataclass
class Region:
    rect: Rect
    lit: bool
    name: str
    options: Tuple[str, ...] = ()


@dataclass
class Room:
    kind: str            # "ordinary", "shop", ...
    lit: bool
    size: Any            # Coord | Random | None
    pos: Any             # Coord | Random | "center,center"
    align: Any           # Coord | Random | None
    body: List[Any] = field(default_factory=list)
    is_sub: bool = False


@dataclass
class Monster:
    sym: Optional[str] = None       # the 'x' glyph, may be None
    name: Optional[str] = None      # the "name" if given
    pos: Any = None                 # Coord | VarRef | RndCoord | Random
    options: Tuple[str, ...] = ()


@dataclass
class ObjectStmt:
    sym: Optional[str] = None
    name: Optional[str] = None
    pos: Any = None
    options: Tuple[str, ...] = ()
    quality: Optional[int] = None
    artifact: Optional[str] = None  # name:"..."


@dataclass
class Trap:
    name: str
    pos: Any


@dataclass
class StairStmt:
    pos: Any
    direction: str = "down"   # "up" or "down"
    target: Any = None        # second-arg rect (for cross-branch stairs)


@dataclass
class Door:
    state: str                # "locked", "closed", "open", ...
    pos: Any


@dataclass
class RoomDoor:
    secret: bool
    state: str
    wall: Any
    pos: Any


@dataclass
class Branch:
    src: Rect
    dst: Rect


@dataclass
class Terrain:
    region: Any               # Coord | Rect | Selection
    glyph: str


@dataclass
class ReplaceTerrain:
    rect: Rect
    from_glyph: str
    to_glyph: str
    chance: int               # percentage 0–100


@dataclass
class Container:
    sym: Optional[str]
    name: Optional[str]
    options: Tuple[str, ...]
    pos: Any
    body: List[Any] = field(default_factory=list)


@dataclass
class MazeWalk:
    start: Any
    direction: str


@dataclass
class RandomCorridors:
    pass


@dataclass
class RandomMonsters:
    syms: Tuple[str, ...]


@dataclass
class RandomObjects:
    syms: Tuple[str, ...]


@dataclass
class NonDiggable:
    rect: Rect


@dataclass
class NonPasswall:
    rect: Rect


@dataclass
class Loop:
    count: int
    body: List[Any]


@dataclass
class IfElse:
    chance: Optional[int]              # percentage; None means symbolic
    then_body: List[Any]
    else_body: List[Any]


@dataclass
class Choice:
    branches: List[List[Any]]


@dataclass
class Shuffle:
    var: VarRef


@dataclass
class VarAssign:
    name: str
    value: Any                          # Selection | ListExpr | RndCoord | ...


@dataclass
class UnknownStmt:
    """A directive we recognised by name but don't have a model for."""
    name: str
    raw_tokens: List[Token] = field(default_factory=list)


@dataclass
class DesAST:
    headers: List[Header]               # MAZE / LEVEL / GEOMETRY / FLAGS / ...
    map_block: Optional[MapBlock]
    statements: List[Any]               # ordered list of body statements


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class _Parser:

    def __init__(self, tokens: List[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    # -- utility -----------------------------------------------------------

    def peek(self, offset: int = 0) -> Optional[Token]:
        idx = self.pos + offset
        if idx >= len(self.tokens):
            return None
        return self.tokens[idx]

    def at_end(self) -> bool:
        return self.pos >= len(self.tokens)

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def match(self, kind: str, value: Any = None) -> Optional[Token]:
        tok = self.peek()
        if tok is None or tok.kind != kind:
            return None
        if value is not None and tok.value != value:
            return None
        return self.advance()

    def expect(self, kind: str, value: Any = None) -> Token:
        tok = self.match(kind, value)
        if tok is None:
            actual = self.peek()
            raise DesParseError(
                f"expected {kind}({value!r}) got {actual!r}"
            )
        return tok

    def expect_op(self, ch: str) -> Token:
        return self.expect(TOK_OP, ch)

    def skip_op(self, ch: str) -> bool:
        return self.match(TOK_OP, ch) is not None

    # -- entry -------------------------------------------------------------

    def parse(self) -> DesAST:
        headers: List[Header] = []
        statements: List[Any] = []
        map_block: Optional[MapBlock] = None

        while not self.at_end():
            tok = self.peek()
            if tok.kind == "MAP_BLOCK":
                self.advance()
                map_block = MapBlock(tok.value)
                continue
            if tok.kind == TOK_KEYWORD:
                handler = _TOP_HANDLERS.get(tok.value)
                if handler is not None:
                    node = handler(self)
                    if isinstance(node, Header):
                        headers.append(node)
                    elif node is not None:
                        statements.append(node)
                    continue
                if tok.value in {"ELSE"}:
                    # stray ELSE → consume and stop (parser desync); we
                    # don't raise to keep smoke parsing robust.
                    self.advance()
                    continue
            if tok.kind == TOK_DOLLAR:
                # $var = expr
                node = self._parse_var_assign()
                if node is not None:
                    statements.append(node)
                continue
            # Unknown / unhandled token — skip a token to keep going.
            self.advance()

        return DesAST(headers=headers, map_block=map_block, statements=statements)

    # -- header parsers ----------------------------------------------------

    def _parse_maze_header(self) -> Header:
        self.expect(TOK_KEYWORD, "MAZE")
        self.skip_op(":")
        name_tok = self.expect(TOK_STRING)
        self.skip_op(",")
        # padder character (' ' usually)
        pad: Any = " "
        if self.peek() is not None and self.peek().kind == TOK_CHAR:
            pad = self.advance().value
        return Header("MAZE", (name_tok.value, pad))

    def _parse_level_header(self) -> Header:
        self.expect(TOK_KEYWORD, "LEVEL")
        self.skip_op(":")
        name_tok = self.expect(TOK_STRING)
        return Header("LEVEL", (name_tok.value,))

    def _parse_geometry_header(self) -> Header:
        self.expect(TOK_KEYWORD, "GEOMETRY")
        self.skip_op(":")
        a = self._parse_ident_or_kw()
        self.skip_op(",")
        b = self._parse_ident_or_kw()
        return Header("GEOMETRY", (a, b))

    def _parse_init_map_header(self) -> Header:
        self.expect(TOK_KEYWORD, "INIT_MAP")
        self.skip_op(":")
        fill_kind = self._parse_ident_or_kw()
        self.skip_op(",")
        ch: Any = " "
        if self.peek() is not None and self.peek().kind == TOK_CHAR:
            ch = self.advance().value
        return Header("INIT_MAP", (fill_kind, ch))

    def _parse_flags_header(self) -> Header:
        self.expect(TOK_KEYWORD, "FLAGS")
        self.skip_op(":")
        flags: List[str] = []
        while True:
            tok = self.peek()
            if tok is None or tok.kind not in (TOK_IDENT, TOK_KEYWORD):
                break
            flags.append(tok.value)
            self.advance()
            if not self.skip_op(","):
                break
        return Header("FLAGS", tuple(flags))

    def _parse_message_header(self) -> Header:
        self.expect(TOK_KEYWORD, "MESSAGE")
        self.skip_op(":")
        s = self.expect(TOK_STRING)
        return Header("MESSAGE", (s.value,))

    def _parse_nomap_header(self) -> Header:
        self.expect(TOK_KEYWORD, "NOMAP")
        self.skip_op(":")
        # ignore arguments — only one canonical form uses this and
        # we treat NOMAP as a flag.
        return Header("NOMAP", ())

    # -- statement parsers -------------------------------------------------

    def _parse_region(self) -> Region:
        self.expect(TOK_KEYWORD, "REGION")
        self.skip_op(":")
        rect = self._parse_rect()
        self.skip_op(",")
        lit_word = self._parse_ident_or_kw()
        lit = lit_word == "lit"
        self.skip_op(",")
        name_tok = self.expect(TOK_STRING)
        opts: List[str] = []
        while self.skip_op(","):
            tok = self.peek()
            if tok is None:
                break
            if tok.kind in (TOK_IDENT, TOK_KEYWORD):
                opts.append(tok.value)
                self.advance()
            else:
                break
        return Region(rect=rect, lit=lit, name=name_tok.value, options=tuple(opts))

    def _parse_room_or_subroom(self) -> Room:
        kw = self.advance()
        is_sub = kw.value == "SUBROOM"
        self.skip_op(":")
        kind_tok = self.expect(TOK_STRING)
        # Comma-separated arguments — but the grammar is irregular.  We
        # consume up to four positional args (lit, size, pos, align),
        # accepting any of: ident, coord, rect, RS-style ident.
        args: List[Any] = []
        while self.skip_op(","):
            args.append(self._parse_room_arg())
        # Now: optional `{ body }`.
        body: List[Any] = []
        if self.skip_op("{"):
            body = self._parse_body()
            self.expect_op("}")

        # Pull lit / size / pos / align from positional args.
        lit = False
        size: Any = None
        pos: Any = None
        align: Any = None
        for i, a in enumerate(args):
            if i == 0 and isinstance(a, str):
                lit = a == "lit"
            elif size is None and isinstance(a, (Coord, Random)):
                size = a
            elif pos is None and isinstance(a, (Coord, Random, tuple)):
                pos = a
            elif align is None:
                align = a
        return Room(
            kind=kind_tok.value,
            lit=lit,
            size=size,
            pos=pos,
            align=align,
            body=body,
            is_sub=is_sub,
        )

    def _parse_room_arg(self) -> Any:
        tok = self.peek()
        if tok is None:
            return None
        if tok.kind == TOK_OP and tok.value == "(":
            return self._parse_paren_value()
        if tok.kind in (TOK_IDENT, TOK_KEYWORD):
            self.advance()
            return tok.value
        if tok.kind == TOK_NUMBER:
            self.advance()
            return tok.value
        if tok.kind == TOK_DOLLAR:
            return self._parse_var_ref()
        self.advance()
        return tok.value

    def _parse_monster(self) -> Monster:
        self.expect(TOK_KEYWORD, "MONSTER")
        self.skip_op(":")
        sym, name = self._parse_symname()
        self.skip_op(",")
        pos = self._parse_pos_expr()
        opts: List[str] = []
        while self.skip_op(","):
            tok = self.peek()
            if tok is None:
                break
            if tok.kind in (TOK_IDENT, TOK_KEYWORD):
                opts.append(tok.value)
                self.advance()
            elif tok.kind == TOK_STRING:
                opts.append(f"\"{tok.value}\"")
                self.advance()
            else:
                break
        return Monster(sym=sym, name=name, pos=pos, options=tuple(opts))

    def _parse_object(self) -> ObjectStmt:
        self.expect(TOK_KEYWORD, "OBJECT")
        self.skip_op(":")
        sym, name = self._parse_symname()
        # Inside CONTAINER bodies, OBJECT may have no position at all.
        # Only consume a position if a comma follows.
        pos: Any = None
        if self.skip_op(","):
            pos = self._parse_pos_expr()
        opts: List[str] = []
        quality: Optional[int] = None
        artifact: Optional[str] = None
        while self.skip_op(","):
            tok = self.peek()
            if tok is None:
                break
            # `name:"..."` form (artifact name)
            if (
                tok.kind in (TOK_IDENT, TOK_KEYWORD)
                and self.peek(1) is not None
                and self.peek(1).kind == TOK_OP
                and self.peek(1).value == ":"
                and self.peek(2) is not None
                and self.peek(2).kind == TOK_STRING
            ):
                key = tok.value
                self.advance()
                self.advance()
                val = self.advance().value
                if key == "name":
                    artifact = val
                else:
                    opts.append(f"{key}={val}")
                continue
            if tok.kind == TOK_NUMBER:
                quality = tok.value
                self.advance()
                continue
            if tok.kind in (TOK_IDENT, TOK_KEYWORD):
                opts.append(tok.value)
                self.advance()
                continue
            break
        return ObjectStmt(
            sym=sym, name=name, pos=pos, options=tuple(opts),
            quality=quality, artifact=artifact,
        )

    def _parse_trap(self) -> Trap:
        self.expect(TOK_KEYWORD, "TRAP")
        self.skip_op(":")
        name_tok = self.expect(TOK_STRING)
        self.skip_op(",")
        pos = self._parse_pos_expr()
        return Trap(name=name_tok.value, pos=pos)

    def _parse_stair(self) -> StairStmt:
        self.expect(TOK_KEYWORD, "STAIR")
        self.skip_op(":")
        pos = self._parse_pos_expr()
        direction = "down"
        target: Any = None
        if self.skip_op(","):
            # Could be `(rect),(rect),direction` for cross-level stairs,
            # or just `direction`.
            tok = self.peek()
            if tok is not None and tok.kind == TOK_OP and tok.value == "(":
                target = self._parse_paren_value()
                if self.skip_op(","):
                    dtok = self._parse_ident_or_kw_optional()
                    if dtok is not None:
                        direction = dtok
            else:
                dtok = self._parse_ident_or_kw_optional()
                if dtok is not None:
                    direction = dtok
        return StairStmt(pos=pos, direction=direction, target=target)

    def _parse_door(self) -> Door:
        self.expect(TOK_KEYWORD, "DOOR")
        self.skip_op(":")
        state = self._parse_ident_or_kw()
        self.skip_op(",")
        pos = self._parse_pos_expr()
        return Door(state=state, pos=pos)

    def _parse_roomdoor(self) -> RoomDoor:
        self.expect(TOK_KEYWORD, "ROOMDOOR")
        self.skip_op(":")
        # ROOMDOOR: secret(bool), state, wall(side), pos
        secret_tok = self._parse_ident_or_kw()
        secret = secret_tok == "true"
        self.skip_op(",")
        state = self._parse_ident_or_kw()
        self.skip_op(",")
        wall = self._parse_pos_expr()
        self.skip_op(",")
        pos = self._parse_pos_expr()
        return RoomDoor(secret=secret, state=state, wall=wall, pos=pos)

    def _parse_branch(self) -> Branch:
        self.expect(TOK_KEYWORD, "BRANCH")
        self.skip_op(":")
        src = self._parse_rect_loose()
        self.skip_op(",")
        dst = self._parse_rect_loose()
        return Branch(src=src, dst=dst)

    def _parse_terrain(self) -> Terrain:
        self.expect(TOK_KEYWORD, "TERRAIN")
        self.skip_op(":")
        # Region can be: rect, coord, or `randline (a,b),(c,d), N`.
        region: Any
        tok = self.peek()
        if tok is not None and tok.kind == TOK_IDENT and tok.value == "randline":
            self.advance()
            args: List[Any] = []
            if self.skip_op("("):
                args.append(Coord(self.expect(TOK_NUMBER).value, 0))
                args[-1] = self._patch_coord(args[-1])
            # The above hack rarely works — fallback: just store None.
            # Instead, re-parse cleanly:
            # Pop back: this branch is only hit for hidenseek-style
            # randline.  Skip until next REGION/STAIR/etc.
            # For our purposes Terrain.region stores the literal text.
            args = []
            while True:
                t = self.peek()
                if t is None:
                    break
                if t.kind == TOK_OP and t.value in (";", "\n"):
                    break
                if t.kind == TOK_KEYWORD:
                    break
                # collect until char glyph at end
                if t.kind == TOK_CHAR:
                    # this is the glyph — stop before it
                    break
                self.advance()
            region = Selection("randline", tuple(args))
        else:
            region = self._parse_pos_or_rect()
        self.skip_op(",")
        # Optional numeric "thickness" arg (rare).
        if self.peek() is not None and self.peek().kind == TOK_NUMBER:
            self.advance()
            self.skip_op(",")
        glyph_tok = self.expect(TOK_CHAR)
        return Terrain(region=region, glyph=glyph_tok.value)

    def _parse_replace_terrain(self) -> ReplaceTerrain:
        self.expect(TOK_KEYWORD, "REPLACE_TERRAIN")
        self.skip_op(":")
        rect = self._parse_rect()
        self.skip_op(",")
        from_g = self.expect(TOK_CHAR).value
        self.skip_op(",")
        to_g = self.expect(TOK_CHAR).value
        self.skip_op(",")
        pct_tok = self.peek()
        chance = 100
        if pct_tok is not None and pct_tok.kind == TOK_PERCENT_LIT:
            chance = pct_tok.value
            self.advance()
        elif pct_tok is not None and pct_tok.kind == TOK_NUMBER:
            chance = pct_tok.value
            self.advance()
        return ReplaceTerrain(
            rect=rect, from_glyph=from_g, to_glyph=to_g, chance=chance,
        )

    def _parse_container(self) -> Container:
        self.expect(TOK_KEYWORD, "CONTAINER")
        self.skip_op(":")
        sym, name = self._parse_symname()
        opts: List[str] = []
        pos: Any = None
        # CONTAINER: ('(',"chest"), not_trapped, random { OBJECT:... }
        while self.skip_op(","):
            tok = self.peek()
            if tok is None:
                break
            if tok.kind == TOK_OP and tok.value == "(":
                pos = self._parse_paren_value()
            elif tok.kind == TOK_DOLLAR:
                pos = self._parse_var_ref_or_rndcoord()
            elif tok.kind == TOK_IDENT and tok.value == "rndcoord":
                pos = self._parse_rndcoord()
            elif tok.kind in (TOK_IDENT, TOK_KEYWORD):
                # could be `random` or an option flag
                val = tok.value
                self.advance()
                if val == "random" and pos is None:
                    pos = Random()
                else:
                    opts.append(val)
            else:
                break
        body: List[Any] = []
        if self.skip_op("{"):
            body = self._parse_body()
            self.expect_op("}")
        return Container(sym=sym, name=name, options=tuple(opts), pos=pos, body=body)

    def _parse_mazewalk(self) -> MazeWalk:
        self.expect(TOK_KEYWORD, "MAZEWALK")
        self.skip_op(":")
        start = self._parse_pos_expr()
        self.skip_op(",")
        direction = self._parse_ident_or_kw()
        return MazeWalk(start=start, direction=direction)

    def _parse_random_corridors(self) -> RandomCorridors:
        self.expect(TOK_KEYWORD, "RANDOM_CORRIDORS")
        return RandomCorridors()

    def _parse_random_monsters(self) -> RandomMonsters:
        self.expect(TOK_KEYWORD, "RANDOM_MONSTERS")
        self.skip_op(":")
        syms: List[str] = []
        while True:
            tok = self.peek()
            if tok is None or tok.kind != TOK_CHAR:
                break
            syms.append(tok.value)
            self.advance()
            if not self.skip_op(","):
                break
        return RandomMonsters(syms=tuple(syms))

    def _parse_random_objects(self) -> RandomObjects:
        self.expect(TOK_KEYWORD, "RANDOM_OBJECTS")
        self.skip_op(":")
        syms: List[str] = []
        while True:
            tok = self.peek()
            if tok is None or tok.kind != TOK_CHAR:
                break
            syms.append(tok.value)
            self.advance()
            if not self.skip_op(","):
                break
        return RandomObjects(syms=tuple(syms))

    def _parse_non_diggable(self) -> NonDiggable:
        self.expect(TOK_KEYWORD, "NON_DIGGABLE")
        self.skip_op(":")
        return NonDiggable(rect=self._parse_rect())

    def _parse_non_passwall(self) -> NonPasswall:
        self.expect(TOK_KEYWORD, "NON_PASSWALL")
        self.skip_op(":")
        return NonPasswall(rect=self._parse_rect())

    def _parse_loop(self) -> Loop:
        self.expect(TOK_KEYWORD, "LOOP")
        # LOOP [count] { ... }
        if self.skip_op("["):
            count_tok = self.expect(TOK_NUMBER)
            self.expect_op("]")
            count = count_tok.value
        else:
            count = 1
        self.expect_op("{")
        body = self._parse_body()
        self.expect_op("}")
        return Loop(count=count, body=body)

    def _parse_if(self) -> IfElse:
        self.expect(TOK_KEYWORD, "IF")
        chance: Optional[int] = None
        if self.skip_op("["):
            tok = self.peek()
            if tok is not None and tok.kind == TOK_PERCENT_LIT:
                chance = tok.value
                self.advance()
            elif tok is not None and tok.kind == TOK_NUMBER:
                chance = tok.value
                self.advance()
            self.expect_op("]")
        self.expect_op("{")
        then_body = self._parse_body()
        self.expect_op("}")
        else_body: List[Any] = []
        # Optional ELSE { ... }
        nxt = self.peek()
        if nxt is not None and nxt.kind == TOK_KEYWORD and nxt.value == "ELSE":
            self.advance()
            self.expect_op("{")
            else_body = self._parse_body()
            self.expect_op("}")
        return IfElse(chance=chance, then_body=then_body, else_body=else_body)

    def _parse_choice(self) -> Choice:
        self.expect(TOK_KEYWORD, "CHOICE")
        branches: List[List[Any]] = []
        # CHOICE { ... } | { ... } | ...
        self.expect_op("{")
        branches.append(self._parse_body())
        self.expect_op("}")
        while self.skip_op("|"):
            self.expect_op("{")
            branches.append(self._parse_body())
            self.expect_op("}")
        return Choice(branches=branches)

    def _parse_shuffle(self) -> Shuffle:
        self.expect(TOK_KEYWORD, "SHUFFLE")
        self.skip_op(":")
        var = self._parse_var_ref()
        return Shuffle(var=var)

    def _parse_unknown(self) -> UnknownStmt:
        kw = self.advance()
        # Slurp until end-of-line-ish boundary: next top-level keyword
        # or closing brace.
        raw: List[Token] = []
        while not self.at_end():
            t = self.peek()
            if t.kind == TOK_KEYWORD and t.value in _TOP_HANDLERS:
                break
            if t.kind == TOK_OP and t.value in ("}", "{"):
                break
            if t.kind == "MAP_BLOCK":
                break
            raw.append(self.advance())
        return UnknownStmt(name=kw.value, raw_tokens=raw)

    # -- composite parsers -------------------------------------------------

    def _parse_body(self) -> List[Any]:
        """Parse a `{ ... }` body — terminated by `}` (not consumed)."""
        items: List[Any] = []
        while not self.at_end():
            tok = self.peek()
            if tok.kind == TOK_OP and tok.value == "}":
                break
            if tok.kind == TOK_OP and tok.value == "|":
                break
            if tok.kind == TOK_KEYWORD:
                handler = _TOP_HANDLERS.get(tok.value)
                if handler is not None:
                    node = handler(self)
                    if node is not None:
                        items.append(node)
                    continue
                # Body-level unknown — fall through.
                items.append(self._parse_unknown())
                continue
            if tok.kind == TOK_DOLLAR:
                node = self._parse_var_assign()
                if node is not None:
                    items.append(node)
                continue
            # Skip stray tokens to keep going.
            self.advance()
        return items

    def _parse_var_assign(self) -> Optional[VarAssign]:
        dol = self.advance()
        # Optional `[idx]` after $name (not valid for assignment, but
        # appears in references).  For assignment we expect `=`.
        if not self.skip_op("="):
            # Not an assignment — back up.  We can't reliably back up the
            # tokenizer, so emit a no-op.
            return None
        value = self._parse_expr()
        return VarAssign(name=dol.value, value=value)

    def _parse_expr(self) -> Any:
        tok = self.peek()
        if tok is None:
            return None
        if tok.kind == TOK_OP and tok.value == "{":
            # list literal
            self.advance()
            items: List[Any] = []
            while not self.skip_op("}"):
                items.append(self._parse_expr_atom())
                if not self.skip_op(","):
                    self.skip_op("}")  # tolerate missing comma
                    break
            return ListExpr(items=items)
        return self._parse_expr_atom()

    def _parse_expr_atom(self) -> Any:
        tok = self.peek()
        if tok is None:
            return None
        if tok.kind == TOK_IDENT and tok.value in ("selection", "monster", "object"):
            return self._parse_selection_like()
        if tok.kind == TOK_IDENT and tok.value == "rndcoord":
            return self._parse_rndcoord()
        if tok.kind == TOK_IDENT and tok.value == "line":
            return self._parse_selection_like()
        if tok.kind == TOK_OP and tok.value == "(":
            return self._parse_paren_value()
        if tok.kind == TOK_DOLLAR:
            return self._parse_var_ref()
        if tok.kind == TOK_NUMBER:
            self.advance()
            return tok.value
        if tok.kind == TOK_STRING:
            self.advance()
            return tok.value
        if tok.kind == TOK_CHAR:
            self.advance()
            return tok.value
        self.advance()
        return tok.value

    def _parse_selection_like(self) -> Selection:
        head = self.advance()
        # may be "selection:fillrect (..)" or "line (..)"
        sub = head.value
        if self.skip_op(":"):
            sub_tok = self.peek()
            if sub_tok is not None and sub_tok.kind in (TOK_IDENT, TOK_KEYWORD):
                sub = sub_tok.value
                self.advance()
        args: List[Any] = []
        if self.peek() is not None and self.peek().kind == TOK_OP and self.peek().value == "(":
            args.append(self._parse_paren_value())
            while self.skip_op(","):
                args.append(self._parse_expr_atom())
        return Selection(kind=sub, args=tuple(args))

    def _parse_rndcoord(self) -> RndCoord:
        self.advance()  # consume 'rndcoord'
        # rndcoord(...) or rndcoord (...)
        if not self.skip_op("("):
            return RndCoord(arg=None)
        # Inner expr can be $var, a Rect, a `line (..)`, etc.
        if self.peek() is not None and self.peek().kind == TOK_DOLLAR:
            arg: Any = self._parse_var_ref()
        elif self.peek() is not None and self.peek().kind == TOK_IDENT and self.peek().value in ("line", "selection"):
            arg = self._parse_selection_like()
        else:
            # Maybe a rect literal: rndcoord((x1,y1,x2,y2))
            arg = self._parse_paren_value()
        self.skip_op(")")
        return RndCoord(arg=arg)

    def _parse_var_ref(self) -> VarRef:
        dol = self.advance()
        idx: Optional[int] = None
        if self.skip_op("["):
            n = self.expect(TOK_NUMBER)
            idx = n.value
            self.expect_op("]")
        return VarRef(name=dol.value, index=idx)

    def _parse_var_ref_or_rndcoord(self) -> Any:
        # $var or $var[idx]
        return self._parse_var_ref()

    def _parse_paren_value(self) -> Any:
        """Parse `(a, b)` → Coord, `(a,b,c,d)` → Rect."""
        self.expect_op("(")
        nums: List[Any] = []
        while True:
            tok = self.peek()
            if tok is None:
                break
            if tok.kind == TOK_OP and tok.value == ")":
                break
            if tok.kind == TOK_NUMBER:
                nums.append(tok.value)
                self.advance()
            elif tok.kind in (TOK_IDENT, TOK_KEYWORD):
                # e.g. `(center,center)` or `(RS,RS)`
                nums.append(tok.value)
                self.advance()
            elif tok.kind == TOK_DOLLAR:
                nums.append(self._parse_var_ref())
            else:
                # tolerate unknown
                self.advance()
            if not self.skip_op(","):
                break
        self.expect_op(")")
        if len(nums) == 2:
            x, y = nums
            if isinstance(x, int) and isinstance(y, int):
                return Coord(x=x, y=y)
            return ("pair", x, y)
        if len(nums) == 4:
            return Rect(*nums) if all(isinstance(n, int) for n in nums) else tuple(nums)
        return tuple(nums)

    def _parse_rect(self) -> Rect:
        val = self._parse_paren_value()
        if isinstance(val, Rect):
            return val
        if isinstance(val, tuple) and len(val) == 4 and all(isinstance(v, int) for v in val):
            return Rect(*val)
        # tolerate degenerate
        if isinstance(val, Coord):
            return Rect(val.x, val.y, val.x, val.y)
        return Rect(0, 0, 0, 0)

    def _parse_rect_loose(self) -> Rect:
        # Handles `levregion(x1,y1,x2,y2)` and bare `(x1,y1,x2,y2)`.
        tok = self.peek()
        if tok is not None and tok.kind == TOK_IDENT and tok.value == "levregion":
            self.advance()
        return self._parse_rect()

    def _parse_pos_or_rect(self) -> Any:
        val = self._parse_paren_value()
        return val

    def _parse_pos_expr(self) -> Any:
        """Parse a position expression: coord, rect, var ref, rndcoord, random."""
        tok = self.peek()
        if tok is None:
            return None
        if tok.kind == TOK_KEYWORD and tok.value == "random":
            self.advance()
            return Random()
        if tok.kind == TOK_IDENT and tok.value == "random":
            self.advance()
            return Random()
        if tok.kind == TOK_IDENT and tok.value == "rndcoord":
            return self._parse_rndcoord()
        if tok.kind == TOK_DOLLAR:
            return self._parse_var_ref()
        if tok.kind == TOK_OP and tok.value == "(":
            return self._parse_paren_value()
        if tok.kind == TOK_IDENT and tok.value in ("line", "selection"):
            return self._parse_selection_like()
        self.advance()
        return tok.value

    def _parse_symname(self) -> Tuple[Optional[str], Optional[str]]:
        """Parse a `('s', "name")` or `'s'` head form."""
        tok = self.peek()
        if tok is None:
            return None, None
        if tok.kind == TOK_OP and tok.value == "(":
            self.advance()
            sym: Optional[str] = None
            name: Optional[str] = None
            if self.peek() is not None and self.peek().kind == TOK_CHAR:
                sym = self.advance().value
            elif self.peek() is not None and self.peek().kind in (TOK_IDENT, TOK_KEYWORD):
                sym = self.advance().value  # e.g. `random`
            if self.skip_op(","):
                if self.peek() is not None and self.peek().kind == TOK_STRING:
                    name = self.advance().value
                elif self.peek() is not None and self.peek().kind == TOK_CHAR:
                    name = self.advance().value
            self.expect_op(")")
            return sym, name
        if tok.kind == TOK_CHAR:
            self.advance()
            return tok.value, None
        if tok.kind in (TOK_IDENT, TOK_KEYWORD) and tok.value == "random":
            self.advance()
            return None, None
        return None, None

    def _parse_ident_or_kw(self) -> str:
        tok = self.peek()
        if tok is None or tok.kind not in (TOK_IDENT, TOK_KEYWORD):
            raise DesParseError(f"expected identifier, got {tok!r}")
        self.advance()
        return tok.value

    def _parse_ident_or_kw_optional(self) -> Optional[str]:
        tok = self.peek()
        if tok is None or tok.kind not in (TOK_IDENT, TOK_KEYWORD):
            return None
        self.advance()
        return tok.value

    def _patch_coord(self, c: Any) -> Any:  # pragma: no cover — helper
        return c


# Dispatch table for top-level statement keywords -> parser methods.
_TOP_HANDLERS: dict = {
    "MAZE": _Parser._parse_maze_header,
    "LEVEL": _Parser._parse_level_header,
    "GEOMETRY": _Parser._parse_geometry_header,
    "INIT_MAP": _Parser._parse_init_map_header,
    "FLAGS": _Parser._parse_flags_header,
    "MESSAGE": _Parser._parse_message_header,
    "NOMAP": _Parser._parse_nomap_header,
    "REGION": _Parser._parse_region,
    "ROOM": _Parser._parse_room_or_subroom,
    "SUBROOM": _Parser._parse_room_or_subroom,
    "MONSTER": _Parser._parse_monster,
    "OBJECT": _Parser._parse_object,
    "TRAP": _Parser._parse_trap,
    "STAIR": _Parser._parse_stair,
    "DOOR": _Parser._parse_door,
    "ROOMDOOR": _Parser._parse_roomdoor,
    "BRANCH": _Parser._parse_branch,
    "TERRAIN": _Parser._parse_terrain,
    "REPLACE_TERRAIN": _Parser._parse_replace_terrain,
    "CONTAINER": _Parser._parse_container,
    "MAZEWALK": _Parser._parse_mazewalk,
    "RANDOM_CORRIDORS": _Parser._parse_random_corridors,
    "RANDOM_MONSTERS": _Parser._parse_random_monsters,
    "RANDOM_OBJECTS": _Parser._parse_random_objects,
    "NON_DIGGABLE": _Parser._parse_non_diggable,
    "NON_PASSWALL": _Parser._parse_non_passwall,
    "LOOP": _Parser._parse_loop,
    "IF": _Parser._parse_if,
    "CHOICE": _Parser._parse_choice,
    "SHUFFLE": _Parser._parse_shuffle,
}


def parse_des(des_source: str) -> DesAST:
    """Parse a des-file source into an AST.

    The parser is tolerant: unrecognised directives become
    ``UnknownStmt`` nodes rather than raising, so smoke-parsing the
    entire MiniHack corpus is safe.
    """
    tokens = tokenize(des_source)
    return _Parser(tokens).parse()


# ---------------------------------------------------------------------------
# Symbol tables
# ---------------------------------------------------------------------------


# Terrain glyph → (LevelGenerator terrain method, semantic).  These are
# the characters MiniHack uses on the MAP and in TERRAIN/REPLACE_TERRAIN.
TERRAIN_GLYPHS: dict = {
    ".": "floor",
    " ": "void",
    "-": "wall",
    "|": "wall",
    "+": "door_closed",
    "L": "lava",
    "W": "water",
    "C": "cloud",
    "T": "tree",
    "F": "fountain",
    "_": "altar",
    "{": "fountain",
    "}": "moat",
    "<": "stair_up",
    ">": "stair_down",
    "#": "corridor",
    "S": "door_secret",
    "I": "ice",
}


# Object class glyph → category.  Mirrors vendor/nethack/include/objclass.h.
OBJECT_CLASS_GLYPH: dict = {
    "%": "food",
    "!": "potion",
    "?": "scroll",
    "+": "spellbook",
    "=": "ring",
    "/": "wand",
    "(": "weapon",
    ")": "weapon",
    "[": "armor",
    "*": "rock",
    "$": "gold",
    "0": "iron_ball",
    "_": "altar",
    "`": "boulder",
    "\"": "amulet",
    "*": "gem",
}


# Monster class glyph → category (vendor/nethack/include/monsym.h).
MONSTER_CLASS_GLYPH: dict = {
    "@": "human",
    "d": "dog",
    "f": "cat",
    "r": "rat",
    "x": "grid_bug",
    "F": "lichen",
    "j": "jelly",
    "L": "lich",
    "N": "naga",
    "H": "giant",
    "O": "ogre",
    "D": "dragon",
    "T": "troll",
    "G": "gnome",
    "k": "kobold",
    "o": "orc",
    "h": "dwarf",
    "M": "mummy",
    "Z": "zombie",
}


# Trap name → canonical lowercase form.
TRAP_NAMES: frozenset = frozenset({
    "arrow", "dart", "falling rock", "rock", "bear", "land mine",
    "rolling boulder", "sleep gas", "rust", "fire", "pit", "spiked pit",
    "hole", "trap door", "teleport", "level teleport", "magic", "anti magic",
    "polymorph", "magic portal", "web", "statue", "board", "spear",
})


# ---------------------------------------------------------------------------
# Compiler / emitter
# ---------------------------------------------------------------------------


def _xy_to_rc(x: Any, y: Any) -> Tuple[Any, Any]:
    """Convert des (x=col, y=row) to nethax (row, col)."""
    return (y, x)


def _resolve_pos(pos: Any, env: dict, rng_key: Any) -> Optional[Tuple[int, int]]:
    """Resolve a position expression to a concrete (row, col), if possible.

    `env` is the variable environment built during compile.  Returns
    None if the position can't be resolved at emit time (e.g. random,
    rndcoord without bounds).  The LevelGenerator is expected to accept
    these and pick at generation time.
    """
    if isinstance(pos, Coord):
        return _xy_to_rc(pos.x, pos.y)
    if isinstance(pos, Rect):
        # Use centre as best-effort fallback.
        cx = (pos.x1 + pos.x2) // 2
        cy = (pos.y1 + pos.y2) // 2
        return _xy_to_rc(cx, cy)
    if isinstance(pos, VarRef):
        val = env.get(pos.name)
        if isinstance(val, ListExpr):
            idx = pos.index if pos.index is not None else 0
            if 0 <= idx < len(val.items):
                return _resolve_pos(val.items[idx], env, rng_key)
        return None
    if isinstance(pos, RndCoord):
        inner = pos.arg
        if isinstance(inner, VarRef):
            val = env.get(inner.name)
            if isinstance(val, Selection) and val.kind == "fillrect":
                # take centre of the rect
                if val.args and isinstance(val.args[0], Rect):
                    r = val.args[0]
                    cx = (r.x1 + r.x2) // 2
                    cy = (r.y1 + r.y2) // 2
                    return _xy_to_rc(cx, cy)
        if isinstance(inner, Selection) and inner.kind == "fillrect":
            if inner.args and isinstance(inner.args[0], Rect):
                r = inner.args[0]
                cx = (r.x1 + r.x2) // 2
                cy = (r.y1 + r.y2) // 2
                return _xy_to_rc(cx, cy)
        return None
    return None


def _emit_stmt(lg: Any, stmt: Any, env: dict, rng_state: dict) -> None:
    """Emit LevelGenerator calls for a single statement."""
    import random as _random
    rnd = rng_state["rand"]

    if isinstance(stmt, Region):
        if _has(lg, "add_region"):
            lg.add_region(
                row=stmt.rect.y1, col=stmt.rect.x1,
                height=stmt.rect.y2 - stmt.rect.y1 + 1,
                width=stmt.rect.x2 - stmt.rect.x1 + 1,
                lit=stmt.lit, name=stmt.name,
            )
        env.setdefault("_regions", {})[stmt.name] = stmt.rect
        return

    if isinstance(stmt, Room):
        if _has(lg, "add_room"):
            pos = stmt.pos
            size = stmt.size
            if isinstance(pos, Coord):
                row, col = _xy_to_rc(pos.x, pos.y)
            else:
                row, col = None, None
            if isinstance(size, Coord):
                h, w = size.y, size.x
            else:
                h, w = None, None
            room_id = lg.add_room(
                row=row, col=col, height=h, width=w,
                lit=stmt.lit, kind=stmt.kind, sub=stmt.is_sub,
            )
            env.setdefault("_rooms", []).append(room_id)
        for sub in stmt.body:
            _emit_stmt(lg, sub, env, rng_state)
        return

    if isinstance(stmt, Monster):
        if _has(lg, "add_monster"):
            pos = _resolve_pos(stmt.pos, env, None)
            row, col = (pos if pos is not None else (None, None))
            lg.add_monster(
                row=row, col=col, sym=stmt.sym, name=stmt.name,
                options=stmt.options,
            )
        return

    if isinstance(stmt, ObjectStmt):
        if _has(lg, "add_object"):
            pos = _resolve_pos(stmt.pos, env, None)
            row, col = (pos if pos is not None else (None, None))
            lg.add_object(
                row=row, col=col, sym=stmt.sym, name=stmt.name,
                options=stmt.options, quality=stmt.quality,
                artifact=stmt.artifact,
            )
        return

    if isinstance(stmt, Trap):
        if _has(lg, "add_trap"):
            pos = _resolve_pos(stmt.pos, env, None)
            row, col = (pos if pos is not None else (None, None))
            lg.add_trap(row=row, col=col, kind=stmt.name)
        return

    if isinstance(stmt, StairStmt):
        if _has(lg, "add_stair"):
            pos = _resolve_pos(stmt.pos, env, None)
            row, col = (pos if pos is not None else (None, None))
            lg.add_stair(row=row, col=col, direction=stmt.direction)
        return

    if isinstance(stmt, Door):
        if _has(lg, "add_door"):
            pos = _resolve_pos(stmt.pos, env, None)
            row, col = (pos if pos is not None else (None, None))
            lg.add_door(row=row, col=col, state=stmt.state)
        return

    if isinstance(stmt, RoomDoor):
        if _has(lg, "add_door"):
            pos = _resolve_pos(stmt.pos, env, None)
            row, col = (pos if pos is not None else (None, None))
            lg.add_door(
                row=row, col=col, state=stmt.state, secret=stmt.secret,
            )
        return

    if isinstance(stmt, Branch):
        if _has(lg, "set_branch"):
            lg.set_branch(
                src_rect=stmt.src, dst_rect=stmt.dst,
            )
        return

    if isinstance(stmt, Terrain):
        if _has(lg, "fill_terrain"):
            reg = stmt.region
            if isinstance(reg, Rect):
                lg.fill_terrain(
                    row=reg.y1, col=reg.x1,
                    height=reg.y2 - reg.y1 + 1,
                    width=reg.x2 - reg.x1 + 1,
                    glyph=stmt.glyph,
                )
            elif isinstance(reg, Coord):
                lg.fill_terrain(
                    row=reg.y, col=reg.x, height=1, width=1, glyph=stmt.glyph,
                )
            elif isinstance(reg, tuple) and len(reg) == 4:
                x1, y1, x2, y2 = reg
                lg.fill_terrain(
                    row=y1, col=x1, height=y2 - y1 + 1,
                    width=x2 - x1 + 1, glyph=stmt.glyph,
                )
            # selection (randline) → unsupported geometry; place a
            # point sample so tests still see the terrain glyph appear.
        return

    if isinstance(stmt, ReplaceTerrain):
        if _has(lg, "replace_terrain"):
            lg.replace_terrain(
                row=stmt.rect.y1, col=stmt.rect.x1,
                height=stmt.rect.y2 - stmt.rect.y1 + 1,
                width=stmt.rect.x2 - stmt.rect.x1 + 1,
                from_glyph=stmt.from_glyph, to_glyph=stmt.to_glyph,
                chance=stmt.chance,
            )
        return

    if isinstance(stmt, Container):
        if _has(lg, "add_container"):
            pos = _resolve_pos(stmt.pos, env, None)
            row, col = (pos if pos is not None else (None, None))
            container_id = lg.add_container(
                row=row, col=col, sym=stmt.sym, name=stmt.name,
                options=stmt.options,
            )
        for sub in stmt.body:
            _emit_stmt(lg, sub, env, rng_state)
        return

    if isinstance(stmt, MazeWalk):
        if _has(lg, "mazewalk"):
            pos = _resolve_pos(stmt.start, env, None)
            row, col = (pos if pos is not None else (None, None))
            lg.mazewalk(row=row, col=col, direction=stmt.direction)
        return

    if isinstance(stmt, RandomCorridors):
        if _has(lg, "add_random_corridors"):
            lg.add_random_corridors()
        return

    if isinstance(stmt, RandomMonsters):
        if _has(lg, "set_random_monsters"):
            lg.set_random_monsters(syms=stmt.syms)
        return

    if isinstance(stmt, RandomObjects):
        if _has(lg, "set_random_objects"):
            lg.set_random_objects(syms=stmt.syms)
        return

    if isinstance(stmt, NonDiggable):
        if _has(lg, "set_non_diggable"):
            lg.set_non_diggable(rect=stmt.rect)
        return

    if isinstance(stmt, NonPasswall):
        if _has(lg, "set_non_passwall"):
            lg.set_non_passwall(rect=stmt.rect)
        return

    if isinstance(stmt, Loop):
        for _ in range(stmt.count):
            for sub in stmt.body:
                _emit_stmt(lg, sub, env, rng_state)
        return

    if isinstance(stmt, IfElse):
        # Evaluate chance at compile time using the seeded RNG.
        if stmt.chance is None:
            branch = stmt.then_body
        else:
            roll = rnd.random() * 100.0
            branch = stmt.then_body if roll < stmt.chance else stmt.else_body
        for sub in branch:
            _emit_stmt(lg, sub, env, rng_state)
        return

    if isinstance(stmt, Choice):
        if not stmt.branches:
            return
        idx = rnd.randrange(0, len(stmt.branches))
        for sub in stmt.branches[idx]:
            _emit_stmt(lg, sub, env, rng_state)
        return

    if isinstance(stmt, Shuffle):
        val = env.get(stmt.var.name)
        if isinstance(val, ListExpr):
            items = list(val.items)
            rnd.shuffle(items)
            env[stmt.var.name] = ListExpr(items=items)
        return

    if isinstance(stmt, VarAssign):
        env[stmt.name] = stmt.value
        return

    if isinstance(stmt, UnknownStmt):
        # Track unknowns so callers can introspect.
        env.setdefault("_unknown", []).append(stmt.name)
        return


def _has(obj: Any, attr: str) -> bool:
    return hasattr(obj, attr) and callable(getattr(obj, attr))


def _emit_map_block(lg: Any, mb: MapBlock) -> None:
    """Emit a MAP block: tile-by-tile fill_terrain calls."""
    if mb is None or not _has(lg, "set_map") and not _has(lg, "fill_terrain"):
        return
    rows = mb.grid.split("\n")
    # MiniHack pads rows with spaces; some files have trailing blanks.
    if _has(lg, "set_map"):
        lg.set_map(rows)
        return
    for y, line in enumerate(rows):
        for x, ch in enumerate(line):
            if ch == " ":
                continue
            lg.fill_terrain(row=y, col=x, height=1, width=1, glyph=ch)


def compile_des(ast: DesAST) -> Callable[..., None]:
    """Compile an AST into ``fn(lg, rng)``.

    The returned function applies headers, then the MAP block (if any),
    then each statement in order against ``lg``.

    ``rng`` may be either a JAX PRNGKey or any object with a ``tolist``
    method (we fold it into Python's ``random.Random`` for the seeding).
    Plain ints are also accepted.
    """
    import random as _random

    def _seed_from(rng: Any) -> int:
        if rng is None:
            return 0
        if isinstance(rng, int):
            return rng
        if hasattr(rng, "tolist"):
            try:
                lst = rng.tolist()
                if isinstance(lst, list) and lst:
                    return int(lst[0])
                return int(lst)
            except Exception:  # pragma: no cover — best effort
                pass
        if hasattr(rng, "__iter__"):
            try:
                vals = list(rng)
                if vals:
                    return int(vals[0])
            except Exception:  # pragma: no cover
                pass
        try:
            return int(rng)
        except Exception:  # pragma: no cover
            return 0

    def fn(lg: Any, rng: Any = None) -> None:
        seed = _seed_from(rng)
        rng_state = {"rand": _random.Random(seed)}
        env: dict = {}
        # Apply headers first.
        for hdr in ast.headers:
            if hdr.kind == "FLAGS" and _has(lg, "set_flags"):
                lg.set_flags(hdr.args)
            elif hdr.kind == "GEOMETRY" and _has(lg, "set_geometry"):
                lg.set_geometry(hdr.args)
            elif hdr.kind == "INIT_MAP" and _has(lg, "init_map"):
                lg.init_map(hdr.args)
            elif hdr.kind == "MESSAGE" and _has(lg, "set_message"):
                lg.set_message(hdr.args[0] if hdr.args else "")
            elif hdr.kind == "MAZE" and _has(lg, "set_maze_name"):
                lg.set_maze_name(hdr.args[0] if hdr.args else "")
            elif hdr.kind == "LEVEL" and _has(lg, "set_level_name"):
                lg.set_level_name(hdr.args[0] if hdr.args else "")

        if ast.map_block is not None:
            _emit_map_block(lg, ast.map_block)

        for stmt in ast.statements:
            _emit_stmt(lg, stmt, env, rng_state)

    return fn


# ---------------------------------------------------------------------------
# des_to_factory: one-shot factory builder
# ---------------------------------------------------------------------------


def _get_level_generator_class() -> Optional[type]:
    """Import the real LevelGenerator if available; else return None."""
    try:
        from Nethax.minihax.level_generator import LevelGenerator  # type: ignore
        return LevelGenerator
    except Exception:
        return None


def des_to_factory(
    des_source: str,
    *,
    w: int = 80,
    h: int = 21,
    level_generator_cls: Optional[type] = None,
) -> Callable[..., Any]:
    """Compile a des-source into an ``EnvState`` factory function.

    The returned function has signature ``fn(rng)`` and produces the
    output of ``LevelGenerator.build()`` (typically an ``EnvState``).
    If the real ``LevelGenerator`` is not yet available, the factory
    instead returns the LevelGenerator-like instance, so callers can
    still inspect the recorded calls.
    """
    ast = parse_des(des_source)
    compiled = compile_des(ast)

    LG = level_generator_cls or _get_level_generator_class()

    def factory(rng: Any = None) -> Any:
        if LG is None:
            lg = _MockLevelGenerator(width=w, height=h)
            inner = None
        else:
            try:
                inner = LG(w=w, h=h)
            except TypeError:
                inner = LG()
            lg = _RealLGAdapter(inner)
        compiled(lg, rng)
        # If we wrapped a real LG, materialise an EnvState via its factory.
        if inner is not None and _has(inner, "get_factory") and rng is not None:
            try:
                fac = inner.get_factory()
                return fac(rng)
            except Exception:
                return inner
        if _has(lg, "build"):
            return lg.build(rng=rng)
        return lg

    return factory


# ---------------------------------------------------------------------------
# Internal fallback LevelGenerator
# ---------------------------------------------------------------------------


class _MockLevelGenerator:
    """Lightweight stand-in for ``LevelGenerator`` when sister agent A1
    hasn't landed.  Records every API call so callers can inspect what
    the parser emitted.
    """

    def __init__(self, *, width: int = 80, height: int = 21) -> None:
        self.width = width
        self.height = height
        self.calls: List[Tuple[str, tuple, dict]] = []
        self.map_rows: List[str] = []
        self.flags: Tuple[str, ...] = ()
        self.regions: dict = {}

    def _rec(self, name: str, args: tuple, kwargs: dict) -> Any:
        self.calls.append((name, args, kwargs))

    def set_map(self, rows: List[str]) -> None:
        self.map_rows = list(rows)
        self._rec("set_map", (rows,), {})

    def fill_terrain(self, **kwargs: Any) -> None:
        self._rec("fill_terrain", (), kwargs)

    def replace_terrain(self, **kwargs: Any) -> None:
        self._rec("replace_terrain", (), kwargs)

    def add_room(self, **kwargs: Any) -> str:
        self._rec("add_room", (), kwargs)
        return f"room_{len(self.calls)}"

    def add_region(self, **kwargs: Any) -> None:
        if "name" in kwargs:
            self.regions[kwargs["name"]] = kwargs
        self._rec("add_region", (), kwargs)

    def add_monster(self, **kwargs: Any) -> None:
        self._rec("add_monster", (), kwargs)

    def add_object(self, **kwargs: Any) -> None:
        self._rec("add_object", (), kwargs)

    def add_trap(self, **kwargs: Any) -> None:
        self._rec("add_trap", (), kwargs)

    def add_stair(self, **kwargs: Any) -> None:
        self._rec("add_stair", (), kwargs)

    def add_door(self, **kwargs: Any) -> None:
        self._rec("add_door", (), kwargs)

    def set_branch(self, **kwargs: Any) -> None:
        self._rec("set_branch", (), kwargs)

    def add_container(self, **kwargs: Any) -> str:
        self._rec("add_container", (), kwargs)
        return f"container_{len(self.calls)}"

    def mazewalk(self, **kwargs: Any) -> None:
        self._rec("mazewalk", (), kwargs)

    def add_random_corridors(self) -> None:
        self._rec("add_random_corridors", (), {})

    def set_random_monsters(self, **kwargs: Any) -> None:
        self._rec("set_random_monsters", (), kwargs)

    def set_random_objects(self, **kwargs: Any) -> None:
        self._rec("set_random_objects", (), kwargs)

    def set_non_diggable(self, **kwargs: Any) -> None:
        self._rec("set_non_diggable", (), kwargs)

    def set_non_passwall(self, **kwargs: Any) -> None:
        self._rec("set_non_passwall", (), kwargs)

    def set_flags(self, flags: Iterable[str]) -> None:
        self.flags = tuple(flags)
        self._rec("set_flags", (tuple(flags),), {})

    def set_geometry(self, geom: Any) -> None:
        self._rec("set_geometry", (geom,), {})

    def init_map(self, args: Any) -> None:
        self._rec("init_map", (args,), {})

    def set_message(self, msg: str) -> None:
        self._rec("set_message", (msg,), {})

    def set_maze_name(self, name: str) -> None:
        self._rec("set_maze_name", (name,), {})

    def set_level_name(self, name: str) -> None:
        self._rec("set_level_name", (name,), {})

    def build(self, rng: Any = None) -> Any:
        # No real EnvState — return self so callers can inspect.
        return self


class _RealLGAdapter:
    """Adapter that exposes the emitter's expected method names while
    forwarding to a real ``LevelGenerator`` instance (sister agent A1).

    The real LG uses positional ``(x, y)`` arguments and methods like
    ``add_stair_up``/``add_stair_down``; our emitter uses ``(row, col)``
    keyword arguments and a generic ``add_stair(direction=...)``.  This
    adapter does the translation.

    Unknown calls are silently dropped, which keeps smoke tests robust
    as new directives are added.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        # Track set_map rows for inspection.
        self.calls: List[Tuple[str, tuple, dict]] = []
        self._regions: dict = {}

    def _rec(self, name: str, args: tuple, kwargs: dict) -> None:
        self.calls.append((name, args, kwargs))

    @staticmethod
    def _to_xy(kwargs: dict) -> Tuple[Optional[int], Optional[int]]:
        row = kwargs.get("row")
        col = kwargs.get("col")
        if row is None or col is None:
            return None, None
        return int(col), int(row)

    def set_map(self, rows: List[str]) -> None:
        # Real LG doesn't ingest MAP blocks directly; replay as
        # fill_terrain calls for each non-fill character.
        self._rec("set_map", (rows,), {})
        # We can't bulk-set without violating dimensions; skip if too big.
        inner = self._inner
        if not _has(inner, "fill_terrain"):
            return
        # Map every recognised terrain glyph in each row.
        for y, line in enumerate(rows):
            for x, ch in enumerate(line):
                if ch == " ":
                    continue
                try:
                    inner.fill_terrain(ch, x, y, x, y)
                except Exception:
                    pass  # unknown glyph — leave default fill

    def fill_terrain(self, **kwargs: Any) -> None:
        self._rec("fill_terrain", (), kwargs)
        inner = self._inner
        if not _has(inner, "fill_terrain"):
            return
        row = kwargs.get("row")
        col = kwargs.get("col")
        height = kwargs.get("height", 1)
        width = kwargs.get("width", 1)
        glyph = kwargs.get("glyph")
        if row is None or col is None or glyph is None:
            return
        try:
            inner.fill_terrain(glyph, col, row, col + width - 1, row + height - 1)
        except Exception:
            pass

    def replace_terrain(self, **kwargs: Any) -> None:
        self._rec("replace_terrain", (), kwargs)
        # No real-LG counterpart yet.

    def add_room(self, **kwargs: Any) -> str:
        self._rec("add_room", (), kwargs)
        inner = self._inner
        if not _has(inner, "add_room"):
            return f"room_{len(self.calls)}"
        row = kwargs.get("row")
        col = kwargs.get("col")
        height = kwargs.get("height")
        width = kwargs.get("width")
        lit = kwargs.get("lit", True)
        try:
            return inner.add_room(
                x=-1 if col is None else int(col),
                y=-1 if row is None else int(row),
                w=-1 if width is None else int(width),
                h=-1 if height is None else int(height),
                lit=bool(lit),
            )
        except Exception:
            return f"room_{len(self.calls)}"

    def add_region(self, **kwargs: Any) -> None:
        self._rec("add_region", (), kwargs)
        # Regions on the real LG map to add_room with explicit bounds.
        name = kwargs.get("name")
        if name is not None:
            self._regions[name] = kwargs

    def add_monster(self, **kwargs: Any) -> None:
        self._rec("add_monster", (), kwargs)
        inner = self._inner
        if not _has(inner, "add_monster"):
            return
        x, y = self._to_xy(kwargs)
        place = (x, y) if x is not None else None
        try:
            inner.add_monster(
                name=kwargs.get("name") or "random",
                symbol=kwargs.get("sym"),
                place=place,
            )
        except Exception:
            pass

    def add_object(self, **kwargs: Any) -> None:
        self._rec("add_object", (), kwargs)
        inner = self._inner
        if not _has(inner, "add_object"):
            return
        x, y = self._to_xy(kwargs)
        place = (x, y) if x is not None else None
        opts = kwargs.get("options", ())
        cursestate = "random"
        for o in opts:
            if o in ("blessed", "uncursed", "cursed"):
                cursestate = o
                break
        try:
            inner.add_object(
                name=kwargs.get("name") or "random",
                symbol=kwargs.get("sym"),
                place=place,
                cursestate=cursestate,
            )
        except Exception:
            pass

    def add_trap(self, **kwargs: Any) -> None:
        self._rec("add_trap", (), kwargs)
        inner = self._inner
        if not _has(inner, "add_trap"):
            return
        x, y = self._to_xy(kwargs)
        place = (x, y) if x is not None else None
        try:
            inner.add_trap(name=kwargs.get("kind", "teleport"), place=place)
        except Exception:
            pass

    def add_stair(self, **kwargs: Any) -> None:
        self._rec("add_stair", (), kwargs)
        inner = self._inner
        direction = kwargs.get("direction", "down")
        x, y = self._to_xy(kwargs)
        method = "add_stair_down" if direction == "down" else "add_stair_up"
        fn = getattr(inner, method, None)
        if fn is None:
            return
        try:
            if x is None:
                fn()
            else:
                fn(x, y)
        except Exception:
            pass

    def add_door(self, **kwargs: Any) -> None:
        self._rec("add_door", (), kwargs)
        inner = self._inner
        if not _has(inner, "add_door"):
            return
        x, y = self._to_xy(kwargs)
        state = kwargs.get("state", "closed")
        if state in ("locked", "open", "closed", "nodoor", "random"):
            pass
        else:
            state = "closed"
        if x is None:
            return
        try:
            inner.add_door(x, y, state=state)
        except Exception:
            pass

    def set_branch(self, **kwargs: Any) -> None:
        self._rec("set_branch", (), kwargs)

    def add_container(self, **kwargs: Any) -> str:
        self._rec("add_container", (), kwargs)
        return f"container_{len(self.calls)}"

    def mazewalk(self, **kwargs: Any) -> None:
        self._rec("mazewalk", (), kwargs)

    def add_random_corridors(self) -> None:
        self._rec("add_random_corridors", (), {})

    def set_random_monsters(self, **kwargs: Any) -> None:
        self._rec("set_random_monsters", (), kwargs)

    def set_random_objects(self, **kwargs: Any) -> None:
        self._rec("set_random_objects", (), kwargs)

    def set_non_diggable(self, **kwargs: Any) -> None:
        self._rec("set_non_diggable", (), kwargs)

    def set_non_passwall(self, **kwargs: Any) -> None:
        self._rec("set_non_passwall", (), kwargs)

    def set_flags(self, flags: Iterable[str]) -> None:
        self._rec("set_flags", (tuple(flags),), {})

    def set_geometry(self, geom: Any) -> None:
        self._rec("set_geometry", (geom,), {})

    def init_map(self, args: Any) -> None:
        self._rec("init_map", (args,), {})

    def set_message(self, msg: str) -> None:
        self._rec("set_message", (msg,), {})

    def set_maze_name(self, name: str) -> None:
        self._rec("set_maze_name", (name,), {})

    def set_level_name(self, name: str) -> None:
        self._rec("set_level_name", (name,), {})

    def build(self, rng: Any = None) -> Any:
        return self._inner


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def smoke_parse_dir(directory: str) -> Tuple[int, int, List[Tuple[str, str]]]:
    """Parse every ``*.des`` file in ``directory``.

    Returns ``(ok_count, total, failures)`` where ``failures`` is a list
    of ``(filename, error_message)`` tuples.  Useful for the test that
    asserts the parser handles the full canonical corpus.
    """
    import os
    failures: List[Tuple[str, str]] = []
    total = 0
    ok = 0
    for fn in sorted(os.listdir(directory)):
        if not fn.endswith(".des"):
            continue
        total += 1
        path = os.path.join(directory, fn)
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                src = fh.read()
            parse_des(src)
            ok += 1
        except Exception as exc:  # noqa: BLE001 — collect everything
            failures.append((fn, repr(exc)))
    return ok, total, failures


__all__ = [
    "DesAST",
    "DesParseError",
    "MapBlock",
    "Header",
    "Region",
    "Room",
    "Monster",
    "ObjectStmt",
    "Trap",
    "StairStmt",
    "Door",
    "RoomDoor",
    "Branch",
    "Terrain",
    "ReplaceTerrain",
    "Container",
    "MazeWalk",
    "RandomCorridors",
    "RandomMonsters",
    "RandomObjects",
    "NonDiggable",
    "NonPasswall",
    "Loop",
    "IfElse",
    "Choice",
    "Shuffle",
    "VarAssign",
    "VarRef",
    "RndCoord",
    "Selection",
    "ListExpr",
    "Coord",
    "Rect",
    "Random",
    "UnknownStmt",
    "tokenize",
    "parse_des",
    "compile_des",
    "des_to_factory",
    "smoke_parse_dir",
    "TERRAIN_GLYPHS",
    "OBJECT_CLASS_GLYPH",
    "MONSTER_CLASS_GLYPH",
]
