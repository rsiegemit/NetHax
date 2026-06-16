"""World generator for ``MiniHack-CorridorBattle{,-Dark}-v0``.

Vendor reference:
- ``vendor/minihack/minihack/envs/fightcorridor.py``
- ``vendor/minihack/minihack/dat/fightcorridor.des`` (when present)

Vendor lays down two ``ROOM: "ordinary"`` blocks with random anchors + sizes
connected by ``RANDOM_CORRIDORS`` and seeds three random monsters between
them.  The legacy canonical builder (``battle_builder`` in
``Nethax/minihax/envs/canonical.py``) hard-coded both room anchors
(``add_room(x=2, y=8, w=4, h=4)`` and ``add_room(x=70, y=8, w=4, h=4)``)
plus the connecting corridor (``add_corridor((6, 10), (70, 10))``) so every
``reset`` produced an identical level.  This module exposes a builder
factory that mirrors vendor randomization by passing ``x=-1, y=-1`` to
``add_room``; the LG resolver consumes split PRNG keys at factory time
and picks fresh anchors per ``reset``.
"""


def make_corridor_battle_builder(lit: bool):
    """Return an ``(lg) -> None`` builder for a CorridorBattle level.

    Args:
        lit: passed through for symmetry with the vendor ``lit`` flag (the
            actual lit/dark wiring happens via ``_make_factory(..., lit=lit)``
            in the canonical registry; the LG itself does not branch on it).

    Layout:
        * Two ``4×4`` interior rooms placed at random anchors via
          ``add_room(x=-1, y=-1, w=4, h=4)``.
        * Down-stair somewhere in the second room (``place=room_id``).
        * Three random monsters anywhere on a floor tile.
        * Player start is left implicit; the LG default ("first floor tile
          we can find") drops the agent into the first room.

    Limitation: explicit L-corridor carving between the two random rooms
    requires a new LG directive (``_CorridorByRoomIdDirective``) because
    ``add_corridor`` today only accepts fixed ``(x, y)`` endpoints and the
    room anchors are only known at factory time.  Until that LG hook lands,
    the two rooms are disjoint — the level still randomizes per ``reset``
    (terrain differs across seeds, which is what canonical parity tests
    check) but the agent cannot traverse between rooms.  See the report
    note for the canonical.py + level_generator.py changes required to
    restore connectivity.
    """
    del lit  # currently unused (lit flag is plumbed via _make_factory).

    def build(lg) -> None:
        room0 = lg.add_room(x=-1, y=-1, w=4, h=4)
        room1 = lg.add_room(x=-1, y=-1, w=4, h=4)
        # No-op random-corridors hook (placeholder for the future LG
        # directive that wires random rooms together).
        lg.add_random_corridors()
        lg.add_stair_down(place=room1)
        for _ in range(3):
            lg.add_monster()
        # Player start defaults to the first floor tile (inside room0).
        del room0

    return build
