"""Vendor per-role / per-XP-level rank-title lookup.

Vendor source: ``vendor/nethack/src/botl.c::rank_of`` lines 331-364 plus the
per-role ``rank[9]`` arrays in ``vendor/nethack/src/role.c`` lines 30-573.

The vendor table is structured as 13 roles x 9 ranks; experience level is
mapped to a rank index by ``botl.c::xlev_to_rank`` (lines 296-311):

    xlev  in [1, 2]   -> rank 0
    xlev  in [3, 5]   -> rank 1
    xlev  in [6, 9]   -> rank 2
    xlev  in [10, 13] -> rank 3
    xlev  in [14, 17] -> rank 4
    xlev  in [18, 21] -> rank 5
    xlev  in [22, 25] -> rank 6
    xlev  in [26, 29] -> rank 7
    xlev  == 30        -> rank 8

For NLE-style consumers we expose the fully expanded 13 x 30 table
``_RANK_TITLES`` (XP level 1..30 inclusive on the column axis) plus a
``rank_title(role, xp_level)`` helper.

Role order matches ``vendor/nethack/src/role.c::roles[]`` and the
``Nethax.nethax.constants.roles.Role`` IntEnum:

    0  ARCHEOLOGIST
    1  BARBARIAN
    2  CAVEMAN
    3  HEALER
    4  KNIGHT
    5  MONK
    6  PRIEST
    7  RANGER
    8  ROGUE
    9  SAMURAI
   10  TOURIST
   11  VALKYRIE
   12  WIZARD

Cite for the individual rank strings: each role's ``rank`` member in
``vendor/nethack/src/role.c``:
    Archeologist  lines  32-40
    Barbarian     lines  73-81
    Caveman       lines 114-122
    Healer        lines 155-163
    Knight        lines 195-203
    Monk          lines 235-243
    Priest        lines 276-284
    Ranger        lines 373-381
    Rogue         lines 319-327
    Samurai       lines 414-422
    Tourist       lines 454-462
    Valkyrie      lines 494-502
    Wizard        lines 534-542

Female-variant titles are not modeled; ``rank_title`` returns the male
form (vendor falls back to ``role->rank[i].m`` when ``.f`` is null).
"""

from __future__ import annotations

from typing import Tuple


__all__ = [
    "RANK_TITLES_BY_ROLE",
    "_RANK_TITLES",
    "rank_title",
    "xlev_to_rank",
]


# ---------------------------------------------------------------------------
# Per-role rank[9] tables — vendor-exact strings.
# ---------------------------------------------------------------------------
RANK_TITLES_BY_ROLE: Tuple[Tuple[str, ...], ...] = (
    # 0 ARCHEOLOGIST  (role.c:32-40)
    ("Digger", "Field Worker", "Investigator", "Exhumer", "Excavator",
     "Spelunker", "Speleologist", "Collector", "Curator"),
    # 1 BARBARIAN     (role.c:73-81)
    ("Plunderer", "Pillager", "Bandit", "Brigand", "Raider",
     "Reaver", "Slayer", "Chieftain", "Conqueror"),
    # 2 CAVEMAN       (role.c:114-122)
    ("Troglodyte", "Aborigine", "Wanderer", "Vagrant", "Wayfarer",
     "Roamer", "Nomad", "Rover", "Pioneer"),
    # 3 HEALER        (role.c:155-163)
    ("Rhizotomist", "Empiric", "Embalmer", "Dresser", "Medicus ossium",
     "Herbalist", "Magister", "Physician", "Chirurgeon"),
    # 4 KNIGHT        (role.c:195-203)
    ("Gallant", "Esquire", "Bachelor", "Sergeant", "Knight",
     "Banneret", "Chevalier", "Seignieur", "Paladin"),
    # 5 MONK          (role.c:235-243)
    ("Candidate", "Novice", "Initiate", "Student of Stones", "Student of Waters",
     "Student of Metals", "Student of Winds", "Student of Fire", "Master"),
    # 6 PRIEST        (role.c:276-284)
    ("Aspirant", "Acolyte", "Adept", "Priest", "Curate",
     "Canon", "Lama", "Patriarch", "High Priest"),
    # 7 RANGER        (role.c:373-381)
    ("Tenderfoot", "Lookout", "Trailblazer", "Reconnoiterer", "Scout",
     "Arbalester", "Archer", "Sharpshooter", "Marksman"),
    # 8 ROGUE         (role.c:319-327)
    ("Footpad", "Cutpurse", "Rogue", "Pilferer", "Robber",
     "Burglar", "Filcher", "Magsman", "Thief"),
    # 9 SAMURAI       (role.c:414-422)
    ("Hatamoto", "Ronin", "Ninja", "Joshu", "Ryoshu",
     "Kokushu", "Daimyo", "Kuge", "Shogun"),
    # 10 TOURIST      (role.c:454-462)
    ("Rambler", "Sightseer", "Excursionist", "Peregrinator", "Traveler",
     "Journeyer", "Voyager", "Explorer", "Adventurer"),
    # 11 VALKYRIE     (role.c:494-502)
    ("Stripling", "Skirmisher", "Fighter", "Man-at-arms", "Warrior",
     "Swashbuckler", "Hero", "Champion", "Lord"),
    # 12 WIZARD       (role.c:534-542)
    ("Evoker", "Conjurer", "Thaumaturge", "Magician", "Enchanter",
     "Sorcerer", "Necromancer", "Wizard", "Mage"),
)


# ---------------------------------------------------------------------------
# Vendor xlev_to_rank  (botl.c:296-311).
# ---------------------------------------------------------------------------
def xlev_to_rank(xlev: int) -> int:
    """Map experience level [1..30] to rank index [0..8]."""
    if xlev <= 2:
        return 0
    if xlev <= 30:
        return (xlev + 2) // 4
    return 8


# ---------------------------------------------------------------------------
# Fully-expanded 13 x 30 table indexed as ``_RANK_TITLES[role][xp_level-1]``.
# Built once at module load.
# ---------------------------------------------------------------------------
def _build_expanded_table() -> Tuple[Tuple[str, ...], ...]:
    rows = []
    for role_ranks in RANK_TITLES_BY_ROLE:
        # xp_level ranges 1..30, columns 0..29.
        row = tuple(role_ranks[xlev_to_rank(xp)] for xp in range(1, 31))
        rows.append(row)
    return tuple(rows)


_RANK_TITLES: Tuple[Tuple[str, ...], ...] = _build_expanded_table()


# ---------------------------------------------------------------------------
# Public helper.
# ---------------------------------------------------------------------------
def rank_title(role: int, xp_level: int) -> str:
    """Return the vendor rank title for ``(role, xp_level)``.

    Args:
        role: ``Nethax.nethax.constants.roles.Role`` int value (0..12).
        xp_level: experience level in ``[1, 30]``; values outside the range
            are clamped to the nearest endpoint per vendor ``xlev_to_rank``.

    Returns:
        The vendor rank-title string (male form when role-specific gendered
        form is null, matching vendor ``rank_of`` fallback).  Returns
        ``"Player"`` when ``role`` is out of range, mirroring vendor's
        terminal fallback at ``botl.c:357``.

    Cite: vendor/nethack/src/botl.c::rank_of (lines 331-358).
    """
    r = int(role)
    if r < 0 or r >= len(_RANK_TITLES):
        return "Player"
    xl = max(1, min(30, int(xp_level)))
    return _RANK_TITLES[r][xl - 1]
