"""Skills subsystem — weapon and spell skill advancement.

Canonical sources:
  vendor/nethack/include/skills.h — SkillLevel enum, practice_needed_to_advance
  vendor/nethack/src/weapon.c     — use_skill (line 1424), skill_advance
  vendor/nethack/src/u_init.c     — per-role skill caps (Skill_A … Skill_W tables)
"""
from __future__ import annotations

import enum

import jax.numpy as jnp
from flax import struct

from Nethax.nethax.constants.roles import Role


# ---------------------------------------------------------------------------
# SkillLevel  (vendor/nethack/include/skills.h lines 92-103)
# ---------------------------------------------------------------------------
class SkillLevel(enum.IntEnum):
    P_UNSKILLED   = 0
    P_BASIC       = 1
    P_SKILLED     = 2
    P_EXPERT      = 3
    P_MASTER      = 4
    P_GRAND_MASTER = 5


# ---------------------------------------------------------------------------
# SkillId  (vendor/nethack/include/skills.h lines 24-65; 0-indexed Nethax mapping)
# Vendor uses 1-based; we use 0-based contiguous IDs for array indexing.
# ---------------------------------------------------------------------------
class SkillId(enum.IntEnum):
    # Weapons (vendor P_DAGGER=1 … P_UNICORN_HORN=27, P_WHIP=26)
    DAGGER           = 0
    KNIFE            = 1
    AXE              = 2
    PICK_AXE         = 3
    SHORT_SWORD      = 4
    BROAD_SWORD      = 5
    LONG_SWORD       = 6
    TWO_HANDED_SWORD = 7
    SABER            = 8
    CLUB             = 9
    MACE             = 10
    MORNING_STAR     = 11
    FLAIL            = 12
    HAMMER           = 13
    QUARTERSTAFF     = 14
    POLEARMS         = 15
    SPEAR            = 16
    TRIDENT          = 17
    LANCE            = 18
    BOW              = 19
    SLING            = 20
    CROSSBOW         = 21
    DART             = 22
    SHURIKEN         = 23
    BOOMERANG        = 24
    WHIP             = 25
    UNICORN_HORN     = 26
    # Spell categories (vendor P_ATTACK_SPELL=28 … P_MATTER_SPELL=34)
    ATTACK_SPELL       = 27
    HEALING_SPELL      = 28
    DIVINATION_SPELL   = 29
    ENCHANTMENT_SPELL  = 30
    CLERIC_SPELL       = 31
    ESCAPE_SPELL       = 32
    MATTER_SPELL       = 33
    # Special combat (vendor P_BARE_HANDED_COMBAT=35, P_TWO_WEAPON_COMBAT=36, P_RIDING=37)
    MARTIAL_ARTS       = 34
    TWO_WEAPON_COMBAT  = 35
    RIDING             = 36


N_SKILLS: int = len(SkillId)
N_ROLES:  int = len(Role)


# ---------------------------------------------------------------------------
# practice_needed_to_advance  (vendor/nethack/include/skills.h:106)
#   practice_needed_to_advance(level) = level * level * 20
# ---------------------------------------------------------------------------
def practice_needed_to_advance(level: jnp.ndarray) -> jnp.ndarray:
    """Return practice points needed to advance from ``level`` to level+1.

    Cite: vendor/nethack/include/skills.h:106 —
          ``#define practice_needed_to_advance(level) ((level)*(level)*20)``

    P_UNSKILLED(0)→P_BASIC(1):   0*0*20 =   0  (advance immediately once ≥0)
    P_BASIC(1)→P_SKILLED(2):     1*1*20 =  20
    P_SKILLED(2)→P_EXPERT(3):    2*2*20 =  80
    P_EXPERT(3)→P_MASTER(4):     3*3*20 = 180
    P_MASTER(4)→P_GRAND_MASTER:  4*4*20 = 320

    Note: vendor uses 1-based levels (P_UNSKILLED=1).  We use 0-based so
    the formula shifts: advance from our level L requires L*L*20 practice.
    This preserves the sequence 0, 20, 80, 180, 320, 500 for L=0..5.
    """
    lv = level.astype(jnp.int32)
    return (lv * lv * jnp.int32(20)).astype(jnp.int32)


# ---------------------------------------------------------------------------
# Per-role skill caps  (vendor/nethack/src/u_init.c Skill_A … Skill_W)
# Array shape: [N_ROLES, N_SKILLS], dtype int8, values = SkillLevel.
# Roles are ordered per Role enum:
#   0=ARCH, 1=BARB, 2=CAVE, 3=HEAL, 4=KNIGHT, 5=MONK, 6=PRIEST,
#   7=RANGER, 8=ROGUE, 9=SAMURAI, 10=TOURIST, 11=VALKYRIE, 12=WIZARD
# ---------------------------------------------------------------------------
_U = int(SkillLevel.P_UNSKILLED)
_B = int(SkillLevel.P_BASIC)
_S = int(SkillLevel.P_SKILLED)
_E = int(SkillLevel.P_EXPERT)
_M = int(SkillLevel.P_MASTER)
_G = int(SkillLevel.P_GRAND_MASTER)

def _build_role_skill_caps() -> jnp.ndarray:
    """Build [N_ROLES, N_SKILLS] int8 cap table from u_init.c Skill_X tables.

    Cite: vendor/nethack/src/u_init.c lines 257-571.
    Unlisted skills default to P_UNSKILLED (restricted).
    """
    # Initialize all to P_UNSKILLED
    caps = [[_U] * N_SKILLS for _ in range(N_ROLES)]

    def _set(role_idx: int, skill: SkillId, cap: int) -> None:
        caps[role_idx][int(skill)] = cap

    S = SkillId

    # ---- ARCHEOLOGIST (0) — Skill_A, u_init.c:257 ----
    r = int(Role.ARCHEOLOGIST)
    _set(r, S.DAGGER,           _B)
    _set(r, S.KNIFE,            _B)
    _set(r, S.PICK_AXE,         _E)
    _set(r, S.SHORT_SWORD,      _B)
    _set(r, S.SABER,            _E)
    _set(r, S.CLUB,             _S)
    _set(r, S.QUARTERSTAFF,     _S)
    _set(r, S.SLING,            _S)
    _set(r, S.DART,             _B)
    _set(r, S.BOOMERANG,        _E)
    _set(r, S.WHIP,             _E)
    _set(r, S.UNICORN_HORN,     _S)
    _set(r, S.ATTACK_SPELL,     _B)
    _set(r, S.HEALING_SPELL,    _B)
    _set(r, S.DIVINATION_SPELL, _E)
    _set(r, S.MATTER_SPELL,     _B)
    _set(r, S.RIDING,           _B)
    _set(r, S.TWO_WEAPON_COMBAT,_B)
    _set(r, S.MARTIAL_ARTS,     _E)

    # ---- BARBARIAN (1) — Skill_B, u_init.c:279 ----
    r = int(Role.BARBARIAN)
    _set(r, S.DAGGER,           _B)
    _set(r, S.AXE,              _E)
    _set(r, S.PICK_AXE,         _S)
    _set(r, S.SHORT_SWORD,      _E)
    _set(r, S.BROAD_SWORD,      _S)
    _set(r, S.LONG_SWORD,       _S)
    _set(r, S.TWO_HANDED_SWORD, _E)
    _set(r, S.SABER,            _S)
    _set(r, S.CLUB,             _S)
    _set(r, S.MACE,             _S)
    _set(r, S.MORNING_STAR,     _S)
    _set(r, S.FLAIL,            _B)
    _set(r, S.HAMMER,           _E)
    _set(r, S.QUARTERSTAFF,     _B)
    _set(r, S.SPEAR,            _S)
    _set(r, S.TRIDENT,          _S)
    _set(r, S.BOW,              _B)
    _set(r, S.ATTACK_SPELL,     _B)
    _set(r, S.ESCAPE_SPELL,     _B)
    _set(r, S.RIDING,           _B)
    _set(r, S.TWO_WEAPON_COMBAT,_B)
    _set(r, S.MARTIAL_ARTS,     _M)

    # ---- CAVEMAN (2) — Skill_C, u_init.c:304 ----
    r = int(Role.CAVEMAN)
    _set(r, S.DAGGER,           _B)
    _set(r, S.KNIFE,            _S)
    _set(r, S.AXE,              _S)
    _set(r, S.PICK_AXE,         _B)
    _set(r, S.CLUB,             _E)
    _set(r, S.MACE,             _E)
    _set(r, S.MORNING_STAR,     _B)
    _set(r, S.FLAIL,            _S)
    _set(r, S.HAMMER,           _S)
    _set(r, S.QUARTERSTAFF,     _E)
    _set(r, S.POLEARMS,         _S)
    _set(r, S.SPEAR,            _E)
    _set(r, S.TRIDENT,          _S)
    _set(r, S.BOW,              _S)
    _set(r, S.SLING,            _E)
    _set(r, S.ATTACK_SPELL,     _B)
    _set(r, S.MATTER_SPELL,     _S)
    _set(r, S.BOOMERANG,        _E)
    _set(r, S.UNICORN_HORN,     _B)
    _set(r, S.MARTIAL_ARTS,     _M)

    # ---- HEALER (3) — Skill_H, u_init.c:327 ----
    r = int(Role.HEALER)
    _set(r, S.DAGGER,           _S)
    _set(r, S.KNIFE,            _E)
    _set(r, S.SHORT_SWORD,      _S)
    _set(r, S.SABER,            _B)
    _set(r, S.CLUB,             _S)
    _set(r, S.MACE,             _B)
    _set(r, S.QUARTERSTAFF,     _E)
    _set(r, S.POLEARMS,         _B)
    _set(r, S.SPEAR,            _B)
    _set(r, S.TRIDENT,          _B)
    _set(r, S.SLING,            _S)
    _set(r, S.DART,             _E)
    _set(r, S.SHURIKEN,         _S)
    _set(r, S.UNICORN_HORN,     _E)
    _set(r, S.HEALING_SPELL,    _E)
    _set(r, S.MARTIAL_ARTS,     _B)

    # ---- KNIGHT (4) — Skill_K, u_init.c:346 ----
    r = int(Role.KNIGHT)
    _set(r, S.DAGGER,           _B)
    _set(r, S.KNIFE,            _B)
    _set(r, S.AXE,              _S)
    _set(r, S.PICK_AXE,         _B)
    _set(r, S.SHORT_SWORD,      _S)
    _set(r, S.BROAD_SWORD,      _S)
    _set(r, S.LONG_SWORD,       _E)
    _set(r, S.TWO_HANDED_SWORD, _S)
    _set(r, S.SABER,            _S)
    _set(r, S.CLUB,             _B)
    _set(r, S.MACE,             _S)
    _set(r, S.MORNING_STAR,     _S)
    _set(r, S.FLAIL,            _B)
    _set(r, S.HAMMER,           _B)
    _set(r, S.POLEARMS,         _S)
    _set(r, S.SPEAR,            _S)
    _set(r, S.TRIDENT,          _B)
    _set(r, S.LANCE,            _E)
    _set(r, S.BOW,              _B)
    _set(r, S.CROSSBOW,         _S)
    _set(r, S.ATTACK_SPELL,     _S)
    _set(r, S.HEALING_SPELL,    _S)
    _set(r, S.CLERIC_SPELL,     _S)
    _set(r, S.RIDING,           _E)
    _set(r, S.TWO_WEAPON_COMBAT,_S)
    _set(r, S.MARTIAL_ARTS,     _E)

    # ---- MONK (5) — Skill_Mon, u_init.c:375 ----
    r = int(Role.MONK)
    _set(r, S.QUARTERSTAFF,     _B)
    _set(r, S.SPEAR,            _B)
    _set(r, S.CROSSBOW,         _B)
    _set(r, S.SHURIKEN,         _B)
    _set(r, S.ATTACK_SPELL,     _B)
    _set(r, S.HEALING_SPELL,    _E)
    _set(r, S.DIVINATION_SPELL, _B)
    _set(r, S.ENCHANTMENT_SPELL,_B)
    _set(r, S.CLERIC_SPELL,     _S)
    _set(r, S.ESCAPE_SPELL,     _S)
    _set(r, S.MATTER_SPELL,     _B)
    _set(r, S.MARTIAL_ARTS,     _G)

    # ---- PRIEST (6) — Skill_P, u_init.c:390 ----
    r = int(Role.PRIEST)
    _set(r, S.CLUB,             _E)
    _set(r, S.MACE,             _E)
    _set(r, S.MORNING_STAR,     _E)
    _set(r, S.FLAIL,            _E)
    _set(r, S.HAMMER,           _E)
    _set(r, S.QUARTERSTAFF,     _E)
    _set(r, S.POLEARMS,         _S)
    _set(r, S.SPEAR,            _S)
    _set(r, S.TRIDENT,          _S)
    _set(r, S.LANCE,            _B)
    _set(r, S.BOW,              _B)
    _set(r, S.SLING,            _B)
    _set(r, S.CROSSBOW,         _B)
    _set(r, S.DART,             _B)
    _set(r, S.SHURIKEN,         _B)
    _set(r, S.BOOMERANG,        _B)
    _set(r, S.UNICORN_HORN,     _S)
    _set(r, S.HEALING_SPELL,    _E)
    _set(r, S.DIVINATION_SPELL, _E)
    _set(r, S.CLERIC_SPELL,     _E)
    _set(r, S.MARTIAL_ARTS,     _B)

    # ---- RANGER (7) — Skill_Ran, u_init.c:440 ----
    r = int(Role.RANGER)
    _set(r, S.DAGGER,           _E)
    _set(r, S.KNIFE,            _S)
    _set(r, S.AXE,              _S)
    _set(r, S.PICK_AXE,         _B)
    _set(r, S.SHORT_SWORD,      _B)
    _set(r, S.MORNING_STAR,     _B)
    _set(r, S.FLAIL,            _S)
    _set(r, S.HAMMER,           _B)
    _set(r, S.QUARTERSTAFF,     _B)
    _set(r, S.POLEARMS,         _S)
    _set(r, S.SPEAR,            _E)
    _set(r, S.TRIDENT,          _B)
    _set(r, S.BOW,              _E)
    _set(r, S.SLING,            _E)
    _set(r, S.CROSSBOW,         _E)
    _set(r, S.DART,             _E)
    _set(r, S.SHURIKEN,         _S)
    _set(r, S.BOOMERANG,        _E)
    _set(r, S.WHIP,             _B)
    _set(r, S.HEALING_SPELL,    _B)
    _set(r, S.DIVINATION_SPELL, _E)
    _set(r, S.ESCAPE_SPELL,     _B)
    _set(r, S.RIDING,           _B)
    _set(r, S.MARTIAL_ARTS,     _B)

    # ---- ROGUE (8) — Skill_R, u_init.c:414 ----
    r = int(Role.ROGUE)
    _set(r, S.DAGGER,           _E)
    _set(r, S.KNIFE,            _E)
    _set(r, S.SHORT_SWORD,      _E)
    _set(r, S.BROAD_SWORD,      _S)
    _set(r, S.LONG_SWORD,       _S)
    _set(r, S.TWO_HANDED_SWORD, _B)
    _set(r, S.SABER,            _S)
    _set(r, S.CLUB,             _S)
    _set(r, S.MACE,             _S)
    _set(r, S.MORNING_STAR,     _B)
    _set(r, S.FLAIL,            _B)
    _set(r, S.HAMMER,           _B)
    _set(r, S.POLEARMS,         _B)
    _set(r, S.SPEAR,            _B)
    _set(r, S.CROSSBOW,         _E)
    _set(r, S.DART,             _E)
    _set(r, S.SHURIKEN,         _S)
    _set(r, S.DIVINATION_SPELL, _S)
    _set(r, S.ESCAPE_SPELL,     _S)
    _set(r, S.MATTER_SPELL,     _S)
    _set(r, S.RIDING,           _B)
    _set(r, S.TWO_WEAPON_COMBAT,_E)
    _set(r, S.MARTIAL_ARTS,     _E)

    # ---- SAMURAI (9) — Skill_S, u_init.c:467 ----
    r = int(Role.SAMURAI)
    _set(r, S.DAGGER,           _B)
    _set(r, S.KNIFE,            _S)
    _set(r, S.SHORT_SWORD,      _E)
    _set(r, S.BROAD_SWORD,      _S)
    _set(r, S.LONG_SWORD,       _E)
    _set(r, S.TWO_HANDED_SWORD, _E)
    _set(r, S.SABER,            _B)
    _set(r, S.FLAIL,            _S)
    _set(r, S.QUARTERSTAFF,     _B)
    _set(r, S.POLEARMS,         _S)
    _set(r, S.SPEAR,            _S)
    _set(r, S.LANCE,            _S)
    _set(r, S.BOW,              _E)
    _set(r, S.SHURIKEN,         _E)
    _set(r, S.ATTACK_SPELL,     _B)
    _set(r, S.DIVINATION_SPELL, _B)
    _set(r, S.CLERIC_SPELL,     _S)
    _set(r, S.RIDING,           _S)
    _set(r, S.TWO_WEAPON_COMBAT,_E)
    _set(r, S.MARTIAL_ARTS,     _M)

    # ---- TOURIST (10) — Skill_T, u_init.c:490 ----
    r = int(Role.TOURIST)
    _set(r, S.DAGGER,           _E)
    _set(r, S.KNIFE,            _S)
    _set(r, S.AXE,              _B)
    _set(r, S.PICK_AXE,         _B)
    _set(r, S.SHORT_SWORD,      _E)
    _set(r, S.BROAD_SWORD,      _B)
    _set(r, S.LONG_SWORD,       _B)
    _set(r, S.TWO_HANDED_SWORD, _B)
    _set(r, S.SABER,            _S)
    _set(r, S.MACE,             _B)
    _set(r, S.MORNING_STAR,     _B)
    _set(r, S.FLAIL,            _B)
    _set(r, S.HAMMER,           _B)
    _set(r, S.QUARTERSTAFF,     _B)
    _set(r, S.POLEARMS,         _B)
    _set(r, S.SPEAR,            _B)
    _set(r, S.TRIDENT,          _B)
    _set(r, S.LANCE,            _B)
    _set(r, S.BOW,              _B)
    _set(r, S.SLING,            _B)
    _set(r, S.CROSSBOW,         _B)
    _set(r, S.DART,             _E)
    _set(r, S.SHURIKEN,         _B)
    _set(r, S.BOOMERANG,        _B)
    _set(r, S.WHIP,             _B)
    _set(r, S.UNICORN_HORN,     _S)
    _set(r, S.DIVINATION_SPELL, _B)
    _set(r, S.ENCHANTMENT_SPELL,_B)
    _set(r, S.ESCAPE_SPELL,     _S)
    _set(r, S.RIDING,           _B)
    _set(r, S.TWO_WEAPON_COMBAT,_S)
    _set(r, S.MARTIAL_ARTS,     _S)

    # ---- VALKYRIE (11) — Skill_V, u_init.c:525 ----
    r = int(Role.VALKYRIE)
    _set(r, S.DAGGER,           _E)
    _set(r, S.AXE,              _E)
    _set(r, S.PICK_AXE,         _S)
    _set(r, S.SHORT_SWORD,      _S)
    _set(r, S.BROAD_SWORD,      _S)
    _set(r, S.LONG_SWORD,       _E)
    _set(r, S.TWO_HANDED_SWORD, _E)
    _set(r, S.SABER,            _B)
    _set(r, S.HAMMER,           _E)
    _set(r, S.QUARTERSTAFF,     _B)
    _set(r, S.POLEARMS,         _S)
    _set(r, S.SPEAR,            _E)
    _set(r, S.TRIDENT,          _B)
    _set(r, S.LANCE,            _S)
    _set(r, S.SLING,            _B)
    _set(r, S.ATTACK_SPELL,     _B)
    _set(r, S.ESCAPE_SPELL,     _B)
    _set(r, S.RIDING,           _S)
    _set(r, S.TWO_WEAPON_COMBAT,_S)
    _set(r, S.MARTIAL_ARTS,     _E)

    # ---- WIZARD (12) — Skill_W, u_init.c:548 ----
    r = int(Role.WIZARD)
    _set(r, S.DAGGER,           _E)
    _set(r, S.KNIFE,            _S)
    _set(r, S.AXE,              _S)
    _set(r, S.SHORT_SWORD,      _B)
    _set(r, S.CLUB,             _S)
    _set(r, S.MACE,             _B)
    _set(r, S.QUARTERSTAFF,     _E)
    _set(r, S.POLEARMS,         _S)
    _set(r, S.SPEAR,            _B)
    _set(r, S.TRIDENT,          _B)
    _set(r, S.SLING,            _S)
    _set(r, S.DART,             _E)
    _set(r, S.SHURIKEN,         _B)
    _set(r, S.ATTACK_SPELL,     _E)
    _set(r, S.HEALING_SPELL,    _S)
    _set(r, S.DIVINATION_SPELL, _E)
    _set(r, S.ENCHANTMENT_SPELL,_S)
    _set(r, S.CLERIC_SPELL,     _S)
    _set(r, S.ESCAPE_SPELL,     _E)
    _set(r, S.MATTER_SPELL,     _E)
    _set(r, S.RIDING,           _B)
    _set(r, S.MARTIAL_ARTS,     _B)

    return jnp.array(caps, dtype=jnp.int8)


_ROLE_SKILL_CAPS: jnp.ndarray = _build_role_skill_caps()


# ---------------------------------------------------------------------------
# Vendor P_* skill id (1-based) → Nethax SkillId (0-based) mapping.
# skills.h: P_DAGGER=1 … P_UNICORN_HORN=27; P_ATTACK_SPELL=28…P_MATTER_SPELL=34;
# P_BARE_HANDED_COMBAT=35, P_TWO_WEAPON_COMBAT=36, P_RIDING=37.
# P_NONE=0 and out-of-range values → MARTIAL_ARTS (slot 34) as safe default.
# ---------------------------------------------------------------------------
_VENDOR_SKILL_TO_NETHAX: list[int] = [
    int(SkillId.MARTIAL_ARTS),   # 0 = P_NONE / unset
    int(SkillId.DAGGER),         # 1 = P_DAGGER
    int(SkillId.KNIFE),          # 2 = P_KNIFE
    int(SkillId.AXE),            # 3 = P_AXE
    int(SkillId.PICK_AXE),       # 4 = P_PICK_AXE
    int(SkillId.SHORT_SWORD),    # 5 = P_SHORT_SWORD
    int(SkillId.BROAD_SWORD),    # 6 = P_BROAD_SWORD
    int(SkillId.LONG_SWORD),     # 7 = P_LONG_SWORD
    int(SkillId.TWO_HANDED_SWORD),# 8 = P_TWO_HANDED_SWORD
    int(SkillId.SABER),          # 9 = P_SABER
    int(SkillId.CLUB),           # 10 = P_CLUB
    int(SkillId.MACE),           # 11 = P_MACE
    int(SkillId.MORNING_STAR),   # 12 = P_MORNING_STAR
    int(SkillId.FLAIL),          # 13 = P_FLAIL
    int(SkillId.HAMMER),         # 14 = P_HAMMER
    int(SkillId.QUARTERSTAFF),   # 15 = P_QUARTERSTAFF
    int(SkillId.POLEARMS),       # 16 = P_POLEARMS
    int(SkillId.SPEAR),          # 17 = P_SPEAR
    int(SkillId.TRIDENT),        # 18 = P_TRIDENT
    int(SkillId.LANCE),          # 19 = P_LANCE
    int(SkillId.BOW),            # 20 = P_BOW
    int(SkillId.SLING),          # 21 = P_SLING
    int(SkillId.CROSSBOW),       # 22 = P_CROSSBOW
    int(SkillId.DART),           # 23 = P_DART
    int(SkillId.SHURIKEN),       # 24 = P_SHURIKEN
    int(SkillId.BOOMERANG),      # 25 = P_BOOMERANG
    int(SkillId.WHIP),           # 26 = P_WHIP
    int(SkillId.UNICORN_HORN),   # 27 = P_UNICORN_HORN
    int(SkillId.ATTACK_SPELL),   # 28 = P_ATTACK_SPELL
    int(SkillId.HEALING_SPELL),  # 29 = P_HEALING_SPELL
    int(SkillId.DIVINATION_SPELL),# 30 = P_DIVINATION_SPELL
    int(SkillId.ENCHANTMENT_SPELL),# 31 = P_ENCHANTMENT_SPELL
    int(SkillId.CLERIC_SPELL),   # 32 = P_CLERIC_SPELL
    int(SkillId.ESCAPE_SPELL),   # 33 = P_ESCAPE_SPELL
    int(SkillId.MATTER_SPELL),   # 34 = P_MATTER_SPELL
    int(SkillId.MARTIAL_ARTS),   # 35 = P_BARE_HANDED_COMBAT
    int(SkillId.TWO_WEAPON_COMBAT),# 36 = P_TWO_WEAPON_COMBAT
    int(SkillId.RIDING),         # 37 = P_RIDING
]


def _build_weapon_type_to_skill() -> jnp.ndarray:
    """Build int32[NUM_OBJECTS] mapping each object type_id → SkillId.

    Reads oc_skill from OBJECTS table (vendor P_* 1-based values, or negative
    for ammo linked to a launcher class — those default to MARTIAL_ARTS/0).
    Cite: vendor/nethack/include/objclass.h — oc_skill = oc_subtyp for weapons.
    """
    from Nethax.nethax.constants.objects import OBJECTS, NUM_OBJECTS
    mapping = []
    lut = _VENDOR_SKILL_TO_NETHAX
    for obj in OBJECTS:
        v = int(obj.oc_skill)
        if 0 <= v < len(lut):
            mapping.append(lut[v])
        else:
            mapping.append(int(SkillId.MARTIAL_ARTS))
    return jnp.array(mapping, dtype=jnp.int32)


_WEAPON_TYPE_TO_SKILL: jnp.ndarray = _build_weapon_type_to_skill()


# ---------------------------------------------------------------------------
# SpellSchool → SkillId mapping
# vendor/nethack/include/skills.h SpellSchool enum maps directly.
# SpellSchool in magic.py uses 0-based; SkillId.ATTACK_SPELL=27 etc.
# ---------------------------------------------------------------------------
_SPELL_SCHOOL_TO_SKILL_ID: jnp.ndarray = jnp.array(
    [
        int(SkillId.ATTACK_SPELL),       # SpellSchool.ATTACK_SPELL      = 0
        int(SkillId.HEALING_SPELL),      # SpellSchool.HEALING_SPELL     = 1
        int(SkillId.DIVINATION_SPELL),   # SpellSchool.DIVINATION_SPELL  = 2
        int(SkillId.ENCHANTMENT_SPELL),  # SpellSchool.ENCHANTMENT_SPELL = 3
        int(SkillId.CLERIC_SPELL),       # SpellSchool.CLERIC_SPELL      = 4
        int(SkillId.ESCAPE_SPELL),       # SpellSchool.ESCAPE_SPELL      = 5
        int(SkillId.MATTER_SPELL),       # SpellSchool.MATTER_SPELL      = 6
    ],
    dtype=jnp.int32,
)


# ---------------------------------------------------------------------------
# SkillState
# ---------------------------------------------------------------------------
@struct.dataclass
class SkillState:
    """Per-character skill advancement state.

    Fields
    ------
    level     : current skill tier per skill (int8, SkillLevel values, 0-based)
    advance   : practice counter toward next tier (int32)
    max_level : per-role cap (int8, SkillLevel values)
    """
    level:     jnp.ndarray  # [N_SKILLS] int8
    advance:   jnp.ndarray  # [N_SKILLS] int32
    max_level: jnp.ndarray  # [N_SKILLS] int8

    @classmethod
    def default(cls) -> "SkillState":
        """Return a zero-initialised SkillState (all P_UNSKILLED, no caps).

        The role-aware initialisation happens in env.reset() via init_skills().
        """
        return cls(
            level=jnp.zeros((N_SKILLS,), dtype=jnp.int8),
            advance=jnp.zeros((N_SKILLS,), dtype=jnp.int32),
            max_level=jnp.zeros((N_SKILLS,), dtype=jnp.int8),
        )


# ---------------------------------------------------------------------------
# use_skill  (vendor/nethack/src/weapon.c:1424)
# ---------------------------------------------------------------------------
def use_skill(state, skill_id: jnp.ndarray, degree: int = 1):
    """Increment practice counter for ``skill_id`` by ``degree``.

    Cite: vendor/nethack/src/weapon.c:1424 — use_skill() increments
    u.weapon_skills[skill].advance by the given amount.

    JIT-pure: no Python branches on traced values.
    """
    sid = jnp.clip(skill_id.astype(jnp.int32), 0, N_SKILLS - 1)
    skills = state.skills
    new_advance = skills.advance.at[sid].add(jnp.int32(degree))
    new_skills = skills.replace(advance=new_advance)
    return state.replace(skills=new_skills)


# ---------------------------------------------------------------------------
# try_advance_skill  (vendor/nethack/src/weapon.c::skill_advance)
# ---------------------------------------------------------------------------
def try_advance_skill(state, skill_id: jnp.ndarray):
    """Advance skill tier if practice threshold is met and cap allows.

    Cite: vendor/nethack/src/weapon.c::skill_advance — checks advance >=
    practice_needed_to_advance(level) and max_skill > current skill.

    JIT-pure.
    """
    sid = jnp.clip(skill_id.astype(jnp.int32), 0, N_SKILLS - 1)
    skills = state.skills
    cur_level   = skills.level[sid].astype(jnp.int32)
    cur_advance = skills.advance[sid].astype(jnp.int32)
    cap         = skills.max_level[sid].astype(jnp.int32)
    threshold   = practice_needed_to_advance(cur_level)
    can_advance = (cur_advance >= threshold) & (cur_level < cap)
    new_level = jnp.where(can_advance, cur_level + jnp.int32(1), cur_level).astype(jnp.int8)
    new_level_arr = skills.level.at[sid].set(new_level)
    new_skills = skills.replace(level=new_level_arr)
    return state.replace(skills=new_skills)


# ---------------------------------------------------------------------------
# init_skills  — role-aware initialisation for env.reset()
# ---------------------------------------------------------------------------
def init_skills(role) -> SkillState:
    """Return starting SkillState for the given role.

    Sets max_level from _ROLE_SKILL_CAPS and initialises level to
    P_UNSKILLED everywhere (starting practice levels are handled by
    weapon.c::init_weapons which we do not yet model; keeping them at 0
    is conservative and consistent with the default).

    Additional vendor parity: roles whose ``petnum == PM_PONY`` (Knight)
    start with ``P_SKILL(P_RIDING) = P_BASIC`` per vendor
    ``weapon.c::skill_init`` lines 1787-1789:

        /* Roles that start with a horse know how to ride it */
        if (gu.urole.petnum == PM_PONY)
            P_SKILL(P_RIDING) = P_BASIC;

    Cite: vendor/nethack/src/u_init.c — Skill_X tables;
          vendor/nethack/src/weapon.c::skill_init lines 1737-1810.
    """
    from Nethax.nethax.constants.roles import get_role, PM_PONY

    role_idx = int(role) if not isinstance(role, int) else role
    caps = _ROLE_SKILL_CAPS[role_idx]  # [N_SKILLS] int8

    # Vendor weapon.c:1787-1789 — pony-starting roles begin at P_BASIC riding.
    initial_levels = jnp.zeros((N_SKILLS,), dtype=jnp.int8)
    role_entry = get_role(role_idx)
    if role_entry.petnum == PM_PONY:
        initial_levels = initial_levels.at[int(SkillId.RIDING)].set(
            jnp.int8(SkillLevel.P_BASIC)
        )

    return SkillState(
        level=initial_levels,
        advance=jnp.zeros((N_SKILLS,), dtype=jnp.int32),
        max_level=caps,
    )
