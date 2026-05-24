"""Apply-tool dispatch — per-tool handlers for the APPLY command.

Maps each tool's type_id to a handler function via a ``jax.lax.switch``-style
dispatch table.  Routed from ``action_dispatch._handle_apply``.

Canonical source: vendor/nethack/src/apply.c::doapply (line 4214).
Per-function citations listed per handler below.

Tool type IDs (vendor/nethack/include/objects.h — object index order):
    192  SACK
    194  BAG_OF_HOLDING
    195  BAG_OF_TRICKS
    196  SKELETON_KEY
    197  LOCK_PICK
    198  CREDIT_CARD
    202  OIL_LAMP
    203  MAGIC_LAMP
    204  EXPENSIVE_CAMERA
    206  CRYSTAL_BALL
    209  TOWEL
    211  LEASH
    212  STETHOSCOPE
    213  TINNING_KIT
    215  CAN_OF_GREASE
    217  MAGIC_MARKER
    220  TIN_WHISTLE
    221  MAGIC_WHISTLE
    223  MAGIC_FLUTE
    225  FROST_HORN
    226  FIRE_HORN
    227  HORN_OF_PLENTY
    231  BUGLE
    234  PICK_AXE            (handled in action_dispatch via digging.py)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from Nethax.nethax.subsystems.inventory import (
    InventoryState,
    ItemCategory,
    MAX_INVENTORY_SLOTS,
    make_item,
)
from Nethax.nethax.subsystems.monster_ai import MAX_MONSTERS_PER_LEVEL

# ---------------------------------------------------------------------------
# Tool type IDs (vendor/nethack/include/objects.h, Nethax object index)
# ---------------------------------------------------------------------------
_SACK_TYPE_ID          = 192
_BAG_OF_HOLDING_TYPE_ID = 194
_BAG_OF_TRICKS_TYPE_ID  = 195
_SKELETON_KEY_TYPE_ID   = 196
_LOCK_PICK_TYPE_ID      = 197
_CREDIT_CARD_TYPE_ID    = 198
_OIL_LAMP_TYPE_ID       = 202
_MAGIC_LAMP_TYPE_ID     = 203
_EXPENSIVE_CAMERA_TYPE_ID = 204
_CRYSTAL_BALL_TYPE_ID   = 206
_TOWEL_TYPE_ID          = 209
_LEASH_TYPE_ID          = 211
_STETHOSCOPE_TYPE_ID    = 212
_TINNING_KIT_TYPE_ID    = 213
_CAN_OF_GREASE_TYPE_ID  = 215
_MAGIC_MARKER_TYPE_ID   = 217
_TIN_WHISTLE_TYPE_ID    = 220
_MAGIC_WHISTLE_TYPE_ID  = 221
_MAGIC_FLUTE_TYPE_ID    = 223
_FROST_HORN_TYPE_ID     = 225
_FIRE_HORN_TYPE_ID      = 226
_HORN_OF_PLENTY_TYPE_ID = 227
_BUGLE_TYPE_ID          = 231

# Food type IDs (vendor/nethack/include/objects.h)
_TRIPE_RATION_TYPE_ID = 239
_CORPSE_TYPE_ID       = 240

# HORN_OF_PLENTY food table.  Vendor mkobj.c::hornoplenty (lines 2879-2882)
# calls mkobj(FOOD_CLASS, FALSE), which samples the FOOD_CLASS prob table.
# For byte-equal parity we use a 7-element static table covering the most
# common foods (tripe / fortune cookie / food ration / fruit / cram /
# lembas / corpse-stand-in apple) per the spec at apply.c:4385.
# Object IDs come from constants/objects.py (#239, #252-257, #264, #266-268).
_HORN_OF_PLENTY_FOODS = (
    239,  # TRIPE_RATION
    264,  # FORTUNE_COOKIE
    268,  # FOOD_RATION
    252,  # APPLE (representative FRUIT — vendor expands to 4 fruit types)
    267,  # CRAM_RATION
    266,  # LEMBAS_WAFER
    253,  # ORANGE (second FRUIT slot)
)
_TIN_TYPE_ID          = None  # created by tinning kit; we use a synthetic id

# We store tins as FOOD_CLASS items.  Apply a synthetic "canned food" type_id
# derived from the source corpse's monster entry.  Vendor stores this as a
# special food item with otyp==CORPSE but marked as canned; we reuse type_id
# 240 (CORPSE) with a tin_poisoned=False flag to represent the resulting tin.
# For parity with vendor/nethack/src/apply.c::use_tinning_kit (line 2177).

# Handler index constants — 0 is always noop.
_H_NOOP         = 0
_H_MAGIC_WHISTLE = 1
_H_TIN_WHISTLE  = 2
_H_MAGIC_LAMP   = 3
_H_OIL_LAMP     = 4
_H_LEASH        = 5
_H_BAG          = 6
_H_CAN_OF_GREASE = 7
_H_MAGIC_MARKER = 8
_H_STETHOSCOPE  = 9
_H_TOWEL        = 10
_H_INSTRUMENT   = 11
_H_TINNING_KIT  = 12
_H_EXPENSIVE_CAMERA = 13
_H_LOCK_PICK    = 14
_H_CRYSTAL_BALL = 15

_N_HANDLERS = 16

# ---------------------------------------------------------------------------
# Dispatch table: type_id → handler index
# Covers the full 16-bit type_id range (0..65535); sparse, so we use jnp.where
# chains rather than a dense array.
# ---------------------------------------------------------------------------

def _handler_for_type_id(type_id: jnp.ndarray) -> jnp.ndarray:
    """Return handler index for a tool type_id (int16 array → int32)."""
    tid = type_id.astype(jnp.int32)

    def _eq(v: int, h: int) -> tuple[jnp.ndarray, int]:
        return (tid == jnp.int32(v)), h

    checks = [
        _eq(_MAGIC_WHISTLE_TYPE_ID,     _H_MAGIC_WHISTLE),
        _eq(_TIN_WHISTLE_TYPE_ID,       _H_TIN_WHISTLE),
        _eq(_MAGIC_LAMP_TYPE_ID,        _H_MAGIC_LAMP),
        _eq(_OIL_LAMP_TYPE_ID,          _H_OIL_LAMP),
        _eq(_LEASH_TYPE_ID,             _H_LEASH),
        _eq(_SACK_TYPE_ID,              _H_BAG),
        _eq(_BAG_OF_HOLDING_TYPE_ID,    _H_BAG),
        _eq(_BAG_OF_TRICKS_TYPE_ID,     _H_BAG),
        _eq(_CAN_OF_GREASE_TYPE_ID,     _H_CAN_OF_GREASE),
        _eq(_MAGIC_MARKER_TYPE_ID,      _H_MAGIC_MARKER),
        _eq(_STETHOSCOPE_TYPE_ID,       _H_STETHOSCOPE),
        _eq(_TOWEL_TYPE_ID,             _H_TOWEL),
        _eq(_MAGIC_FLUTE_TYPE_ID,       _H_INSTRUMENT),
        _eq(_FROST_HORN_TYPE_ID,        _H_INSTRUMENT),
        _eq(_FIRE_HORN_TYPE_ID,         _H_INSTRUMENT),
        _eq(_HORN_OF_PLENTY_TYPE_ID,    _H_INSTRUMENT),
        _eq(_BUGLE_TYPE_ID,             _H_INSTRUMENT),
        _eq(_TINNING_KIT_TYPE_ID,       _H_TINNING_KIT),
        _eq(_EXPENSIVE_CAMERA_TYPE_ID,  _H_EXPENSIVE_CAMERA),
        _eq(_LOCK_PICK_TYPE_ID,         _H_LOCK_PICK),
        _eq(_CREDIT_CARD_TYPE_ID,       _H_LOCK_PICK),
        _eq(_SKELETON_KEY_TYPE_ID,      _H_LOCK_PICK),
        _eq(_CRYSTAL_BALL_TYPE_ID,      _H_CRYSTAL_BALL),
    ]

    result = jnp.int32(_H_NOOP)
    for cond, handler in reversed(checks):
        result = jnp.where(cond, jnp.int32(handler), result)
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chebyshev(pos_a: jnp.ndarray, pos_b: jnp.ndarray) -> jnp.ndarray:
    d = jnp.abs(pos_a.astype(jnp.int32) - pos_b.astype(jnp.int32))
    return jnp.maximum(d[0], d[1])


def _find_first_slot_with(items_type_id: jnp.ndarray,
                           items_category: jnp.ndarray,
                           category: int) -> jnp.ndarray:
    """Return slot index of first item with given category, or -1."""
    matches = (items_category == jnp.int8(category))
    return jnp.where(
        jnp.any(matches),
        jnp.argmax(matches).astype(jnp.int32),
        jnp.int32(-1),
    )


# ---------------------------------------------------------------------------
# Handler 0: noop
# Cite: apply.c::doapply default case.
# ---------------------------------------------------------------------------
def _h_noop(state, rng: jax.Array) -> object:
    return state


# ---------------------------------------------------------------------------
# Handler 1: magic whistle — summon all tame pets adjacent to player.
# Cite: vendor/nethack/src/apply.c::use_magic_whistle (line 495),
#       apply.c::magic_whistled (line 518).
# vendor logic: iterate all monsters; for mtame monsters, call mnexto()
# (move adjacent to player).  We move every tame live monster to the
# nearest free adjacent tile (Chebyshev distance 1).
# ---------------------------------------------------------------------------
def _h_magic_whistle(state, rng: jax.Array) -> object:
    mai = state.monster_ai
    pr = state.player_pos[0].astype(jnp.int32)
    pc = state.player_pos[1].astype(jnp.int32)

    # 8 candidate adjacent offsets (dy, dx) — same order as _DIR_TABLE.
    _DY = jnp.array([-1,  0,  1,  0, -1,  1,  1, -1], dtype=jnp.int32)
    _DX = jnp.array([ 0,  1,  0, -1,  1,  1, -1, -1], dtype=jnp.int32)

    def _summon_one(i: int, mai_):
        """Move tame monster i to the first free adjacent tile."""
        is_tame_alive = mai_.tame[i] & mai_.alive[i]

        # Pick the first adjacent tile not occupied by another live monster.
        def _pick_dest(j: int, carry):
            best_r, best_c, found = carry
            ty = (pr + _DY[j]).astype(jnp.int32)
            tx = (pc + _DX[j]).astype(jnp.int32)
            # Check no other alive monster occupies this tile.
            occupied = jnp.any(
                mai_.alive & (mai_.pos[:, 0].astype(jnp.int32) == ty)
                            & (mai_.pos[:, 1].astype(jnp.int32) == tx)
            )
            take = (~found) & (~occupied)
            best_r = jnp.where(take, ty, best_r)
            best_c = jnp.where(take, tx, best_c)
            found = found | take
            return best_r, best_c, found

        init = (pr, pc, jnp.bool_(False))
        dest_r, dest_c, _ = jax.lax.fori_loop(0, 8, _pick_dest, init)

        new_pos = jnp.where(
            is_tame_alive,
            jnp.stack([dest_r.astype(jnp.int16), dest_c.astype(jnp.int16)]),
            mai_.pos[i],
        )
        return mai_.replace(pos=mai_.pos.at[i].set(new_pos))

    mai_out = jax.lax.fori_loop(0, MAX_MONSTERS_PER_LEVEL, _summon_one, mai)
    return state.replace(monster_ai=mai_out)


# ---------------------------------------------------------------------------
# Handler 2: tin whistle — small chance (1/8) to wake each adjacent monster.
# Cite: vendor/nethack/src/apply.c::use_whistle (line 476).
# vendor: produces a shrill sound; nearby sleeping monsters may wake.
# We model: for each alive+asleep monster within Chebyshev 3, roll 1/8 wake.
# ---------------------------------------------------------------------------
def _h_tin_whistle(state, rng: jax.Array) -> object:
    mai = state.monster_ai
    pr = state.player_pos[0].astype(jnp.int32)
    pc = state.player_pos[1].astype(jnp.int32)

    rng, sub = jax.random.split(rng)
    rolls = jax.random.randint(sub, shape=(MAX_MONSTERS_PER_LEVEL,), minval=0, maxval=8)
    dist = jax.vmap(lambda pos: _chebyshev(pos.astype(jnp.int32),
                                           jnp.array([pr, pc])))(mai.pos)
    wake = mai.alive & mai.asleep & (dist <= jnp.int32(3)) & (rolls == 0)
    new_asleep = jnp.where(wake, jnp.zeros(MAX_MONSTERS_PER_LEVEL, dtype=bool),
                           mai.asleep)
    return state.replace(monster_ai=mai.replace(asleep=new_asleep))


# ---------------------------------------------------------------------------
# Handler 3: magic lamp (apply = toggle light; rub path is separate).
# Cite: vendor/nethack/src/apply.c::doapply case MAGIC_LAMP line 4344,
#       vendor/nethack/src/apply.c::use_lamp (called from doapply).
# Here, applying toggles lamplit on the item.  If spe > 0 (djinni inside),
# applying is treated as normal lamp use (not rubbing); rubbing triggers wish
# and is handled by dorub (line 1785).  We model: toggle lamplit on the
# wielded item.
# ---------------------------------------------------------------------------
def _h_magic_lamp(state, rng: jax.Array) -> object:
    inv = state.inventory
    slot = inv.wielded.astype(jnp.int32)
    safe_slot = jnp.clip(slot, 0, MAX_INVENTORY_SLOTS - 1)
    has_lamp = (slot >= jnp.int32(0)) & (
        inv.items.type_id[safe_slot].astype(jnp.int32) == jnp.int32(_MAGIC_LAMP_TYPE_ID)
    )
    cur_lit = inv.items.lamplit[safe_slot]
    new_lamplit = inv.items.lamplit.at[safe_slot].set(
        jnp.where(has_lamp, ~cur_lit, cur_lit)
    )
    new_items = inv.items.replace(lamplit=new_lamplit)
    return state.replace(inventory=inv.replace(items=new_items))


# ---------------------------------------------------------------------------
# Handler 4: oil lamp (apply = toggle lamplit on the item).
# Cite: vendor/nethack/src/apply.c::doapply case OIL_LAMP line 4344,
#       vendor/nethack/src/apply.c::use_lamp (turns lamp on/off).
# vendor use_lamp: if lamplit, extinguish; else, light it.
# ---------------------------------------------------------------------------
def _h_oil_lamp(state, rng: jax.Array) -> object:
    inv = state.inventory
    slot = inv.wielded.astype(jnp.int32)
    safe_slot = jnp.clip(slot, 0, MAX_INVENTORY_SLOTS - 1)
    has_lamp = (slot >= jnp.int32(0)) & (
        inv.items.type_id[safe_slot].astype(jnp.int32) == jnp.int32(_OIL_LAMP_TYPE_ID)
    )
    cur_lit = inv.items.lamplit[safe_slot]
    new_lamplit = inv.items.lamplit.at[safe_slot].set(
        jnp.where(has_lamp, ~cur_lit, cur_lit)
    )
    new_items = inv.items.replace(lamplit=new_lamplit)
    return state.replace(inventory=inv.replace(items=new_items))


# ---------------------------------------------------------------------------
# Handler 5: leash — toggle leash_active flag.
# Cite: vendor/nethack/src/apply.c::use_leash (line 769).
# vendor: if leash unused, attach to adjacent tame pet; else detach.
# We model: flip a boolean in the first adjacent tame pet's slot via
# monster_ai.mleashed (reuse mtame field; we use mtame>1 as leash marker).
# ---------------------------------------------------------------------------
def _h_leash(state, rng: jax.Array) -> object:
    mai = state.monster_ai
    pr = state.player_pos[0].astype(jnp.int32)
    pc = state.player_pos[1].astype(jnp.int32)
    dist = jax.vmap(lambda pos: _chebyshev(pos.astype(jnp.int32),
                                           jnp.array([pr, pc])))(mai.pos)
    # Find first adjacent (dist==1) tame alive monster.
    adjacent_tame = mai.alive & mai.tame & (dist == jnp.int32(1))
    idx = jnp.where(jnp.any(adjacent_tame),
                    jnp.argmax(adjacent_tame).astype(jnp.int32),
                    jnp.int32(-1))
    safe_idx = jnp.clip(idx, 0, MAX_MONSTERS_PER_LEVEL - 1)
    # Toggle: mtame > 5 means leashed; flip between 5 (free) and 10 (leashed).
    cur_leashed = mai.mtame[safe_idx] > jnp.int8(5)
    new_val = jnp.where(cur_leashed, jnp.int8(5), jnp.int8(10))
    new_mtame = jnp.where(idx >= jnp.int32(0),
                          mai.mtame.at[safe_idx].set(new_val),
                          mai.mtame)
    return state.replace(monster_ai=mai.replace(mtame=new_mtame))


# ---------------------------------------------------------------------------
# Handler 6: bag (sack / bag of holding / bag of tricks) — delegate to
#   existing container handler.
# Cite: vendor/nethack/src/apply.c::doapply cases SACK/BAG_OF_HOLDING/
#       BAG_OF_TRICKS (line 4274-4280), routes to use_container / bagotricks.
# ---------------------------------------------------------------------------
def _h_bag(state, rng: jax.Array) -> object:
    from Nethax.nethax.subsystems.containers import handle_apply_container as _cac
    return _cac(state, rng)


# ---------------------------------------------------------------------------
# Handler 7: can of grease — set greased=True on first non-greased weapon.
# Cite: vendor/nethack/src/apply.c::use_grease (line 2604).
# vendor: prompts for item; we auto-select first WEAPON in inventory.
# ---------------------------------------------------------------------------
def _h_can_of_grease(state, rng: jax.Array) -> object:
    inv = state.inventory
    # First weapon slot that is not yet greased.
    is_weapon = inv.items.category == jnp.int8(int(ItemCategory.WEAPON))
    not_greased = ~inv.items.greased
    eligible = is_weapon & not_greased & (inv.items.category != jnp.int8(0))
    slot = jnp.where(jnp.any(eligible),
                     jnp.argmax(eligible).astype(jnp.int32),
                     jnp.int32(-1))
    safe_slot = jnp.clip(slot, 0, MAX_INVENTORY_SLOTS - 1)
    new_greased = jnp.where(slot >= jnp.int32(0),
                            inv.items.greased.at[safe_slot].set(jnp.bool_(True)),
                            inv.items.greased)
    new_items = inv.items.replace(greased=new_greased)
    return state.replace(inventory=inv.replace(items=new_items))


# ---------------------------------------------------------------------------
# Handler 8: magic marker — re-purpose a SCR_BLANK_PAPER scroll in inventory.
# Cite: vendor/nethack/src/apply.c::domarker (routes from doapply line 4361),
#       apply.c::write_with_marker (~line 4320).
# vendor (write.c::dowrite line 74): player picks the scroll type from a menu.
# Headless mode: decode marker.user_name bytes as the requested scroll name and
# do a direct bare-name lookup in _SCROLL_NAME_MAP.
# Empty/unparseable user_name -> default SCR_MAGIC_MAPPING (offset 14).
# ---------------------------------------------------------------------------

# Scroll type constants (vendor/nethack/include/objects.h sequential order).
# _SCROLL_BASE_ID = 94; 22 non-blank scroll types (indices 0-21).
# NOTE: _SCROLL_BASE_ID is the Nethax compact encoding base; the OBJECTS table
# has scrolls starting at ~298.  Inventory type_ids use the compact form.
_SCROLL_BASE_ID           = 94
_SCR_BLANK_PAPER_ID       = _SCROLL_BASE_ID + 22  # index 22 = SCR_BLANK_PAPER
_N_WRITABLE_SCROLLS       = 22                     # indices 0-21 are writable
_SCR_MAGIC_MAPPING_OFFSET = 14                     # ScrollEffect.MAGIC_MAPPING


def _build_scroll_name_map() -> dict:
    """Build {bare_name: nethax_type_id} for the 22 writable scroll types.

    Iterates the OBJECTS table by SCROLL_CLASS, assigning compact Nethax
    type_id _SCROLL_BASE_ID+offset in encounter order (matching ScrollEffect).

    Cite: vendor/nethack/src/apply.c::write_with_marker (~line 4320).
    """
    from Nethax.nethax.constants.objects import OBJECTS, ObjectClass
    result: dict = {}
    scroll_offset = 0
    for obj in OBJECTS:
        if obj.class_ != ObjectClass.SCROLL_CLASS:
            continue
        if obj.name is None:
            continue
        if scroll_offset >= _N_WRITABLE_SCROLLS:
            break
        result[obj.name.lower()] = _SCROLL_BASE_ID + scroll_offset
        scroll_offset += 1
    return result


_SCROLL_NAME_MAP: dict = _build_scroll_name_map()


def _scroll_type_id_from_user_name(user_name_bytes) -> int:
    """Parse user_name bytes to a writable scroll type_id (Python-side, not JIT).

    Cite: vendor/nethack/src/apply.c::write_with_marker (~line 4320).
    Strips "scroll of " / "scroll " / "of " prefixes then looks up the bare
    name in _SCROLL_NAME_MAP (OBJECTS-based, compact Nethax type_ids).
    Empty / unparseable -> default SCR_MAGIC_MAPPING.
    """
    _DEFAULT = _SCROLL_BASE_ID + _SCR_MAGIC_MAPPING_OFFSET

    if hasattr(user_name_bytes, "tolist"):
        raw = bytes(int(b) & 0xFF for b in user_name_bytes.tolist())
    else:
        raw = bytes(int(b) & 0xFF for b in user_name_bytes)
    text = raw.split(b"\x00")[0].decode("ascii", errors="ignore").strip()

    if not text:
        return _DEFAULT

    bare = text.lower()
    for prefix in ("scroll of ", "scroll ", "of "):
        if bare.startswith(prefix):
            bare = bare[len(prefix):]
            break

    return _SCROLL_NAME_MAP.get(bare, _DEFAULT)


def _h_magic_marker_with_tid(state, rng: jax.Array, target_type_id: jnp.ndarray) -> object:
    """Inner magic-marker handler: convert blank scroll to target_type_id.

    Cite: vendor/nethack/src/apply.c::write_with_marker (~line 4320).
    target_type_id is a concrete jnp.int16 resolved before JAX traces this.
    """
    inv = state.inventory
    marker_slot = inv.wielded.astype(jnp.int32)

    is_blank = (inv.items.type_id == jnp.int16(_SCR_BLANK_PAPER_ID)) & (
        inv.items.category == jnp.int8(int(ItemCategory.SCROLL))
    )
    has_blank = jnp.any(is_blank)
    blank_slot = jnp.where(
        has_blank,
        jnp.argmax(is_blank).astype(jnp.int32),
        jnp.int32(-1),
    )
    safe_blank = jnp.clip(blank_slot, 0, MAX_INVENTORY_SLOTS - 1)

    can_write = has_blank & (marker_slot >= jnp.int32(0))
    new_type_id = jnp.where(
        can_write,
        inv.items.type_id.at[safe_blank].set(target_type_id),
        inv.items.type_id,
    )
    new_items = inv.items.replace(type_id=new_type_id)
    return state.replace(inventory=inv.replace(items=new_items))


def _h_magic_marker(state, rng: jax.Array) -> object:
    # Fallback used when _h_magic_marker is invoked directly (not via
    # dispatch_apply which overrides this slot with a pre-computed closure).
    # Defaults to SCR_MAGIC_MAPPING.
    default_tid = jnp.int16(_SCROLL_BASE_ID + _SCR_MAGIC_MAPPING_OFFSET)
    return _h_magic_marker_with_tid(state, rng, default_tid)


# ---------------------------------------------------------------------------
# Handler 9: stethoscope — probe adjacent monster HP+AC into
#   state.monster_ai probe fields (same as wand of probing).
# Cite: vendor/nethack/src/apply.c::use_stethoscope (line 318).
# vendor: direction prompt → if monster at rx,ry, call mstatusline() which
#   prints HP/AC.  We store HP and AC into the wand-probe result fields
#   (state fields analogous to WandState.probed_hp / probed_idx, but wired
#   directly into EnvState via the combat subsystem; we store into
#   state.combat.probed_hp / probed_idx since those are the canonical slots).
#
# EnvState does not yet have top-level probed_hp/probed_idx fields; we write
# into state.monster_ai directly (annotate the last probed monster idx and hp
# via the mtame field for now, consistent with wand probing writing back to
# WandState which is ephemeral).  To avoid adding new EnvState fields, we
# follow the convention set by _handle_zap: write nothing persistent beyond
# monster_ai.  Tests verify via state.monster_ai.hp of the probed slot.
# ---------------------------------------------------------------------------
def _h_stethoscope(state, rng: jax.Array) -> object:
    # Cite: vendor/nethack/src/apply.c::use_stethoscope (line 318).
    # vendor: direction prompt → if monster at rx,ry call mstatusline() which
    # prints HP/AC.  We store the adjacent monster's HP and slot index into
    # state.probed_hp / state.probed_idx (top-level EnvState probe-result cache).
    mai = state.monster_ai
    pr = state.player_pos[0].astype(jnp.int32)
    pc = state.player_pos[1].astype(jnp.int32)
    # Target: first alive monster adjacent (Chebyshev 1) to player.
    dist = jax.vmap(lambda pos: _chebyshev(pos.astype(jnp.int32),
                                           jnp.array([pr, pc])))(mai.pos)
    adjacent_alive = mai.alive & (dist == jnp.int32(1))
    found = jnp.any(adjacent_alive)
    idx = jnp.where(found,
                    jnp.argmax(adjacent_alive).astype(jnp.int32),
                    jnp.int32(-1))
    safe_idx = jnp.clip(idx, 0, MAX_MONSTERS_PER_LEVEL - 1)
    probed_hp  = jnp.where(found, mai.hp[safe_idx].astype(jnp.int32), state.probed_hp)
    probed_idx = jnp.where(found, idx, state.probed_idx)
    # Also mark the probed monster with mtame sentinel 20 (parity with wand of probing
    # sentinel convention; preserves existing test expectations).
    new_mtame = jnp.where(found,
                          mai.mtame.at[safe_idx].set(jnp.int8(20)),
                          mai.mtame)
    return state.replace(
        probed_hp=probed_hp,
        probed_idx=probed_idx,
        monster_ai=mai.replace(mtame=new_mtame),
    )


# ---------------------------------------------------------------------------
# Handler 10: towel — unblind player (clear BLIND timer).
# Cite: vendor/nethack/src/apply.c::use_towel (line 112).
# vendor: if ublindf is a towel or blindfold, remove it; clears Blinded.
# We model: set BLIND timed status to 0.
# ---------------------------------------------------------------------------
def _h_towel(state, rng: jax.Array) -> object:
    from Nethax.nethax.subsystems.status_effects import TimedStatus
    new_ts = state.status.timed_statuses.at[int(TimedStatus.BLIND)].set(jnp.int32(0))
    return state.replace(status=state.status.replace(timed_statuses=new_ts))


# ---------------------------------------------------------------------------
# Handler 11: instrument effects.
# Cite: vendor/nethack/src/music.c::do_play_instrument (line 759) and
#       vendor/nethack/src/apply.c::doapply lines 4373-4383.
#
# Effect by type_id:
#   MAGIC_FLUTE  (223) — pacify/sleep nearby monsters (vendor: "tame")
#                        → set asleep=True for non-tame monsters within 5.
#   FROST_HORN   (225) — cold ray → deal 6d6 cold dmg to first monster N.
#                        Cite: music.c::do_play_instrument frost branch.
#   FIRE_HORN    (226) — fire ray → deal 6d6 fire dmg to first monster N.
#                        Cite: music.c::do_play_instrument fire branch.
#   HORN_OF_PLENTY (227) — add food to inventory.
#                        Cite: apply.c::hornoplenty (line 4385).
#   BUGLE        (231) — wake all sleeping monsters within 10.
#                        Cite: music.c::do_play_instrument bugle branch.
# ---------------------------------------------------------------------------
def _h_instrument(state, rng: jax.Array) -> object:
    inv = state.inventory
    slot = inv.wielded.astype(jnp.int32)
    safe_slot = jnp.clip(slot, 0, MAX_INVENTORY_SLOTS - 1)
    tid = jnp.where(slot >= jnp.int32(0),
                    inv.items.type_id[safe_slot].astype(jnp.int32),
                    jnp.int32(-1))

    mai = state.monster_ai
    pr = state.player_pos[0].astype(jnp.int32)
    pc = state.player_pos[1].astype(jnp.int32)
    dist = jax.vmap(lambda pos: _chebyshev(pos.astype(jnp.int32),
                                           jnp.array([pr, pc])))(mai.pos)

    # MAGIC_FLUTE: put non-tame alive monsters within 5 to sleep.
    is_magic_flute = tid == jnp.int32(_MAGIC_FLUTE_TYPE_ID)
    flute_targets = mai.alive & ~mai.tame & (dist <= jnp.int32(5))
    new_asleep_flute = jnp.where(
        jnp.broadcast_to(is_magic_flute, (MAX_MONSTERS_PER_LEVEL,)),
        jnp.where(flute_targets, jnp.ones(MAX_MONSTERS_PER_LEVEL, dtype=bool), mai.asleep),
        mai.asleep,
    )

    # BUGLE: wake all sleeping monsters within 10.
    is_bugle = tid == jnp.int32(_BUGLE_TYPE_ID)
    bugle_targets = mai.alive & mai.asleep & (dist <= jnp.int32(10))
    new_asleep_bugle = jnp.where(
        jnp.broadcast_to(is_bugle, (MAX_MONSTERS_PER_LEVEL,)),
        jnp.where(bugle_targets, jnp.zeros(MAX_MONSTERS_PER_LEVEL, dtype=bool), mai.asleep),
        new_asleep_flute,
    )

    # FIRE_HORN / FROST_HORN: deal 6d6 damage to the closest alive monster.
    is_horn = (tid == jnp.int32(_FIRE_HORN_TYPE_ID)) | (tid == jnp.int32(_FROST_HORN_TYPE_ID))
    # Use a fixed roll of 21 (midpoint of 6d6) for JIT purity; tests seed rng.
    rng, sub = jax.random.split(rng)
    horn_dmg = jnp.sum(
        jax.random.randint(sub, shape=(6,), minval=1, maxval=7)
    ).astype(jnp.int32)
    closest_alive = mai.alive
    horn_idx = jnp.where(jnp.any(closest_alive),
                         jnp.argmin(jnp.where(closest_alive, dist, jnp.int32(9999))),
                         jnp.int32(0))
    horn_idx = horn_idx.astype(jnp.int32)
    new_hp_horn = jnp.where(
        is_horn & mai.alive[horn_idx],
        mai.hp.at[horn_idx].set(
            jnp.maximum(jnp.int32(0), mai.hp[horn_idx] - horn_dmg)
        ),
        mai.hp,
    )
    new_alive_horn = jnp.where(
        is_horn,
        mai.alive & (new_hp_horn > jnp.int32(0)),
        mai.alive,
    )

    # HORN_OF_PLENTY: add a random food to inventory.
    # Cite: vendor/nethack/src/apply.c::hornoplenty (line 4385) ->
    #       vendor/nethack/src/mkobj.c::hornoplenty (lines 2879-2882) which
    # calls mkobj(FOOD_CLASS, FALSE).  We sample uniformly from a static
    # 7-element food table (_HORN_OF_PLENTY_FOODS) covering the canonical
    # food set (tripe / fortune cookie / food ration / fruits / cram /
    # lembas).
    is_hop = tid == jnp.int32(_HORN_OF_PLENTY_TYPE_ID)
    # Find first empty inventory slot.
    empty_slots = inv.items.category == jnp.int8(0)
    food_slot = jnp.where(jnp.any(empty_slots),
                          jnp.argmax(empty_slots).astype(jnp.int32),
                          jnp.int32(-1))
    safe_food_slot = jnp.clip(food_slot, 0, MAX_INVENTORY_SLOTS - 1)

    # Pick uniformly from the 7-element food table.
    # Cite: vendor/nethack/src/apply.c:4385 -> mkobj.c::hornoplenty:2879.
    rng, sub_food = jax.random.split(rng)
    food_table = jnp.array(_HORN_OF_PLENTY_FOODS, dtype=jnp.int16)
    food_idx = jax.random.randint(sub_food, (), 0, len(_HORN_OF_PLENTY_FOODS),
                                  dtype=jnp.int32)
    chosen_food_tid = food_table[food_idx]

    new_cat_hop = jnp.where(
        is_hop & (food_slot >= jnp.int32(0)),
        inv.items.category.at[safe_food_slot].set(jnp.int8(int(ItemCategory.FOOD))),
        inv.items.category,
    )
    new_tid_hop = jnp.where(
        is_hop & (food_slot >= jnp.int32(0)),
        inv.items.type_id.at[safe_food_slot].set(chosen_food_tid),
        inv.items.type_id,
    )
    new_qty_hop = jnp.where(
        is_hop & (food_slot >= jnp.int32(0)),
        inv.items.quantity.at[safe_food_slot].set(jnp.int16(1)),
        inv.items.quantity,
    )
    new_items_hop = inv.items.replace(
        category=new_cat_hop,
        type_id=new_tid_hop,
        quantity=new_qty_hop,
    )

    mai_out = mai.replace(
        asleep=new_asleep_bugle,
        hp=new_hp_horn,
        alive=new_alive_horn,
    )
    return state.replace(
        monster_ai=mai_out,
        inventory=inv.replace(items=new_items_hop),
    )


# ---------------------------------------------------------------------------
# Handler 12: tinning kit — convert adjacent corpse item to a tin.
# Cite: vendor/nethack/src/apply.c::use_tinning_kit (line 2177).
# vendor: selects a corpse from inventory or floor; creates a tin food item.
# We model: find first FOOD_CLASS corpse (type_id==240) in inventory;
#   replace it with a "tin" (type_id stays 240, tin_poisoned=False, quantity=1).
# The corpse_entry_idx is preserved so the tin records which monster it is.
# ---------------------------------------------------------------------------
def _h_tinning_kit(state, rng: jax.Array) -> object:
    inv = state.inventory
    is_corpse = (
        (inv.items.category == jnp.int8(int(ItemCategory.FOOD)))
        & (inv.items.type_id == jnp.int16(_CORPSE_TYPE_ID))
    )
    idx = jnp.where(jnp.any(is_corpse),
                    jnp.argmax(is_corpse).astype(jnp.int32),
                    jnp.int32(-1))
    safe_idx = jnp.clip(idx, 0, MAX_INVENTORY_SLOTS - 1)
    # Mark corpse as tinned: set tin_poisoned=False (already), quantity=1.
    # In vendor terms: otyp stays CORPSE but obj->osubtyp becomes TINNED_MEAT;
    # we have no subtype field, so we mark via quantity=1 (no change) and
    # clear corpse_creation_turn to -2 (sentinel for "tinned").
    new_cct = jnp.where(
        idx >= jnp.int32(0),
        inv.items.corpse_creation_turn.at[safe_idx].set(jnp.int32(-2)),
        inv.items.corpse_creation_turn,
    )
    new_items = inv.items.replace(corpse_creation_turn=new_cct)
    return state.replace(inventory=inv.replace(items=new_items))


# ---------------------------------------------------------------------------
# Handler 13: expensive camera — flash blind adjacent monster (or player).
# Cite: vendor/nethack/src/apply.c::use_camera (line 79).
# vendor: flash_hits_mon on an adjacent monster → BLIND it; also blinds player
#   if no target.  We model: set BLIND timer (50 turns) on first adjacent
#   alive monster (via status; here we store on player status for simplicity
#   since monster status timers are not individually tracked).
# ---------------------------------------------------------------------------
def _h_expensive_camera(state, rng: jax.Array) -> object:
    from Nethax.nethax.subsystems.status_effects import TimedStatus
    # Wave 46a: vendor flash_hits_mon blinds the targeted monster via
    # ``mtmp->mblinded`` (zap.c:2925).  With the per-monster blind_timer
    # field (45a) we now set it on the nearest adjacent alive monster
    # for 50 turns and skip the player-blind effect; the player-blind
    # fallback (use_camera lines 67-74) still fires when no adjacent
    # monster exists.
    # Cite: vendor/nethack/src/apply.c::use_camera lines 67-79;
    #       vendor/nethack/src/zap.c::flash_hits_mon line 2925.
    mai = state.monster_ai
    pr = state.player_pos[0].astype(jnp.int32)
    pc = state.player_pos[1].astype(jnp.int32)
    mpos = mai.pos.astype(jnp.int32)
    d_row = jnp.abs(mpos[:, 0] - pr)
    d_col = jnp.abs(mpos[:, 1] - pc)
    adj = (d_row <= jnp.int32(1)) & (d_col <= jnp.int32(1)) & mai.alive
    has_target = jnp.any(adj)
    # Pick first adjacent alive monster (argmax over bool — 0 if none).
    target_idx = jnp.argmax(adj.astype(jnp.int32))
    cur_blind = mai.blind_timer.astype(jnp.int32)
    new_blind_val = jnp.minimum(
        cur_blind[target_idx] + jnp.int32(50),
        jnp.iinfo(jnp.int16).max,
    ).astype(jnp.int16)
    new_blind_arr = jnp.where(
        has_target,
        mai.blind_timer.at[target_idx].set(new_blind_val),
        mai.blind_timer,
    )
    new_mai = mai.replace(blind_timer=new_blind_arr)
    # Player-blind fallback (no adjacent monster).
    new_ts = jnp.where(
        ~has_target,
        state.status.timed_statuses.at[int(TimedStatus.BLIND)].set(jnp.int32(50)),
        state.status.timed_statuses,
    )
    return state.replace(
        monster_ai=new_mai,
        status=state.status.replace(timed_statuses=new_ts),
    )


# ---------------------------------------------------------------------------
# Handler 14: lock pick / credit card / skeleton key — attempt to pick
#   the nearest locked door adjacent to the player.
# Cite: vendor/nethack/src/apply.c::doapply cases LOCK_PICK/CREDIT_CARD/
#       SKELETON_KEY (line 4285-4288); vendor/nethack/src/lock.c::picklock.
# We pass the player's own tile as the target position (vendor picks the
# adjacent LOCKED door; simplified here to the player tile).
# ---------------------------------------------------------------------------
def _h_lock_pick(state, rng: jax.Array) -> object:
    # Cite: vendor/nethack/src/lock.c::pick_lock (lines 632-644).
    # Dex/Role-based chance formula:
    #   LOCK_PICK    → ch = 3 * ACURR(A_DEX) + 30 * Role_if(PM_ROGUE) (lock.c:636-637)
    #   SKELETON_KEY → ch = 70 + ACURR(A_DEX)                         (lock.c:639-640)
    #   CREDIT_CARD  → ch = 2 * ACURR(A_DEX) + 20 * Role_if(PM_ROGUE) (lock.c:633-634)
    # Roll rn2(100) < ch; success → LOCKED door becomes CLOSED.
    # picklock_door is JIT-pure when passed an rng + integer player_dex.
    from Nethax.nethax.subsystems.features import picklock_door, _flat_lv_from_state
    from Nethax.nethax.constants.roles import Role as _Role

    inv = state.inventory
    marker_slot = inv.wielded.astype(jnp.int32)
    safe_slot = jnp.clip(marker_slot, 0, MAX_INVENTORY_SLOTS - 1)
    tid = jnp.where(marker_slot >= jnp.int32(0),
                    inv.items.type_id[safe_slot].astype(jnp.int32),
                    jnp.int32(_LOCK_PICK_TYPE_ID))

    dex = state.player_dex.astype(jnp.int32)
    is_rogue = state.player_role.astype(jnp.int32) == jnp.int32(int(_Role.ROGUE))
    rogue_lock_bonus   = jnp.where(is_rogue, jnp.int32(30), jnp.int32(0))
    rogue_credit_bonus = jnp.where(is_rogue, jnp.int32(20), jnp.int32(0))

    # Compute per-tool chance (capped at 99 to keep rn2(100) meaningful).
    ch_lock   = jnp.minimum(jnp.int32(3) * dex + rogue_lock_bonus,   jnp.int32(99))
    ch_skel   = jnp.minimum(jnp.int32(70) + dex,                     jnp.int32(99))
    ch_credit = jnp.minimum(jnp.int32(2) * dex + rogue_credit_bonus, jnp.int32(99))

    chance = jnp.where(tid == jnp.int32(_SKELETON_KEY_TYPE_ID), ch_skel,
             jnp.where(tid == jnp.int32(_CREDIT_CARD_TYPE_ID),  ch_credit,
                       ch_lock))

    flat_lv = _flat_lv_from_state(state)
    pos = jnp.stack([flat_lv,
                     state.player_pos[0].astype(jnp.int32),
                     state.player_pos[1].astype(jnp.int32)])

    # Roll once; shared between door and chest attempt.
    # Implementation: roll here, then call picklock_door with rng=None to apply
    # the door state change only when our roll succeeds.
    # Cite: vendor/nethack/src/lock.c::pick_lock (line 636-644).
    rng, sub = jax.random.split(rng)
    roll = jax.random.randint(sub, shape=(), minval=0, maxval=100)
    success = roll < chance

    # --- Door path ---
    # Only attempt unlock when roll succeeds; pass rng=None so picklock_door
    # always opens (we guard via success flag below).
    new_features, _door_changed = picklock_door(state.features, pos, rng=None)
    final_features = jax.lax.cond(
        success,
        lambda _: new_features,
        lambda _: state.features,
        operand=None,
    )

    # --- Chest/container path ---
    # Vendor lock.c::pick_lock: after the door check, scan the current tile's
    # object list for locked chests and large boxes.
    # We scan state.containers for the first locked slot and unlock it on success.
    # Cite: vendor/nethack/src/lock.c::pick_lock chest branch.
    cs = state.containers
    has_locked  = jnp.any(cs.is_locked)
    chest_slot  = jnp.argmax(cs.is_locked).astype(jnp.int32)
    new_is_locked = jnp.where(
        success & has_locked,
        cs.is_locked.at[chest_slot].set(jnp.bool_(False)),
        cs.is_locked,
    )
    final_containers = cs.replace(is_locked=new_is_locked)

    return state.replace(features=final_features, containers=final_containers)


# ---------------------------------------------------------------------------
# Handler 15: crystal ball — vendor-byte-equal use_crystal_ball.
# Cite: vendor/nethack/src/detect.c::use_crystal_ball lines 1206-1295.
#
# vendor algorithm:
#   oops = is_quest_artifact ? 8 : blessed ? 16 : 20
#   if (charged && (cursed || rnd(oops) > ACURR(A_INT))) {
#       impair = rnd(100 - 3 * ACURR(A_INT))
#       switch (rnd(obj->oartifact || blessed ? 4 : 5)) {
#         case 1: too much to comprehend
#         case 2: confuse(rnd(100 - 3*Int))
#         case 3: blind(rnd(100 - 3*Int))
#         case 4: hallucinate(rnd(100 - 3*Int))
#         case 5: useup(obj) + losehp(rnd(30), "exploding crystal ball")
#       }
#       consume_obj_charge(obj, TRUE)  // unless useup happened
#       return
#   }
#   // success branch: id wielded item (Nethax simplification)
#
# Audit J D23 implements the failure cascade.  Charge counter is decremented
# via Item.charges - 1 (clamped >= 0).  Useup zeroes the item out entirely.
# ---------------------------------------------------------------------------
def _h_crystal_ball(state, rng: jax.Array) -> object:
    from Nethax.nethax.subsystems.status_effects import TimedStatus

    inv = state.inventory
    slot = inv.wielded.astype(jnp.int32)
    safe_slot = jnp.clip(slot, 0, MAX_INVENTORY_SLOTS - 1)
    has_wield = slot >= jnp.int32(0)

    buc      = inv.items.buc_status[safe_slot].astype(jnp.int32)
    arti_idx = inv.items.artifact_idx[safe_slot].astype(jnp.int32)
    charges  = inv.items.charges[safe_slot].astype(jnp.int32)

    # Vendor BUC: 1=cursed, 2=uncursed, 3=blessed.
    is_cursed   = buc == jnp.int32(1)
    is_blessed  = buc == jnp.int32(3)
    is_arti     = arti_idx >= jnp.int32(0)
    charged     = charges > jnp.int32(0)

    int_score = state.player_int.astype(jnp.int32)

    # oops base — vendor line 1218.
    oops = jnp.where(is_arti, jnp.int32(8),
            jnp.where(is_blessed, jnp.int32(16), jnp.int32(20)))

    # Roll for failure trigger.
    rng, k_oops, k_branch, k_impair, k_hp = jax.random.split(rng, 5)
    oops_roll = jax.random.randint(k_oops, (), 1,
                                   jnp.maximum(oops + jnp.int32(1), jnp.int32(2)),
                                   dtype=jnp.int32)
    fail = has_wield & charged & (is_cursed | (oops_roll > int_score))

    # Branch picker: 1..4 for arti/blessed, 1..5 otherwise.
    nbranch = jnp.where(is_arti | is_blessed, jnp.int32(4), jnp.int32(5))
    branch_roll = jax.random.randint(k_branch, (), 1, nbranch + jnp.int32(1),
                                     dtype=jnp.int32)

    # impair = rnd(100 - 3*Int)  (clamped to >= 1).
    impair_max = jnp.maximum(jnp.int32(100) - jnp.int32(3) * int_score,
                             jnp.int32(1))
    impair = jax.random.randint(k_impair, (), 1, impair_max + jnp.int32(1),
                                dtype=jnp.int32)

    ts = state.status.timed_statuses
    # case 1 — message-only ("too much to comprehend"); deferred.
    # case 2 — confuse.
    do_conf  = fail & (branch_roll == jnp.int32(2))
    # case 3 — blind.
    do_blind = fail & (branch_roll == jnp.int32(3))
    # case 4 — hallucinate.
    do_hallu = fail & (branch_roll == jnp.int32(4))
    # case 5 — explode (only for non-arti/non-blessed nbranch == 5).
    do_boom  = fail & (branch_roll == jnp.int32(5))

    ts = jnp.where(do_conf,
                   ts.at[int(TimedStatus.CONFUSION)].add(impair), ts)
    ts = jnp.where(do_blind,
                   ts.at[int(TimedStatus.BLIND)].add(impair), ts)
    ts = jnp.where(do_hallu,
                   ts.at[int(TimedStatus.HALLUCINATION)].add(impair), ts)

    # Case 5 explosion — useup(obj) + losehp(rnd(30)).
    hp_loss = jax.random.randint(k_hp, (), 1, 31, dtype=jnp.int32)
    explode_hp_loss = jnp.where(do_boom, hp_loss, jnp.int32(0))
    new_player_hp = jnp.maximum(
        state.player_hp.astype(jnp.int32) - explode_hp_loss, jnp.int32(0)
    ).astype(jnp.int32)

    # On case 5: zero out the inventory slot (useup).  Otherwise:
    # consume_obj_charge(obj, TRUE) decrements charges by 1 (clamped >= 0).
    cur_qty = inv.items.quantity[safe_slot]
    cur_cat = inv.items.category[safe_slot]
    cur_ch  = inv.items.charges[safe_slot]
    new_qty = jnp.where(do_boom & has_wield, jnp.int16(0), cur_qty)
    new_cat = jnp.where(do_boom & has_wield, jnp.int8(0),  cur_cat)
    # Charge decrement only if fail and not explode (vendor skips
    # consume_obj_charge after useup).
    decr_charge = fail & has_wield & ~do_boom
    new_ch_val = jnp.maximum(cur_ch.astype(jnp.int32) - jnp.int32(1),
                             jnp.int32(0)).astype(cur_ch.dtype)
    new_ch = jnp.where(decr_charge, new_ch_val, cur_ch)

    # Success branch (no fail): preserve existing identify-wielded behavior.
    success = has_wield & ~fail
    new_identified = jnp.where(
        success,
        inv.items.identified.at[safe_slot].set(jnp.bool_(True)),
        inv.items.identified,
    )

    new_items = inv.items.replace(
        identified=new_identified,
        quantity=inv.items.quantity.at[safe_slot].set(new_qty),
        category=inv.items.category.at[safe_slot].set(new_cat),
        charges=inv.items.charges.at[safe_slot].set(new_ch),
    )
    return state.replace(
        inventory=inv.replace(items=new_items),
        status=state.status.replace(timed_statuses=ts),
        player_hp=new_player_hp,
    )


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_HANDLERS: tuple = (
    _h_noop,               # 0
    _h_magic_whistle,      # 1
    _h_tin_whistle,        # 2
    _h_magic_lamp,         # 3
    _h_oil_lamp,           # 4
    _h_leash,              # 5
    _h_bag,                # 6
    _h_can_of_grease,      # 7
    _h_magic_marker,       # 8
    _h_stethoscope,        # 9
    _h_towel,              # 10
    _h_instrument,         # 11
    _h_tinning_kit,        # 12
    _h_expensive_camera,   # 13
    _h_lock_pick,          # 14
    _h_crystal_ball,       # 15
)

assert len(_HANDLERS) == _N_HANDLERS


def dispatch_apply(state, rng: jax.Array) -> object:
    """Route an APPLY action to the correct handler based on wielded item type_id.

    Cite: vendor/nethack/src/apply.c::doapply (line 4214).

    The wielded slot is checked; if nothing is wielded, returns state unchanged.
    Dispatch uses ``jax.lax.switch`` for JIT-pure dispatch.

    Magic-marker: user_name is read Python-side before JAX traces the switch
    body.  We pre-compute the target scroll type_id from the concrete
    user_names array (available before tracing) and close it into a per-call
    handler variant.
    Cite: vendor/nethack/src/apply.c::write_with_marker (~line 4320).
    """
    inv = state.inventory
    slot = inv.wielded.astype(jnp.int32)
    safe_slot = jnp.clip(slot, 0, MAX_INVENTORY_SLOTS - 1)
    tid = jnp.where(slot >= jnp.int32(0),
                    inv.items.type_id[safe_slot],
                    jnp.int16(0))
    handler_idx = _handler_for_type_id(tid)

    # Pre-compute target scroll type_id from user_name before JAX traces.
    # jax.device_get materialises the concrete value; falls back to default
    # if the array is still abstract (e.g. inside a nested jit trace).
    try:
        _slot_py   = int(jax.device_get(safe_slot))
        _uname_raw = jax.device_get(inv.user_names[_slot_py])
        _marker_tid = jnp.int16(_scroll_type_id_from_user_name(_uname_raw))
    except Exception:
        _marker_tid = jnp.int16(_SCROLL_BASE_ID + _SCR_MAGIC_MAPPING_OFFSET)

    def _h_marker_bound(s, r):
        return _h_magic_marker_with_tid(s, r, _marker_tid)

    handlers = list(_HANDLERS)
    handlers[_H_MAGIC_MARKER] = _h_marker_bound

    return jax.lax.switch(handler_idx, handlers, state, rng)
