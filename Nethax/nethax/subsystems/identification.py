"""Identification subsystem — per-run appearance shuffling and type-level ID.

Canonical sources:
  vendor/nethack/src/o_init.c  — init_objects(), shuffle_all(), shuffle();
                                  Fisher-Yates-style description shuffle at
                                  game start; assigns random descriptions to
                                  potion/scroll/wand/ring/amulet/spellbook
                                  types.
  vendor/nethack/src/objnam.c  — xname(), doname(); consults identification
                                  tables to decide what the player sees.
  vendor/nethack/src/insight.c — partial_id(), enlightenment(); reveals
                                  subset of item properties without full ID.

Status: Wave 6 #77 — vendor-parity Fisher-Yates shuffle of description indices
across each shuffled class.  Mirrors o_init.c::shuffle (lines 113-148) which
walks each unidentified slot and swaps oc_descr_idx with a random later slot.

Item type counts verified against vendor/nethack/include/objects.h
(POTION/SCROLL/WAND/RING/AMULET/SPELL macro count, 2025-05-12):

  N_POTION_TYPES    = 26  (POT_GAIN_ABILITY .. POT_WATER)
  N_SCROLL_TYPES    = 43  (21 real + 20 extra labels + SCR_BLANK_PAPER;
                            SCR_MAIL excluded — conditional on MAIL_STRUCTURES)
  N_WAND_TYPES      = 28  (25 real + 3 extra shuffle descriptions WAN1-3)
  N_RING_TYPES      = 28  (RIN_ADORNMENT .. RIN_PROTECTION_FROM_SHAPE_CHAN)
  N_AMULET_TYPES    = 13  (11 real + FAKE_AMULET_OF_YENDOR + AMULET_OF_YENDOR)
  N_SPELLBOOK_TYPES = 46  (43 spells + SPE_BLANK_PAPER + SPE_NOVEL
                            + SPE_BOOK_OF_THE_DEAD)

Notes on vendor parity (Wave 6 #77)
-----------------------------------
* Vendor's ``shuffle()`` only shuffles slots whose ``oc_name_known`` flag is
  unset, and (for the potion class) it excludes the final slot (potion of
  water has a fixed description — see ``obj_shuffle_range`` in o_init.c).
  At game start no items are pre-identified, so for our purposes every slot
  participates except POT_WATER (the last potion slot).
* The vendor algorithm walks ``j`` from low to high; for each j it picks
  ``i = j + rn2(o_high - j + 1)`` and swaps ``oc_descr_idx[i] <-> [j]``.
  This is the in-place Fisher-Yates variant: identical seeds produce
  identical shuffles.
* We mirror this exactly using a sequence of ``jax.random.randint`` draws
  derived from a single user-supplied PRNG key (which itself is derived
  from the game seed).  Same key → identical permutation.

TODO (future):
  - Bones-file appearance persistence across games (not applicable to JAX sim)
  - Wand appearance also encodes material + shape — currently we only model
    the description index (which is what gameplay-visible name uses).
"""
import jax
import jax.numpy as jnp
from flax import struct


# ---------------------------------------------------------------------------
# Item type counts
# (source: vendor/nethack/include/objects.h — verified 2025-05-12)
# ---------------------------------------------------------------------------

N_POTION_TYPES    = 26   # POT_GAIN_ABILITY .. POT_WATER
N_SCROLL_TYPES    = 43   # 21 real + 20 XTRA_SCROLL_LABEL + SCR_BLANK_PAPER
N_WAND_TYPES      = 28   # 25 real wands + WAN1, WAN2, WAN3 (extra descriptions)
N_RING_TYPES      = 28   # RIN_ADORNMENT .. RIN_PROTECTION_FROM_SHAPE_CHAN
N_AMULET_TYPES    = 13   # 11 magic + FAKE_AMULET_OF_YENDOR + AMULET_OF_YENDOR
N_SPELLBOOK_TYPES = 46   # 43 spells + SPE_BLANK_PAPER + SPE_NOVEL + SPE_BOOK_OF_THE_DEAD

# Indices in vendor potions list where POT_WATER lives (last entry — its
# description is fixed and is excluded from the shuffle by obj_shuffle_range).
POT_WATER_INDEX = N_POTION_TYPES - 1

# Total object types in the OBJECTS table.  Derived at import time so the
# value stays in sync if the vendor objects list is regenerated.
# Cite: vendor/nethack/include/decl.h LAST_OBJECT.
def _num_objects_at_import() -> int:
    from Nethax.nethax.constants.objects import OBJECTS
    return len(OBJECTS)


NUM_OBJECTS: int = _num_objects_at_import()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@struct.dataclass
class IdentificationState:
    """Per-run identification and appearance-shuffle tables.

    All appearance arrays store a permutation: appearance_id[type_id] = N,
    meaning item of type_id looks like appearance N to the player.
    The inverse mapping (what type does appearance N actually correspond to?)
    is exposed via ``type_for_appearance`` below.

    Fields
    ------
    potion_appearance   : int8[N_POTION_TYPES]    shuffled appearance indices
    scroll_appearance   : int8[N_SCROLL_TYPES]    shuffled label indices
    wand_appearance     : int8[N_WAND_TYPES]      shuffled material indices
    ring_appearance     : int8[N_RING_TYPES]      shuffled stone indices
    amulet_appearance   : int8[N_AMULET_TYPES]    shuffled shape indices
    spellbook_appearance: int8[N_SPELLBOOK_TYPES] shuffled cover indices
    identified          : bool[NUM_OBJECTS]       type-level full ID flag
    """

    potion_appearance:    jnp.ndarray   # [N_POTION_TYPES]    int8
    scroll_appearance:    jnp.ndarray   # [N_SCROLL_TYPES]    int8
    wand_appearance:      jnp.ndarray   # [N_WAND_TYPES]      int8
    ring_appearance:      jnp.ndarray   # [N_RING_TYPES]      int8
    amulet_appearance:    jnp.ndarray   # [N_AMULET_TYPES]    int8
    spellbook_appearance: jnp.ndarray   # [N_SPELLBOOK_TYPES] int8
    identified:           jnp.ndarray   # [NUM_OBJECTS]       bool

    # Wave 6 #79: detection-spell timers.  Each holds the turn at which the
    # corresponding detection effect expires (-1 = never active).
    # Cite: vendor/nethack/src/detect.c::monster_detect / food_detect /
    #       trap_detect (used here for "detect treasure").
    detect_monsters_until_turn:  jnp.ndarray  # scalar int32
    detect_food_until_turn:      jnp.ndarray  # scalar int32
    detect_treasure_until_turn:  jnp.ndarray  # scalar int32
    detect_objects_until_turn:   jnp.ndarray  # scalar int32
    detect_magic_until_turn:     jnp.ndarray  # scalar int32

    @classmethod
    def unshuffled(cls) -> "IdentificationState":
        """Return identity-permutation state (type N looks like appearance N).

        DEBUG / TEST FALLBACK ONLY.  Real game start always calls
        ``init_shuffled_appearances`` to derive a random Fisher-Yates
        permutation (o_init.c::shuffle_all).  ``unshuffled`` is used by
        unit tests that need a deterministic identity mapping and by
        early-init code paths that want a placeholder before the game
        seed is available.  All items start unidentified.
        """
        return cls(
            potion_appearance=jnp.arange(N_POTION_TYPES, dtype=jnp.int8),
            scroll_appearance=jnp.arange(N_SCROLL_TYPES, dtype=jnp.int8),
            wand_appearance=jnp.arange(N_WAND_TYPES, dtype=jnp.int8),
            ring_appearance=jnp.arange(N_RING_TYPES, dtype=jnp.int8),
            amulet_appearance=jnp.arange(N_AMULET_TYPES, dtype=jnp.int8),
            spellbook_appearance=jnp.arange(N_SPELLBOOK_TYPES, dtype=jnp.int8),
            identified=jnp.zeros((NUM_OBJECTS,), dtype=jnp.bool_),
            detect_monsters_until_turn=jnp.int32(-1),
            detect_food_until_turn=jnp.int32(-1),
            detect_treasure_until_turn=jnp.int32(-1),
            detect_objects_until_turn=jnp.int32(-1),
            detect_magic_until_turn=jnp.int32(-1),
        )


# ---------------------------------------------------------------------------
# Vendor-parity shuffle (o_init.c::shuffle, lines 113-148)
# ---------------------------------------------------------------------------

def _shuffle_class(rng: jax.Array, n: int, exclude_last: bool = False) -> jnp.ndarray:
    """Fisher-Yates shuffle of [0..n) mirroring vendor o_init.c::shuffle.

    Vendor algorithm (o_init.c lines 125-147):

        for (j = o_low; j <= o_high; j++) {
            if (objects[j].oc_name_known) continue;
            do
                i = j + rn2(o_high - j + 1);
            while (objects[i].oc_name_known);
            swap(objects[j].oc_descr_idx, objects[i].oc_descr_idx);
            ...
        }

    Because no items are pre-identified at game start, the ``oc_name_known``
    guard is trivially false and the inner do/while loop becomes a single
    draw.  Our implementation therefore performs:

        for j in 0..n-1:
            i = j + rn2(n - j)
            swap(perm[j], perm[i])

    which is exactly the in-place Fisher-Yates shuffle.  The number of RNG
    draws (``n``) and the swap semantics match vendor exactly, so identical
    seeds yield identical permutations.

    Parameters
    ----------
    rng : JAX PRNG key (consumed; do not reuse).
    n   : number of slots in the class (Python int, static).
    exclude_last : when True, the final slot is held fixed (used for
                   POTION_CLASS — POT_WATER has a fixed description, see
                   obj_shuffle_range() in o_init.c).

    Returns
    -------
    int8[n] permutation array.
    """
    perm = jnp.arange(n, dtype=jnp.int8)
    upper = n - 1 if exclude_last else n
    if upper <= 1:
        return perm
    # Pre-split: one subkey per swap step (vendor: one rn2 per j).
    keys = jax.random.split(rng, upper)
    for j in range(upper):
        # i = j + rn2(upper - j), inclusive lower j, exclusive upper.
        i_offset = jax.random.randint(keys[j], (), 0, upper - j)
        i = j + i_offset
        # Swap perm[j] and perm[i] using functional updates.
        pj = perm[j]
        pi = perm[i]
        perm = perm.at[j].set(pi)
        perm = perm.at[i].set(pj)
    return perm


def init_shuffled_appearances(rng: jax.Array) -> IdentificationState:
    """Produce a freshly shuffled IdentificationState for a new game.

    Mirrors o_init.c::init_objects() → shuffle_all() (lines 322-347): each
    shufflable object class gets its description indices Fisher-Yates
    shuffled.  RNG key splits are stable, so identical seeds produce
    identical mappings (vendor-parity property).

    Order of class shuffles matches vendor shuffle_all() static array:
        AMULET, POTION, RING, SCROLL, SPELLBOOK, WAND
    (VENOM_CLASS is in vendor's list but doesn't have a Nethax type yet.)

    Cite: vendor/nethack/src/o_init.c::shuffle_all lines 322-347.

    Parameters
    ----------
    rng : JAX PRNG key derived from the game seed.

    Returns
    -------
    IdentificationState with shuffled per-class appearance arrays.
    """
    # Six subkeys, one per shuffled class.
    keys = jax.random.split(rng, 6)
    amulet_perm    = _shuffle_class(keys[0], N_AMULET_TYPES)
    potion_perm    = _shuffle_class(keys[1], N_POTION_TYPES, exclude_last=True)
    ring_perm      = _shuffle_class(keys[2], N_RING_TYPES)
    scroll_perm    = _shuffle_class(keys[3], N_SCROLL_TYPES)
    spellbook_perm = _shuffle_class(keys[4], N_SPELLBOOK_TYPES)
    wand_perm      = _shuffle_class(keys[5], N_WAND_TYPES)

    return IdentificationState(
        potion_appearance=potion_perm,
        scroll_appearance=scroll_perm,
        wand_appearance=wand_perm,
        ring_appearance=ring_perm,
        amulet_appearance=amulet_perm,
        spellbook_appearance=spellbook_perm,
        identified=jnp.zeros((NUM_OBJECTS,), dtype=jnp.bool_),
        detect_monsters_until_turn=jnp.int32(-1),
        detect_food_until_turn=jnp.int32(-1),
        detect_treasure_until_turn=jnp.int32(-1),
        detect_objects_until_turn=jnp.int32(-1),
        detect_magic_until_turn=jnp.int32(-1),
    )


# ---------------------------------------------------------------------------
# Appearance / canonical name lookups
# ---------------------------------------------------------------------------

# Canonical / unidentified description pools.  These are static module-level
# constants — same across all runs.  Cite: vendor scroll_names[], potion_descrs
# (drawing.c / objects.c).  The pool sizes match the class counts above so
# that ``appearance[type_id]`` indexes directly into the pool.

# Scrolls: 23 fake words + extra labels (vendor scroll_names[] has 23 random
# labels chosen from extra pool of 30+).  We use 43 labels to match
# N_SCROLL_TYPES so every shuffled slot has a label.
_SCROLL_LABELS = tuple(f"label-{i:02d}" for i in range(N_SCROLL_TYPES))

# Potion colors.
_POTION_COLORS = (
    "ruby", "pink", "orange", "yellow", "emerald", "dark green",
    "cyan", "sky blue", "brilliant blue", "magenta", "purple-red",
    "puce", "milky", "swirly", "bubbly", "smoky", "cloudy", "effervescent",
    "black", "golden", "brown", "fizzy", "dark", "white", "murky", "clear",
)
assert len(_POTION_COLORS) == N_POTION_TYPES

# Wand materials.
_WAND_MATERIALS = (
    "wooden", "iron", "glass", "balsa", "crystal", "maple", "pine", "oak",
    "ebony", "marble", "tin", "brass", "copper", "silver", "platinum",
    "iridium", "zinc", "aluminum", "uranium", "steel", "hexagonal",
    "short", "runed", "long", "curved", "forked", "spiked", "jeweled",
)
assert len(_WAND_MATERIALS) == N_WAND_TYPES

# Ring stones.
_RING_STONES = (
    "pearl", "iron", "twisted", "steel", "wire", "engagement", "shiny",
    "bronze", "brass", "copper", "silver", "gold", "wooden", "granite",
    "opal", "clay", "coral", "black onyx", "moonstone", "tiger eye",
    "jade", "agate", "topaz", "sapphire", "ruby", "diamond", "ivory", "emerald",
)
assert len(_RING_STONES) == N_RING_TYPES

# Amulet shapes.
_AMULET_SHAPES = (
    "circular", "spherical", "oval", "triangular", "pyramidal", "square",
    "concave", "hexagonal", "octagonal", "obtuse", "wooden", "amber",
    "Amulet of Yendor",
)
assert len(_AMULET_SHAPES) == N_AMULET_TYPES

# Spellbook covers.
_SPELLBOOK_COVERS = tuple(f"cover-{i:02d}" for i in range(N_SPELLBOOK_TYPES))


_POOLS = {
    "potion":    _POTION_COLORS,
    "scroll":    _SCROLL_LABELS,
    "wand":      _WAND_MATERIALS,
    "ring":      _RING_STONES,
    "amulet":    _AMULET_SHAPES,
    "spellbook": _SPELLBOOK_COVERS,
}


def unidentified_appearance(
    state: IdentificationState,
    obj_class: str,
    type_id: int,
) -> str:
    """Return the shuffled appearance string the player sees for type_id.

    Looks up the per-class permutation array in ``state`` to find the
    appearance index, then maps that index to the canonical static pool.

    Parameters
    ----------
    state     : IdentificationState produced by ``init_shuffled_appearances``.
    obj_class : one of {"potion","scroll","wand","ring","amulet","spellbook"}.
    type_id   : canonical type index (0..N_<CLASS>_TYPES-1).

    Returns
    -------
    Python string — appearance label.  Side-channel only (host-side); not
    used inside jitted paths.
    """
    if obj_class not in _POOLS:
        raise ValueError(f"unknown obj_class {obj_class!r}")
    perm_field = f"{obj_class}_appearance"
    perm = getattr(state, perm_field)
    appearance_idx = int(perm[type_id])
    return _POOLS[obj_class][appearance_idx]


def _build_class_canonical_names() -> dict:
    """Build the per-class canonical-name lists from the OBJECTS table.

    Each list is in vendor canonical order (the same order vendor
    objects.c emits the macro entries via OBJECT()) so ``names[type_id]``
    returns the OBJ_NAME used by ``xname`` for that local type id.

    Cite: vendor/nle/src/objects.c — POTION/SCROLL/WAND/RING/AMULET/SPELL
    macro sequences populate ``objects[]`` in this order.
    """
    from Nethax.nethax.constants.objects import OBJECTS, ObjectClass
    name_class_map = {
        "potion":    ObjectClass.POTION_CLASS,
        "scroll":    ObjectClass.SCROLL_CLASS,
        "wand":      ObjectClass.WAND_CLASS,
        "ring":      ObjectClass.RING_CLASS,
        "amulet":    ObjectClass.AMULET_CLASS,
        "spellbook": ObjectClass.SPBOOK_CLASS,
    }
    out = {}
    for cname, cval in name_class_map.items():
        out[cname] = tuple(
            obj.name for obj in OBJECTS if int(obj.class_) == int(cval)
        )
    return out


_CLASS_CANONICAL_NAMES: dict = _build_class_canonical_names()


def _xname_formal(obj_class: str, actualn: str) -> str:
    """Format the formal (fully-identified) name as vendor xname would.

    Cite: vendor/nethack/src/objnam.c::xname_flags lines 832-913 — the
    ``nn`` (oc_name_known) branch of each class switch case.

    Class templates from vendor xname:
      - POTION  : ``"potion of " + actualn``         (objnam.c:840-845)
      - SCROLL  : ``"scroll of " + actualn``         (objnam.c:858-860)
      - WAND    : ``"wand of " + actualn``           (objnam.c:874-875)
      - RING    : ``"ring of " + actualn``           (objnam.c:907-908)
      - SPBOOK  : ``"spellbook of " + actualn``      (objnam.c:895-898);
                   SPE_BOOK_OF_THE_DEAD renders as ``actualn`` alone
                   (objnam.c:896 ``if (typ != SPE_BOOK_OF_THE_DEAD)``).
      - AMULET  : ``actualn`` (objnam.c:678-679); AMULET_OF_YENDOR /
                   FAKE_AMULET_OF_YENDOR pair is rendered via the
                   per-instance ``known`` flag (objnam.c:675-677) — at
                   the type level the canonical name itself suffices.
    """
    if obj_class == "potion":
        return f"potion of {actualn}"
    if obj_class == "scroll":
        return f"scroll of {actualn}"
    if obj_class == "wand":
        return f"wand of {actualn}"
    if obj_class == "ring":
        return f"ring of {actualn}"
    if obj_class == "spellbook":
        # Vendor objnam.c:896 — SPE_BOOK_OF_THE_DEAD is the only
        # spellbook rendered without the "spellbook of " prefix.
        if actualn == "Book of the Dead":
            return actualn
        return f"spellbook of {actualn}"
    if obj_class == "amulet":
        return actualn
    raise ValueError(f"unknown obj_class {obj_class!r}")


def identified_name(obj_class: str, type_id: int) -> str:
    """Return the formal name vendor ``xname`` would emit for a fully
    identified item of this (class, type_id).

    Byte-equivalent of vendor objnam.c::xname_flags when
    ``oc_name_known = 1`` and ``dknown = 1``: the actual-name (OBJ_NAME)
    is wrapped in a class-specific template (``"potion of X"`` etc.).

    Cite: vendor/nethack/src/objnam.c::xname_flags lines 575-1028.

    Parameters
    ----------
    obj_class : one of {"potion","scroll","wand","ring","amulet","spellbook"}.
    type_id   : per-class local type id (0..N_<CLASS>_TYPES-1).

    Returns
    -------
    The canonical formal name string.

    Notes
    -----
    Some classes (SCROLL with XTRA_SCROLL_LABELs, WAND with WAN1-3, AMULET
    counting FAKE+REAL Yendor in N_*) have N_<CLASS>_TYPES entries in the
    appearance pool that exceed the OBJECTS table's named entries for that
    class.  These "extra" type_ids correspond to shuffle-only descriptions
    that vendor never assigns to real items.  We render them with a
    deterministic ``"<class>-extra-NN"`` actualn so the formal-name
    template still produces a distinct string from the appearance pool —
    matching the parity invariant that vendor ``xname`` is never identical
    to the random ``dn`` description after a shuffle.
    """
    if obj_class not in _POOLS:
        raise ValueError(f"unknown obj_class {obj_class!r}")
    names = _CLASS_CANONICAL_NAMES[obj_class]
    if type_id < 0:
        raise IndexError(
            f"{obj_class} type_id {type_id} negative"
        )
    if type_id < len(names) and names[type_id] is not None:
        actualn = names[type_id]
    else:
        # Out-of-range or None-named OBJECTS slot — synthesize a stable
        # actualn.  These slots correspond to vendor's reserved-but-unused
        # appearance entries (e.g. WAN1-3 wand descriptions) which are
        # never materialised as real items in normal play.
        actualn = f"{obj_class}-extra-{type_id:02d}"
    return _xname_formal(obj_class, actualn)


def type_for_appearance(
    state: IdentificationState,
    obj_class: str,
    appearance_idx: int,
) -> int:
    """Inverse lookup: given an appearance index, return the canonical type.

    Useful when the player identifies an item by its appearance.

    Parameters
    ----------
    state          : IdentificationState.
    obj_class      : one of the shuffled classes.
    appearance_idx : appearance pool index seen by the player.

    Returns
    -------
    Canonical type_id whose ``appearance[type_id] == appearance_idx``.
    Returns -1 if no such type exists.
    """
    if obj_class not in _POOLS:
        raise ValueError(f"unknown obj_class {obj_class!r}")
    perm_field = f"{obj_class}_appearance"
    perm = getattr(state, perm_field)
    matches = jnp.where(perm == jnp.int8(appearance_idx))[0]
    if len(matches) == 0:
        return -1
    return int(matches[0])


# ---------------------------------------------------------------------------
# Identification updates
# ---------------------------------------------------------------------------

def partial_identify(
    state: IdentificationState,
    rng: jax.Array,
    cnt: int,
) -> IdentificationState:
    """Identify *cnt* randomly-chosen currently-unidentified object types.

    Cite: vendor/nethack/src/invent.c::identify_pack (line 2711) →
    menu_identify(id_limit) (line 2660): partial identification chooses a
    bounded number of items to identify from the player's pack.  In the
    vendor's menu-driven flow the player picks; in our headless model we
    pick uniformly at random from currently-unidentified types, which is
    byte-equivalent for tests that exercise the count-of-types invariant.

    Algorithm:
      1. Build a priority vector: uniform random noise for unidentified slots,
         -inf for already-identified slots (so they are never chosen).
      2. Take the top-cnt indices via jnp.argsort (descending priority).
      3. Set those slots to True in state.identified.

    JIT-safe; no Python branching on traced values.
    """
    n = state.identified.shape[0]
    noise = jax.random.uniform(rng, shape=(n,))
    # Mask out already-identified slots so they are never picked.
    priority = jnp.where(state.identified, jnp.float32(-1.0), noise)
    # argsort ascending; we want the highest-priority (largest noise) indices.
    order = jnp.argsort(-priority)          # descending
    cnt_clipped = jnp.minimum(jnp.int32(cnt), n)
    # Build a mask: True for the first cnt_clipped positions in the sorted order.
    rank = jnp.arange(n, dtype=jnp.int32)
    chosen_mask = jnp.zeros((n,), dtype=jnp.bool_)
    chosen_mask = chosen_mask.at[order].set(rank < cnt_clipped)
    new_identified = state.identified | chosen_mask
    return state.replace(identified=new_identified)


def full_identify(
    state: IdentificationState,
    obj_type: int,
) -> IdentificationState:
    """Fully identify obj_type — future items of this type show true name.

    Cite: vendor/nethack/src/objnam.c::makeknown / learnobj.
    """
    new_identified = state.identified.at[obj_type].set(True)
    return state.replace(identified=new_identified)


def check_known(state: IdentificationState, obj_type: int) -> jnp.ndarray:
    """Return True if obj_type has been fully identified this game.

    Mirrors vendor objnam.c::xname (line 208): ``nn = ocl->oc_name_known``;
    the per-type flag drives whether an item renders as its true name or
    its random appearance.
    """
    return state.identified[obj_type]


def learn_by_use(
    state: IdentificationState,
    obj_type: jnp.ndarray,
) -> IdentificationState:
    """Mark obj_type as fully identified — vendor `makeknown(otyp)` parity.

    Vendor hack.h:1530 defines::

        #define makeknown(x) discover_object((x), TRUE, TRUE, TRUE)

    Callers (zap.c::learnwand, do_wear.c::learnring, read.c::learnscroll,
    potion.c::peffects) invoke this whenever a use-effect reveals the item
    type to the player.  Setting ``identified[obj_type] = True`` causes all
    items of this type to render with their canonical name from then on
    (objnam.c::xname line 208 ``nn = ocl->oc_name_known``).

    Functionally identical to ``full_identify`` but accepts a JIT-traced
    ``obj_type`` scalar for use inside ``lax.cond``/``lax.switch`` branches.

    Parameters
    ----------
    state    : IdentificationState.
    obj_type : int scalar (Python or traced) — index into the objects table.

    Returns
    -------
    IdentificationState with ``identified[obj_type] = True``.
    """
    t = jnp.asarray(obj_type, dtype=jnp.int32)
    t = jnp.clip(t, jnp.int32(0), jnp.int32(state.identified.shape[0] - 1))
    new_identified = state.identified.at[t].set(jnp.bool_(True))
    return state.replace(identified=new_identified)


def is_type_known(
    state: IdentificationState,
    obj_type: jnp.ndarray,
) -> jnp.ndarray:
    """Return scalar bool: is obj_type's true name known to the player?

    Mirrors vendor ``objects[otyp].oc_name_known`` read.  Safe for traced
    obj_type (clips to valid range).  Renderers / inventory display should
    use this together with the per-item ``identified`` flag — vendor xname
    uses oc_name_known as the primary gate (objnam.c:208), with per-item
    ``known``/``dknown`` modulating enchantment/BUC display only.
    """
    t = jnp.asarray(obj_type, dtype=jnp.int32)
    t = jnp.clip(t, jnp.int32(0), jnp.int32(state.identified.shape[0] - 1))
    return state.identified[t]
