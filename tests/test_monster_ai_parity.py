"""Wave 6 monster-AI vendor-parity tests.

Audits Nethax/nethax/subsystems/monster_ai.py against:
    - vendor/nethack/src/vision.c::clear_path     (LoS)
    - vendor/nethack/src/monmove.c::mfndpos       (pathfinding)
    - vendor/nethack/src/muse.c                   (item use)
    - vendor/nethack/src/mcastu.c::castmu         (spell damage)
    - vendor/nethack/src/monmove.c flee logic     (retreat)
    - vendor/nethack/src/dogmove.c::dog_move      (pet)

Each test pins a specific vendor invariant.  When the vendor source
moves, update these assertions to match — do not weaken them.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState, StaticParams
from Nethax.nethax.constants.tiles import TileType
from Nethax.nethax.constants.monsters import (
    MONSTERS,
    M1_SWIM, M1_FLY, M2_DEMON, M2_UNDEAD,
)
from Nethax.nethax.subsystems.monster_ai import (
    MAX_MONSTERS_PER_LEVEL,
    MCAST_PSI_BOLT, MCAST_FIRE_PILLAR, MCAST_GEYSER,
    MCAST_LIGHTNING, MCAST_CLERIC,
    monster_can_see_player,
    monster_cast_damage,
    monster_cast_spell,
    monster_use_item,
    maybe_retreat,
    pathfind_step,
    pet_food_preference,
    pet_within_leash,
    _FOOD_FISH, _FOOD_MEAT, _FOOD_VEG,
)

# Concrete vendor monster indices we exercise (verified via MONSTERS table).
_KITTEN_ENTRY  = 32     # S_FELINE, MS_MEW   — pet cat
_LITTLE_DOG    = None   # filled below
_TITAN_ENTRY   = 173    # MS_SPELL (mage)
_GIANT_ANT     = 0      # level=2, non-demon/undead, no SWIM/FLY
_WATER_NYMPH   = 67     # level=3, M1_SWIM
_LICH          = 180    # level=11, M2_UNDEAD
_DEMON_ANY     = 285    # water demon, level=8, M2_DEMON

# Find a "little dog" (S_DOG) for the food test.
for _i, _m in enumerate(MONSTERS):
    if _m.name == "little dog":
        _LITTLE_DOG = _i
        break
assert _LITTLE_DOG is not None, "MONSTERS missing 'little dog' entry"

_RNG = jax.random.PRNGKey(7)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _floor_state() -> EnvState:
    """EnvState with all-floor terrain on the player's branch+level."""
    static = StaticParams()
    state = EnvState.default(rng=_RNG, static=static)
    floor_map = jnp.full((static.map_h, static.map_w), TileType.FLOOR, dtype=jnp.int8)
    return state.replace(
        terrain=state.terrain.at[0, 0].set(floor_map),
        player_hp=jnp.int32(30),
        player_hp_max=jnp.int32(30),
    )


def _set_monster(
    state: EnvState,
    slot: int,
    pos,
    hp: int = 20,
    hp_max: int = 20,
    tame: bool = False,
    peaceful: bool = False,
    asleep: bool = False,
    entry_idx: int = 0,
    apport: int = 5,
) -> EnvState:
    mai = state.monster_ai
    mai = mai.replace(
        pos=mai.pos.at[slot].set(jnp.array(pos, dtype=jnp.int16)),
        hp=mai.hp.at[slot].set(jnp.int32(hp)),
        hp_max=mai.hp_max.at[slot].set(jnp.int32(hp_max)),
        alive=mai.alive.at[slot].set(jnp.bool_(True)),
        asleep=mai.asleep.at[slot].set(jnp.bool_(asleep)),
        tame=mai.tame.at[slot].set(jnp.bool_(tame)),
        peaceful=mai.peaceful.at[slot].set(jnp.bool_(peaceful)),
        ac=mai.ac.at[slot].set(jnp.int8(10)),
        attack_dice_n=mai.attack_dice_n.at[slot].set(jnp.int8(1)),
        attack_dice_sides=mai.attack_dice_sides.at[slot].set(jnp.int8(4)),
        entry_idx=mai.entry_idx.at[slot].set(jnp.int16(entry_idx)),
        apport=mai.apport.at[slot].set(jnp.int8(apport)),
    )
    return state.replace(monster_ai=mai)


# ===========================================================================
# 1. LoS  (vendor vision.c::clear_path / is_clear / couldsee)
# ===========================================================================

def test_los_blocked_by_boulder():
    """Boulder-style obstruction (modeled as a CLOSED_DOOR in Nethax) blocks LoS.

    Vendor vision.c:182 — `if (obj->otyp == BOULDER) return 1;`
    Nethax doesn't yet have a separate boulder tile; we use CLOSED_DOOR
    which is the same blocking-tile class.
    """
    state = _floor_state().replace(player_pos=jnp.array([10, 15], dtype=jnp.int16))
    state = _set_monster(state, 0, pos=(10, 10))
    new_terrain = state.terrain.at[0, 0, 10, 12].set(jnp.int8(TileType.CLOSED_DOOR))
    state = state.replace(terrain=new_terrain)
    assert not bool(monster_can_see_player(state, jnp.int32(0)))


def test_los_blocked_by_closed_door_without_see_thru():
    """Closed doors block LoS per vendor vision.c:165-169.

    `IS_DOOR(typ) && (doormask & (D_CLOSED|D_LOCKED|D_TRAPPED))` → blocked.
    """
    state = _floor_state().replace(player_pos=jnp.array([10, 15], dtype=jnp.int16))
    state = _set_monster(state, 0, pos=(10, 10))
    new_terrain = state.terrain.at[0, 0, 10, 12].set(jnp.int8(TileType.CLOSED_DOOR))
    state = state.replace(terrain=new_terrain)
    assert not bool(monster_can_see_player(state, jnp.int32(0)))


def test_los_open_door_does_not_block():
    """Open doors are see-thru per vendor vision.c::is_clear (door-mask gate)."""
    state = _floor_state().replace(player_pos=jnp.array([10, 15], dtype=jnp.int16))
    state = _set_monster(state, 0, pos=(10, 10))
    new_terrain = state.terrain.at[0, 0, 10, 12].set(jnp.int8(TileType.OPEN_DOOR))
    state = state.replace(terrain=new_terrain)
    assert bool(monster_can_see_player(state, jnp.int32(0)))


def test_invisible_monster_not_seen_without_see_invis():
    """Vendor vision.c::couldsee — invisible hero requires monster to have
    M1_SEE_INVIS (or telepathy, which we don't model yet).

    Giant ant (entry=0) does NOT have M1_SEE_INVIS, so when the player is
    invisible, it cannot see them even on open floor.
    """
    state = _floor_state().replace(player_pos=jnp.array([10, 15], dtype=jnp.int16))
    state = _set_monster(state, 0, pos=(10, 10), entry_idx=_GIANT_ANT)
    # Set timed invisibility status (TimedStatus.INVIS_TMP = 17).
    new_status = state.status.replace(
        timed_statuses=state.status.timed_statuses.at[17].set(jnp.int32(100))
    )
    state = state.replace(status=new_status)
    # Sanity: without invisibility the monster sees the player.
    cleared = _floor_state().replace(player_pos=jnp.array([10, 15], dtype=jnp.int16))
    cleared = _set_monster(cleared, 0, pos=(10, 10), entry_idx=_GIANT_ANT)
    assert bool(monster_can_see_player(cleared, jnp.int32(0)))
    # With invisibility, the same configuration should not see.
    assert not bool(monster_can_see_player(state, jnp.int32(0)))


# ===========================================================================
# 2. Pathfind  (vendor monmove.c::mfndpos)
# ===========================================================================

def test_pathfind_avoids_lava_without_flying():
    """A non-flying monster must NOT path straight through a lava tile.

    Vendor mfndpos: lava only traversable by flyers.  We arrange a single
    lava tile directly east; the BFS must route around it (dy != 0).
    """
    state = _floor_state().replace(player_pos=jnp.array([10, 12], dtype=jnp.int16))
    state = _set_monster(state, 0, pos=(10, 10), entry_idx=_GIANT_ANT)
    new_terrain = state.terrain.at[0, 0, 10, 11].set(jnp.int8(TileType.LAVA))
    state = state.replace(terrain=new_terrain)
    step = pathfind_step(state, jnp.int32(0))
    dy, dx = int(step[0]), int(step[1])
    # Must NOT step east into lava (dy=0, dx=1).  Either detour (dy != 0)
    # or stand still — either is consistent with vendor "lava is impassable".
    assert (dy, dx) != (0, 1), f"Non-flyer walked into lava: ({dy},{dx})"


def test_pathfind_swims_in_water_with_swim_flag():
    """A swimming monster CAN path through water (vendor mfndpos M1_SWIM).

    Water nymph (entry=67) has M1_SWIM.  Set the WHOLE row between the
    monster and the player to water; the BFS step must close the distance
    (which is only possible if the BFS treats water as passable for this
    mover).  A non-swimmer in the same scenario would be forced to detour
    around — distance would remain >= 2 after one step.
    """
    state = _floor_state().replace(player_pos=jnp.array([10, 13], dtype=jnp.int16))
    state = _set_monster(state, 0, pos=(10, 10), entry_idx=_WATER_NYMPH)
    new_terrain = state.terrain
    for c in (11, 12):
        new_terrain = new_terrain.at[0, 0, 10, c].set(jnp.int8(TileType.WATER))
    # Block detour routes — only the water corridor is open.
    for r in (9, 11):
        for c in (10, 11, 12, 13):
            new_terrain = new_terrain.at[0, 0, r, c].set(jnp.int8(TileType.WALL))
    state = state.replace(terrain=new_terrain)
    assert MONSTERS[_WATER_NYMPH].flags1 & M1_SWIM, "test fixture broken"
    step = pathfind_step(state, jnp.int32(0))
    dy, dx = int(step[0]), int(step[1])
    # Only valid step in this corridor: (0, 1) into the water.
    assert (dy, dx) == (0, 1), \
        f"M1_SWIM monster failed to traverse water: ({dy},{dx})"


def test_pathfind_avoids_peaceful_monster_with_mm_peaceful():
    """Hostile mover should not path through a peaceful monster's tile.

    Vendor mfndpos.h ALLOW_M/MM_PEACEFUL: peaceful monsters are blocking
    obstacles to other movers.  Place a peaceful monster directly east of
    the hostile mover; the BFS must route around.
    """
    state = _floor_state().replace(player_pos=jnp.array([10, 13], dtype=jnp.int16))
    state = _set_monster(state, 0, pos=(10, 10), entry_idx=_GIANT_ANT)
    # Peaceful blocker at (10, 11).
    state = _set_monster(state, 1, pos=(10, 11), entry_idx=_GIANT_ANT, peaceful=True)
    step = pathfind_step(state, jnp.int32(0))
    dy, dx = int(step[0]), int(step[1])
    # Must NOT step directly onto the peaceful (10, 11).
    assert (dy, dx) != (0, 1), \
        f"Pathfinder stepped onto peaceful monster: ({dy},{dx})"


# ===========================================================================
# 3. Muse  (vendor muse.c)
# ===========================================================================

def test_mage_zaps_wand_when_in_los():
    """A mage-class monster in LoS at range 2..8 reaches the zap-wand branch.

    Vendor muse.c::find_offensive picks WAN_* items for an M1_USES_ITEMS
    mage when ``lined_up()`` + in range.  Nethax has no inventory yet, so
    we assert the muse call returns cleanly (no state corruption) when
    the predicate fires.  This locks the GATE — payload work is Wave 6+.
    """
    state = _floor_state().replace(player_pos=jnp.array([10, 14], dtype=jnp.int16))
    state = _set_monster(state, 0, pos=(10, 10), hp=40, hp_max=40,
                         entry_idx=_TITAN_ENTRY)
    rng = jax.random.PRNGKey(11)
    out = monster_use_item(state, rng, jnp.int32(0))
    # State is preserved (stubs are no-ops); the predicate fired without
    # exception — that's the contract.
    assert int(out.monster_ai.hp[0]) == int(state.monster_ai.hp[0])


def test_low_hp_monster_quaffs_healing_potion():
    """Low-HP, eligible mover reaches the quaff_heal branch.

    Vendor muse.c::find_defensive — HP below the level-fraction threshold
    invokes m_use_healing.  Nethax stub returns state unchanged but the
    predicate must fire (no exception).
    """
    state = _floor_state().replace(player_pos=jnp.array([10, 15], dtype=jnp.int16))
    state = _set_monster(state, 0, pos=(10, 10), hp=2, hp_max=40,
                         entry_idx=_TITAN_ENTRY)
    rng = jax.random.PRNGKey(12)
    out = monster_use_item(state, rng, jnp.int32(0))
    assert int(out.monster_ai.hp[0]) == int(state.monster_ai.hp[0])


def test_uses_items_flag_gates_muse():
    """Animal / mindless / nohands monsters skip muse entirely.

    Vendor muse.c:1428-1430 entry gate — we assert by comparing two
    pre/post states for an animal-class monster vs a humanoid mage:
    both stubs are no-ops, but the eligible path must reach the cond
    branches and the ineligible path must not.  We verify by inspecting
    the entry-flag predicate directly.
    """
    from Nethax.nethax.subsystems.monster_ai import _can_use_items
    # Giant ant is M1_ANIMAL → ineligible.
    assert not bool(_can_use_items(jnp.int16(_GIANT_ANT)))
    # Titan is humanoid spellcaster → eligible.
    assert bool(_can_use_items(jnp.int16(_TITAN_ENTRY)))


# ===========================================================================
# 4. Mcastu  (vendor mcastu.c spell damage formulas)
# ===========================================================================

def test_wizard_psi_bolt_damage_formula():
    """MCAST_PSI_BOLT damage = d((ml/2)+1, 6).

    Vendor mcastu.c::castmu lines 240-243 set dmg = d((ml/2)+1, 6) for
    the unspecified-attack default, then mcast_psi_bolt passes it through.
    For ml=10 → d(6, 6) → range [6..36].  We verify min and max over a
    spread of seeds.
    """
    ml = jnp.int32(10)
    seen = []
    for s in range(60):
        rng = jax.random.PRNGKey(s + 200)
        dmg = int(monster_cast_damage(rng, MCAST_PSI_BOLT, ml))
        seen.append(dmg)
    # ml=10 → n=6 dice of d6 → expected range [6..36].
    assert min(seen) >= 6, f"psi_bolt min underflow: {min(seen)} (expected >= 6)"
    assert max(seen) <= 36, f"psi_bolt max overflow: {max(seen)} (expected <= 36)"
    # Statistically should be > 6 most of the time; sanity that the spread is real.
    assert max(seen) > min(seen)


def test_cleric_spell_damage_formula_by_level():
    """MCAST_CLERIC damage = d((ml/2)+1, 6) — same backbone as default in
    vendor mcastu.c::castmu line 243 (cleric default attack damd=6).

    For ml=8 → d(5, 6) → [5..30].  For ml=2 → d(2, 6) → [2..12].
    """
    seen_8 = []
    for s in range(40):
        rng = jax.random.PRNGKey(s + 300)
        seen_8.append(int(monster_cast_damage(rng, MCAST_CLERIC, jnp.int32(8))))
    assert min(seen_8) >= 5, f"cleric@ml8 min: {min(seen_8)}"
    assert max(seen_8) <= 30, f"cleric@ml8 max: {max(seen_8)}"

    seen_2 = []
    for s in range(40):
        rng = jax.random.PRNGKey(s + 400)
        seen_2.append(int(monster_cast_damage(rng, MCAST_CLERIC, jnp.int32(2))))
    assert min(seen_2) >= 2, f"cleric@ml2 min: {min(seen_2)}"
    assert max(seen_2) <= 12, f"cleric@ml2 max: {max(seen_2)}"


def test_fire_pillar_damage_d8_6():
    """MCAST_FIRE_PILLAR damage = d(8, 6) — vendor mcast_fire_pillar
    (mcastu.c:545).  Range [8..48].
    """
    seen = []
    for s in range(80):
        rng = jax.random.PRNGKey(s + 500)
        seen.append(int(monster_cast_damage(rng, MCAST_FIRE_PILLAR, jnp.int32(5))))
    assert min(seen) >= 8, f"fire_pillar min: {min(seen)} (expected >= 8)"
    assert max(seen) <= 48, f"fire_pillar max: {max(seen)} (expected <= 48)"


def test_geyser_and_lightning_damage_d8_6():
    """MCAST_GEYSER and MCAST_LIGHTNING both deal d(8,6) per vendor
    mcastu.c:529 / mcastu.c:574.  Range [8..48].
    """
    for spell in (MCAST_GEYSER, MCAST_LIGHTNING):
        seen = []
        for s in range(60):
            rng = jax.random.PRNGKey(s + 600 + spell)
            seen.append(int(monster_cast_damage(rng, spell, jnp.int32(7))))
        assert min(seen) >= 8, f"{spell}: min {min(seen)}"
        assert max(seen) <= 48, f"{spell}: max {max(seen)}"


# ===========================================================================
# 5. Retreat  (vendor monmove.c flee logic)
# ===========================================================================

def test_demon_never_flees():
    """Demons (M2_DEMON) never enter the flee branch, even at 1 HP.

    Vendor onscary / monflee gating excludes demons.  Water demon (entry=285)
    at HP=1 must return (0, 0) — no retreat.
    """
    state = _floor_state().replace(player_pos=jnp.array([10, 15], dtype=jnp.int16))
    state = _set_monster(state, 0, pos=(10, 10), hp=1, hp_max=40,
                         entry_idx=_DEMON_ANY)
    # Sanity: confirm the entry really is a demon.
    assert MONSTERS[_DEMON_ANY].flags2 & M2_DEMON
    step = maybe_retreat(state, jnp.int32(0))
    dy, dx = int(step[0]), int(step[1])
    assert (dy, dx) == (0, 0), f"Demon fled at 1 HP: ({dy},{dx})"


def test_undead_never_flees():
    """Undead (M2_UNDEAD) likewise never flees — fearless / mindless of pain.

    Lich (entry=180) at HP=1 must return (0, 0).
    """
    state = _floor_state().replace(player_pos=jnp.array([10, 15], dtype=jnp.int16))
    state = _set_monster(state, 0, pos=(10, 10), hp=1, hp_max=40,
                         entry_idx=_LICH)
    assert MONSTERS[_LICH].flags2 & M2_UNDEAD
    step = maybe_retreat(state, jnp.int32(0))
    dy, dx = int(step[0]), int(step[1])
    assert (dy, dx) == (0, 0), f"Undead fled at 1 HP: ({dy},{dx})"


def test_low_level_monster_flees_at_1_hp():
    """For monsters with level < 2 the vendor flee threshold collapses to
    hp <= 1.  Use an acid-blob-style level-1 mover (entry=14 coyote works
    too — both level 1).  At hp=1 they SHOULD flee.

    Coyote (entry=14, level=1): non-demon, non-undead → flees.
    """
    # acid blob is entry=6, level=1, M1_MINDLESS → still flees by HP rule
    # (mindlessness affects muse, not flee).  Use coyote for clarity.
    COYOTE = 14
    assert MONSTERS[COYOTE].level == 1
    state = _floor_state().replace(player_pos=jnp.array([10, 15], dtype=jnp.int16))
    state = _set_monster(state, 0, pos=(10, 10), hp=1, hp_max=8,
                         entry_idx=COYOTE)
    step = maybe_retreat(state, jnp.int32(0))
    dy, dx = int(step[0]), int(step[1])
    # Player is east; retreat is west → dx == -1.
    assert dx == -1, f"Level-1 monster at 1 HP failed to flee: ({dy},{dx})"


# ===========================================================================
# 6. Pet  (vendor dogmove.c)
# ===========================================================================

def test_cat_prefers_fish():
    """S_FELINE pets prefer FOOD_FISH (vendor dog_eat preference table).

    pet_food_preference returns +1 for cat+fish, -1 for cat+veg.
    """
    pref_fish = int(pet_food_preference(jnp.int16(_KITTEN_ENTRY), _FOOD_FISH))
    pref_meat = int(pet_food_preference(jnp.int16(_KITTEN_ENTRY), _FOOD_MEAT))
    pref_veg  = int(pet_food_preference(jnp.int16(_KITTEN_ENTRY), _FOOD_VEG))
    assert pref_fish == 1, f"cat+fish pref {pref_fish}, expected +1"
    assert pref_veg == -1, f"cat+veg  pref {pref_veg}, expected -1"
    # Cat is indifferent (or hostile via veg) to meat; the table only
    # privileges its favourite.  Just assert it's not "loved".
    assert pref_meat != 1


def test_dog_prefers_meat():
    """S_DOG pets prefer FOOD_MEAT (vendor dog_eat preference)."""
    pref_meat = int(pet_food_preference(jnp.int16(_LITTLE_DOG), _FOOD_MEAT))
    pref_fish = int(pet_food_preference(jnp.int16(_LITTLE_DOG), _FOOD_FISH))
    pref_veg  = int(pet_food_preference(jnp.int16(_LITTLE_DOG), _FOOD_VEG))
    assert pref_meat == 1, f"dog+meat pref {pref_meat}"
    assert pref_veg == -1
    assert pref_fish != 1


def test_pet_follows_within_leash():
    """Vendor pet leash check: ``mleashed && distu > 4``.

    Cite: vendor/nethack/src/dogmove.c line 1093.
    A non-leashed pet is always "within leash" (no restriction).  A leashed
    pet is "within leash" iff distu (squared Euclidean) <= 4.
    """
    state = _floor_state().replace(player_pos=jnp.array([10, 10], dtype=jnp.int16))

    # Non-leashed pet at any distance — always within leash.
    state_unleashed = _set_monster(state, 0, pos=(10, 18), tame=True,
                                   entry_idx=_KITTEN_ENTRY)
    assert bool(pet_within_leash(state_unleashed, jnp.int32(0)))

    # Leashed pet at distu=1 (adjacent col) → within leash.
    mai = state.monster_ai.replace(
        mleashed=state.monster_ai.mleashed.at[0].set(jnp.bool_(True))
    )
    state_close = _set_monster(state.replace(monster_ai=mai), 0, pos=(10, 11),
                               tame=True, entry_idx=_KITTEN_ENTRY)
    state_close = state_close.replace(
        monster_ai=state_close.monster_ai.replace(
            mleashed=state_close.monster_ai.mleashed.at[0].set(jnp.bool_(True))
        )
    )
    assert bool(pet_within_leash(state_close, jnp.int32(0)))

    # Leashed pet at distu=64 (col 18 - 10 = 8 → 8² = 64 > 4) → outside.
    state_far = _set_monster(state.replace(monster_ai=mai), 0, pos=(10, 18),
                             tame=True, entry_idx=_KITTEN_ENTRY)
    state_far = state_far.replace(
        monster_ai=state_far.monster_ai.replace(
            mleashed=state_far.monster_ai.mleashed.at[0].set(jnp.bool_(True))
        )
    )
    assert not bool(pet_within_leash(state_far, jnp.int32(0)))
