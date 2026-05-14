"""Tests for Wave 3 magic subsystem: casting, Pw regen, spellbook reading.

Canonical coverage:
  - cast healing → HP increases
  - cast magic_missile with adjacent monster → monster HP decreases
  - cast with player_pw < cost → state unchanged
  - read spellbook → spell_known flips True
  - Pw regen ticks correctly (threshold ticks → +1 Pw)
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState, StaticParams
from Nethax.nethax.subsystems.magic import (
    MagicState,
    MAX_SPELL_MEMORY,
    N_SPELLS,
    SpellId,
    cast_spell,
    handle_cast,
    pw_regen_tick,
    _SPELL_LEVELS,
)
from Nethax.nethax.subsystems.items_spellbooks import read_spellbook
from Nethax.nethax.subsystems.inventory import make_item, _items_from_list
from Nethax.nethax.subsystems.monster_ai import make_monster_ai_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_state(player_pw: int = 50, player_pw_max: int = 50,
                player_hp: int = 8, player_hp_max: int = 20,
                player_int: int = 16, player_wis: int = 10,
                player_xl: int = 5, player_role: int = 12) -> EnvState:
    """Return a default EnvState with convenient overrides.

    player_role=12 → WIZARD (good spellcaster, INT-based).
    """
    rng = jax.random.PRNGKey(0)
    state = EnvState.default(rng)
    return state.replace(
        player_pw=jnp.int32(player_pw),
        player_pw_max=jnp.int32(player_pw_max),
        player_hp=jnp.int32(player_hp),
        player_hp_max=jnp.int32(player_hp_max),
        player_int=jnp.int8(player_int),
        player_wis=jnp.int8(player_wis),
        player_xl=jnp.int32(player_xl),
        player_role=jnp.int8(player_role),
    )


def _state_with_known_spell(spell_id: int, pw: int = 50) -> EnvState:
    """Return a state where spell_id is known and memorized."""
    state = _base_state(player_pw=pw)
    magic = state.magic
    new_known = magic.spell_known.at[spell_id].set(True)
    new_mem   = magic.spell_memory.at[spell_id].set(jnp.int32(MAX_SPELL_MEMORY))
    state = state.replace(magic=magic.replace(spell_known=new_known, spell_memory=new_mem))
    return state


def _state_with_monster_hp(state: EnvState, monster_hp: int) -> EnvState:
    """Set the HP of monster slot 0 to monster_hp and mark it alive."""
    mai = state.monster_ai
    new_hp    = mai.hp.at[0].set(jnp.int32(monster_hp))
    new_alive = mai.alive.at[0].set(True)
    return state.replace(monster_ai=mai.replace(hp=new_hp, alive=new_alive))


# ---------------------------------------------------------------------------
# cast_spell: HEALING → HP increases
# ---------------------------------------------------------------------------

class TestCastHealing:
    def test_healing_increases_hp(self):
        """Cast HEALING with full Pw → player_hp increases (not exceeding hp_max).

        Wave 6 #73: updated to vendor-correct success rate per
        vendor/nethack/src/spell.c::percent_success.  Wizard INT=16 XL=5 only
        has ~16% chance to land HEALING; we bump INT=18 XL=30 so success% is
        capped at 100 and the cast is deterministic.
        """
        # Wave 6 #73: updated to vendor-correct value per vendor/nethack/src/spell.c::percent_success
        state = _state_with_known_spell(SpellId.HEALING, pw=50)
        # Boost stats so success% pins to 100 under vendor formula.
        state = state.replace(
            player_int=jnp.int8(18),
            player_xl=jnp.int32(30),
        )
        rng = jax.random.PRNGKey(1)
        hp_before = int(state.player_hp)

        new_state, success = cast_spell(state, rng, SpellId.HEALING)

        assert success, "Healing should succeed at vendor 100% success rate"
        new_hp = int(new_state.player_hp)
        assert new_hp > hp_before, f"HP should increase: {hp_before} → {new_hp}"
        assert new_hp <= int(new_state.player_hp_max), "HP must not exceed hp_max"

    def test_healing_decrements_pw(self):
        """Casting healing costs Pw = level * 5 = 5."""
        state = _state_with_known_spell(SpellId.HEALING, pw=50)
        rng = jax.random.PRNGKey(2)
        pw_before = int(state.player_pw)

        new_state, _ = cast_spell(state, rng, SpellId.HEALING)

        pw_after = int(new_state.player_pw)
        expected_cost = int(_SPELL_LEVELS[SpellId.HEALING]) * 5
        assert pw_after == pw_before - expected_cost, (
            f"Pw should decrease by {expected_cost}: {pw_before} → {pw_after}"
        )

    def test_healing_decrements_spell_memory(self):
        """Casting decrements spell_memory by 1."""
        state = _state_with_known_spell(SpellId.HEALING, pw=50)
        rng = jax.random.PRNGKey(3)
        mem_before = int(state.magic.spell_memory[SpellId.HEALING])

        new_state, _ = cast_spell(state, rng, SpellId.HEALING)

        mem_after = int(new_state.magic.spell_memory[SpellId.HEALING])
        assert mem_after == mem_before - 1, (
            f"spell_memory should decrease by 1: {mem_before} → {mem_after}"
        )


# ---------------------------------------------------------------------------
# cast_spell: MAGIC_MISSILE with adjacent monster → monster damaged
# ---------------------------------------------------------------------------

class TestCastMagicMissile:
    def test_magic_missile_damages_monster(self):
        """Cast MAGIC_MISSILE → monster slot 0 HP decreases.

        Wave 6 #73: updated to vendor-correct value per
        vendor/nethack/src/spell.c::percent_success.  Wizard INT=16 XL=5
        only has 0% success on lv-2 MAGIC_MISSILE; boost to INT=18 XL=30
        for 100% success.
        """
        # Wave 6 #73: updated to vendor-correct value per vendor/nethack/src/spell.c::percent_success
        state = _state_with_known_spell(SpellId.MAGIC_MISSILE, pw=50)
        state = state.replace(
            player_int=jnp.int8(18),
            player_xl=jnp.int32(30),
        )
        state = _state_with_monster_hp(state, monster_hp=30)
        rng = jax.random.PRNGKey(10)
        hp_before = int(state.monster_ai.hp[0])

        new_state, success = cast_spell(state, rng, SpellId.MAGIC_MISSILE)

        assert success, "Magic missile should succeed at vendor 100% rate"
        hp_after = int(new_state.monster_ai.hp[0])
        assert hp_after < hp_before, (
            f"Monster HP should decrease: {hp_before} → {hp_after}"
        )

    def test_magic_missile_hp_floor_zero(self):
        """Cast MAGIC_MISSILE on a low-HP monster → HP floored at 0, not negative.

        Wave 6 #73: updated to vendor-correct value per
        vendor/nethack/src/spell.c::percent_success.
        """
        # Wave 6 #73: updated to vendor-correct value per vendor/nethack/src/spell.c::percent_success
        state = _state_with_known_spell(SpellId.MAGIC_MISSILE, pw=50)
        state = state.replace(
            player_int=jnp.int8(18),
            player_xl=jnp.int32(30),
        )
        state = _state_with_monster_hp(state, monster_hp=1)
        rng = jax.random.PRNGKey(11)

        new_state, success = cast_spell(state, rng, SpellId.MAGIC_MISSILE)

        if success:
            hp_after = int(new_state.monster_ai.hp[0])
            assert hp_after >= 0, "Monster HP must not go below 0"


# ---------------------------------------------------------------------------
# cast_spell: insufficient Pw → state unchanged
# ---------------------------------------------------------------------------

class TestCastInsufficientPw:
    def test_cast_fails_when_pw_too_low(self):
        """Cast with player_pw < cost → returns original state, not success."""
        # HEALING costs 5 Pw (level 1 * 5); set pw=4
        state = _state_with_known_spell(SpellId.HEALING, pw=4)
        rng = jax.random.PRNGKey(20)
        pw_before = int(state.player_pw)
        hp_before = int(state.player_hp)

        new_state, success = cast_spell(state, rng, SpellId.HEALING)

        assert not success, "Cast should fail with insufficient Pw"
        assert int(new_state.player_pw) == pw_before, "Pw should be unchanged on Pw failure"
        assert int(new_state.player_hp) == hp_before, "HP should be unchanged on Pw failure"

    def test_high_level_spell_requires_more_pw(self):
        """FINGER_OF_DEATH costs 35 Pw (level 7 * 5); pw=34 should fail."""
        state = _state_with_known_spell(SpellId.FINGER_OF_DEATH, pw=34)
        rng = jax.random.PRNGKey(21)

        new_state, success = cast_spell(state, rng, SpellId.FINGER_OF_DEATH)

        assert not success
        assert int(new_state.player_pw) == 34


# ---------------------------------------------------------------------------
# read_spellbook → spell_known flips True
# ---------------------------------------------------------------------------

class TestReadSpellbook:
    def _state_with_spellbook(self, spell_id: int, slot: int = 0,
                               player_int: int = 18) -> EnvState:
        """Return state with a spellbook in inventory slot `slot`."""
        from Nethax.nethax.subsystems.inventory import (
            InventoryState, _items_from_list, make_item, make_empty_item
        )

        SPBOOK_CLASS = 6  # objects.h SPBOOK_CLASS
        book = make_item(category=SPBOOK_CLASS, type_id=spell_id, quantity=1)

        # Build items list: spellbook at `slot`, rest empty
        items_list = [make_empty_item() for _ in range(52)]
        items_list[slot] = book

        from Nethax.nethax.subsystems.inventory import _stack_items
        new_items = _stack_items(items_list)
        state = _base_state(player_int=player_int)
        new_inv = state.inventory.replace(items=new_items)
        return state.replace(inventory=new_inv)

    def test_read_spellbook_sets_spell_known(self):
        """Reading a HEALING spellbook with high INT → spell_known[HEALING] = True."""
        # Use INT=18 and XL=5; level-1 book → modifier +3, INT bonus +4 → d20+7 >= 10
        # Minimum d20=1 → total=8, which can still fail. Use INT=20 equivalent to guarantee.
        state = self._state_with_spellbook(SpellId.HEALING, player_int=20)
        rng = jax.random.PRNGKey(30)

        # Try multiple RNG seeds to find one that succeeds (d20 >= 3 with bonuses)
        for seed in range(100):
            rng_try = jax.random.PRNGKey(seed)
            new_state = read_spellbook(state, rng_try, slot_idx=0)
            if bool(new_state.magic.spell_known[SpellId.HEALING]):
                break

        assert bool(new_state.magic.spell_known[SpellId.HEALING]), (
            "spell_known[HEALING] should be True after reading spellbook"
        )

    def test_read_spellbook_sets_max_memory(self):
        """On successful read, spell_memory is set to MAX_SPELL_MEMORY."""
        state = self._state_with_spellbook(SpellId.HEALING, player_int=20)

        for seed in range(100):
            rng_try = jax.random.PRNGKey(seed)
            new_state = read_spellbook(state, rng_try, slot_idx=0)
            if bool(new_state.magic.spell_known[SpellId.HEALING]):
                break

        mem = int(new_state.magic.spell_memory[SpellId.HEALING])
        assert mem == MAX_SPELL_MEMORY, f"spell_memory should be {MAX_SPELL_MEMORY}, got {mem}"

    def test_read_spellbook_assigns_letter(self):
        """On successful read, spell_letter[spell_id] is set (>= 0)."""
        state = self._state_with_spellbook(SpellId.HEALING, player_int=20)

        for seed in range(100):
            rng_try = jax.random.PRNGKey(seed)
            new_state = read_spellbook(state, rng_try, slot_idx=0)
            if bool(new_state.magic.spell_known[SpellId.HEALING]):
                break

        letter = int(new_state.magic.spell_letter[SpellId.HEALING])
        assert letter >= 0, f"spell_letter should be assigned (>=0), got {letter}"

    def test_read_unknown_slot_noop(self):
        """Reading a slot with spell_id=-1 (blank) does not change state."""
        state = _base_state()
        rng = jax.random.PRNGKey(40)

        # Slot 0 has type_id=0 → that's SpellId.DIG (valid), not blank.
        # Use a slot with empty item (category=0, type_id=0 = DIG).
        # For blank test: items_spellbooks treats any spell_id >= N_SPELLS as blank.
        from Nethax.nethax.subsystems.items_spellbooks import BLANK_SPELL_ID
        from Nethax.nethax.subsystems.inventory import make_empty_item, _stack_items

        items_list = [make_empty_item() for _ in range(52)]
        new_items = _stack_items(items_list)
        state2 = state.replace(inventory=state.inventory.replace(items=new_items))

        # type_id=0 on an empty slot → SpellId.DIG; this is a valid spell, so
        # we instead test that a blank spellbook id (N_SPELLS) is a no-op.
        # Manually craft a slot with type_id = N_SPELLS (out of range).
        from Nethax.nethax.subsystems.inventory import make_item
        SPBOOK_CLASS = 6
        blank_book = make_item(category=SPBOOK_CLASS, type_id=N_SPELLS, quantity=1)
        items_list[5] = blank_book
        new_items2 = _stack_items(items_list)
        state3 = state.replace(inventory=state.inventory.replace(items=new_items2))

        new_state = read_spellbook(state3, rng, slot_idx=5)
        # No spell should have been learned
        assert not jnp.any(new_state.magic.spell_known), "No spell should be known after blank read"


# ---------------------------------------------------------------------------
# Pw regeneration ticks
# ---------------------------------------------------------------------------

class TestPwRegen:
    def test_pw_regen_eventually_restores_pw(self):
        """Vendor formula (allmain.c::regen_pw): period = (MAXULEV+8-XL)*(wizard?3:4)/6,
        regen fires when moves % period == 0. Test by ticking enough turns
        with increasing timestep to see Pw rise.
        Wave 6 parity-fix: vendor/nethack/src/allmain.c:606-625.
        """
        state = _base_state(player_pw=0, player_pw_max=10, player_xl=1, player_role=12)
        # Wizard XL=1: period = (30+8-1)*3/6 = 18
        # Tick 200 turns with monotonically increasing timestep — Pw must rise > 0.
        current = state
        for t in range(200):
            current = current.replace(timestep=jnp.int32(t))
            current = pw_regen_tick(current, jax.random.PRNGKey(t))
        assert int(current.player_pw) > 0, "Pw should rise after 200 ticks"

    def test_pw_regen_xl1_wizard_fires_within_period(self):
        """Wave 6 parity-fix per vendor/nethack/src/allmain.c:610-625.
        Period for Wizard XL=1 is (30+8-1)*3/6 = 18.
        Across 18*20=360 ticks, regen should fire many times.
        """
        state = _base_state(player_pw=0, player_pw_max=100, player_xl=1, player_role=12)
        current = state
        for t in range(360):
            current = current.replace(timestep=jnp.int32(t))
            current = pw_regen_tick(current, jax.random.PRNGKey(t))
        assert int(current.player_pw) >= 10, (
            f"After 360 ticks at period=18, expect Pw >= 10, got {int(current.player_pw)}"
        )

    def test_pw_regen_does_not_exceed_max(self):
        """pw_regen_tick never increases Pw above player_pw_max."""
        state = _base_state(player_pw=10, player_pw_max=10, player_xl=1, player_role=12)

        for _ in range(100):
            state = pw_regen_tick(state)

        assert int(state.player_pw) == 10, "Pw should not exceed pw_max"

    def test_pw_regen_noop_when_full(self):
        """No counter accumulation needed when already at max Pw."""
        state = _base_state(player_pw=50, player_pw_max=50, player_xl=5, player_role=12)
        pw_before = int(state.player_pw)

        for _ in range(50):
            state = pw_regen_tick(state)

        assert int(state.player_pw) == pw_before

    def test_pw_regen_period_based_gating(self):
        """Vendor formula uses moves % period == 0; regen does NOT fire every turn.
        For Valkyrie XL=20: period = (30+8-20)*4/6 = 12. Across 11 ticks Pw stays 0
        because no timestep aligns with period boundary if we keep timestep at 0.
        Wave 6 parity-fix: vendor/nethack/src/allmain.c:610-625.
        """
        state = _base_state(player_pw=0, player_pw_max=20, player_xl=20, player_role=11)
        # Tick 11 turns all with timestep=1 (not aligned with period=12) → no regen.
        for _ in range(11):
            state = state.replace(timestep=jnp.int32(1))
            state = pw_regen_tick(state, jax.random.PRNGKey(0))
        assert int(state.player_pw) == 0, "No regen when moves % period != 0"


# ---------------------------------------------------------------------------
# handle_cast
# ---------------------------------------------------------------------------

class TestHandleCast:
    def test_handle_cast_no_known_spells_noop(self):
        """handle_cast with no known spells returns state unchanged and sid=-1."""
        state = _base_state()
        rng = jax.random.PRNGKey(50)

        new_state, sid = handle_cast(state, rng)
        assert sid == -1
        assert int(new_state.player_pw) == int(state.player_pw)

    def test_handle_cast_picks_first_known_spell(self):
        """handle_cast selects the first known+memorized spell and casts it."""
        # Know only HEALING
        state = _state_with_known_spell(SpellId.HEALING, pw=50)
        rng = jax.random.PRNGKey(51)

        new_state, sid = handle_cast(state, rng)
        assert sid == SpellId.HEALING
        # Pw should have been decremented
        assert int(new_state.player_pw) < int(state.player_pw)
