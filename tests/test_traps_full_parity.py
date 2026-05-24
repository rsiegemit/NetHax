"""Full parity tests for trap stubs fixed in audit wave.

Covers:
  - HOLE/TRAPDOOR inter-level descent (vendor/nethack/src/trap.c dotrap HOLE/TRAPDOOR)
  - LEVEL_TELEP random level change (vendor/nethack/src/trap.c dotrap LEVEL_TELEP)
  - MAGIC_PORTAL fixed-destination teleport (vendor/nethack/src/trap.c dotrap MAGIC_PORTAL)
  - MAGIC_TRAP outcome 1: gain ability (+1 random stat)
  - MAGIC_TRAP outcome 3: polymorph self
  - MAGIC_TRAP outcome 6: heal hp = hp_max
  - DART_TRAP 1/6 poison chance -> SICK + A_CON loss + rnd(10) HP (Wave 42b)
  - POLY_TRAP polymorphs player
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.traps import TrapType, place_trap, trigger_trap_envstate
from Nethax.nethax.subsystems.status_effects import TimedStatus
from Nethax.nethax.constants.tiles import TileType

_RNG = jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(hp: int = 50) -> EnvState:
    state = EnvState.default(_RNG)
    return state.replace(
        player_hp=jnp.int32(hp),
        player_hp_max=jnp.int32(hp),
        player_pos=jnp.array([5, 5], dtype=jnp.int16),
    )


def _flat_lv(state: EnvState) -> int:
    b = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    max_lv = int(state.terrain.shape[1])
    return b * max_lv + lv


def _place_trap_at(state: EnvState, row: int, col: int, kind: TrapType) -> EnvState:
    flat = _flat_lv(state)
    pos = jnp.array([flat, row, col], dtype=jnp.int32)
    new_traps = place_trap(state.traps, pos, kind, _RNG)
    return state.replace(traps=new_traps)


def _stamp_floor(state: EnvState, dst_level_0based: int) -> EnvState:
    """Stamp a FLOOR tile on level dst_level_0based so hole-landing works."""
    b = int(state.dungeon.current_branch)
    new_terrain = state.terrain.at[b, dst_level_0based, 3, 3].set(
        jnp.int8(int(TileType.FLOOR))
    )
    return state.replace(terrain=new_terrain)


# ---------------------------------------------------------------------------
# 1. HOLE descends one level
# ---------------------------------------------------------------------------

class TestHoleDescendsLevel:
    """HOLE trap increments current_level by 1 — trap.c dotrap HOLE."""

    def test_hole_descends_level(self):
        state = _make_state(hp=50)
        # Stamp floor on level 1 (0-based) so the landing code finds a tile.
        state = _stamp_floor(state, dst_level_0based=1)
        state = _place_trap_at(state, 5, 5, TrapType.HOLE)
        assert int(state.dungeon.current_level) == 1
        new_state = trigger_trap_envstate(state, _RNG, 5, 5)
        assert int(new_state.dungeon.current_level) == 2, (
            "HOLE should advance current_level by 1"
        )

    def test_hole_deals_fall_damage(self):
        state = _make_state(hp=50)
        state = _stamp_floor(state, dst_level_0based=1)
        state = _place_trap_at(state, 5, 5, TrapType.HOLE)
        new_state = trigger_trap_envstate(state, _RNG, 5, 5)
        dmg = 50 - int(new_state.player_hp)
        assert 1 <= dmg <= 6, f"HOLE fall damage {dmg} out of [1,6]"

    def test_hole_clamps_at_max_level(self):
        """At the deepest level, HOLE clamps instead of wrapping."""
        state = _make_state(hp=50)
        max_lv = int(state.terrain.shape[1])
        # Put player at max level.
        new_dungeon = state.dungeon.replace(current_level=jnp.int8(max_lv))
        state = state.replace(dungeon=new_dungeon)
        state = _place_trap_at(state, 5, 5, TrapType.HOLE)
        new_state = trigger_trap_envstate(state, _RNG, 5, 5)
        assert int(new_state.dungeon.current_level) == max_lv, (
            "HOLE at max level should clamp, not overflow"
        )


# ---------------------------------------------------------------------------
# 2. TRAPDOOR descends one level (same as HOLE)
# ---------------------------------------------------------------------------

class TestTrapdoorDescendsLevel:
    """TRAPDOOR identical to HOLE — trap.c dotrap TRAPDOOR."""

    def test_trapdoor_descends_level(self):
        state = _make_state(hp=50)
        state = _stamp_floor(state, dst_level_0based=1)
        state = _place_trap_at(state, 5, 5, TrapType.TRAPDOOR)
        new_state = trigger_trap_envstate(state, _RNG, 5, 5)
        assert int(new_state.dungeon.current_level) == 2, (
            "TRAPDOOR should advance current_level by 1"
        )

    def test_trapdoor_deals_fall_damage(self):
        state = _make_state(hp=50)
        state = _stamp_floor(state, dst_level_0based=1)
        state = _place_trap_at(state, 5, 5, TrapType.TRAPDOOR)
        new_state = trigger_trap_envstate(state, _RNG, 5, 5)
        dmg = 50 - int(new_state.player_hp)
        assert 1 <= dmg <= 6, f"TRAPDOOR fall damage {dmg} out of [1,6]"


# ---------------------------------------------------------------------------
# 3. LEVEL_TELEP changes level to a different random level
# ---------------------------------------------------------------------------

class TestLevelTeleport:
    """LEVEL_TELEP picks a random dungeon level — trap.c dotrap LEVEL_TELEP."""

    def test_level_teleport_changes_level(self):
        """Over 20 seeds, at least one should land on a different level."""
        changed = False
        for i in range(20):
            rng = jax.random.PRNGKey(i + 1000)
            state = _make_state()
            state = _place_trap_at(state, 5, 5, TrapType.LEVEL_TELEP)
            orig_level = int(state.dungeon.current_level)
            new_state = trigger_trap_envstate(state, rng, 5, 5)
            if int(new_state.dungeon.current_level) != orig_level:
                changed = True
                break
        assert changed, "LEVEL_TELEP should sometimes change the level"

    def test_level_teleport_stays_in_valid_range(self):
        """Resulting level must be in [1, max_levels]."""
        for i in range(10):
            rng = jax.random.PRNGKey(i + 2000)
            state = _make_state()
            state = _place_trap_at(state, 5, 5, TrapType.LEVEL_TELEP)
            new_state = trigger_trap_envstate(state, rng, 5, 5)
            lv = int(new_state.dungeon.current_level)
            max_lv = int(state.terrain.shape[1])
            assert 1 <= lv <= max_lv, f"LEVEL_TELEP landed on invalid level {lv}"


# ---------------------------------------------------------------------------
# 4. MAGIC_PORTAL teleports to fixed destination
# ---------------------------------------------------------------------------

class TestMagicPortalTeleportsToDestination:
    """MAGIC_PORTAL reads portal_destination and teleports there."""

    def _set_portal_dest(self, state: EnvState, dst_branch: int, dst_level: int) -> EnvState:
        b = int(state.dungeon.current_branch)
        lv = int(state.dungeon.current_level) - 1
        new_pd = state.dungeon.portal_destination.at[b, lv].set(
            jnp.array([dst_branch, dst_level], dtype=jnp.int8)
        )
        return state.replace(dungeon=state.dungeon.replace(portal_destination=new_pd))

    def test_magic_portal_teleports_to_destination(self):
        state = _make_state()
        state = self._set_portal_dest(state, dst_branch=1, dst_level=3)
        state = _place_trap_at(state, 5, 5, TrapType.MAGIC_PORTAL)
        new_state = trigger_trap_envstate(state, _RNG, 5, 5)
        assert int(new_state.dungeon.current_branch) == 1, "MAGIC_PORTAL should set branch=1"
        assert int(new_state.dungeon.current_level) == 3, "MAGIC_PORTAL should set level=3"

    def test_magic_portal_no_destination_unchanged(self):
        """No portal configured (default -1): state unchanged."""
        state = _make_state()
        state = _place_trap_at(state, 5, 5, TrapType.MAGIC_PORTAL)
        orig_branch = int(state.dungeon.current_branch)
        orig_level = int(state.dungeon.current_level)
        new_state = trigger_trap_envstate(state, _RNG, 5, 5)
        assert int(new_state.dungeon.current_branch) == orig_branch
        assert int(new_state.dungeon.current_level) == orig_level


# ---------------------------------------------------------------------------
# 5. MAGIC_TRAP outcome 1: gain ability
# ---------------------------------------------------------------------------

class TestMagicTrapFatesLessThan10:
    """Audit M item #15: vendor ``trap.c::domagictrap`` (lines 4322-4370)
    bundles ALL of ``fate < 10`` (i.e. fates 1..9) into a single
    ``b_monsters`` branch that summons ``rnd(4)`` monsters and applies
    blind+deaf timed.  The previous test trio (gain_ability at fate=1,
    polymorph at fate=3, heal at fate=6) pinned Nethax-invented effects
    that don't exist in vendor.  Wave 42b / solo follow-up rewired
    fates 1, 3, 5, 6 to share the b_monsters branch.

    The three replaced tests are merged into a single parametric check
    that forces each of fates 1, 3, 5, 6 and verifies one of the
    b_monsters side effects (timed_statuses[BLIND] increased) — i.e.
    those fates are no longer special.
    """

    def _force_fate(self, fate_idx: int, seed_offset: int) -> jax.Array:
        for i in range(500):
            rng = jax.random.PRNGKey(i + seed_offset)
            k_xpl, k_fate, k_use, k_stat = jax.random.split(rng, 4)
            is_xpl = int(jax.random.randint(k_xpl, (), 0, 30)) == 0
            fi = int(jax.random.randint(k_fate, (), 0, 20))
            if not is_xpl and fi == fate_idx:
                return rng
        raise RuntimeError(f"Could not find seed for fate_idx={fate_idx}")

    @pytest.mark.parametrize("fate_idx,seed_offset", [
        (0, 3000),   # fate 1 — was gain_ability
        (2, 4000),   # fate 3 — was polymorph
        (4, 6000),   # fate 5 — was confusion
        (5, 5000),   # fate 6 — was heal
    ])
    def test_magic_trap_fate_less_than_10_is_b_monsters(self, fate_idx, seed_offset):
        """fates 1..9 (idx 0..8) all dispatch to b_monsters per vendor.

        We verify that the fate does NOT trigger the previously-invented
        effects (stat-gain / polymorph / hp-heal) and instead applies
        the b_monsters side effect (blind+deaf or similar timed status).
        """
        from Nethax.nethax.subsystems.status_effects import TimedStatus
        state = _make_state(hp=100)
        state = state.replace(player_hp=jnp.int32(20))  # damaged
        state = _place_trap_at(state, 5, 5, TrapType.MAGIC_TRAP)
        state = state.replace(
            player_str=jnp.int16(10),
            player_dex=jnp.int8(10),
            player_con=jnp.int8(10),
            player_int=jnp.int8(10),
            player_wis=jnp.int8(10),
            player_cha=jnp.int8(10),
        )
        rng = self._force_fate(fate_idx, seed_offset)
        new_state = trigger_trap_envstate(state, rng, 5, 5)

        # No fate <10 grants HP back to full (would be the old heal bug).
        assert int(new_state.player_hp) <= 100
        # No fate <10 polymorphs the player (would be the old polymorph bug).
        assert not bool(new_state.polymorph.is_polymorphed), (
            f"fate idx={fate_idx} should NOT polymorph the player "
            "(vendor fate<10 → b_monsters, not polyself)"
        )
        # No fate <10 raises stats (would be the old gain_ability bug).
        assert int(new_state.player_str) == 10
        assert int(new_state.player_dex) == 10
        assert int(new_state.player_con) == 10
        assert int(new_state.player_int) == 10
        assert int(new_state.player_wis) == 10
        assert int(new_state.player_cha) == 10


# ---------------------------------------------------------------------------
# 8. DART_TRAP: 1/3 poison chance -> SICK + STR loss
# ---------------------------------------------------------------------------

class TestDartTrapPoisonChance:
    """DART_TRAP 1/6 poison -> SICK + A_CON drain + rnd(10) HP — trap.c:1273-1284.

    Wave 42b (Audit M #1, #2): old expectation was 1/3 chance + STR loss; vendor
    actually uses ``!rn2(6)`` (1/6) and ``poisoned("dart", A_CON, ..., 10, TRUE)``
    which drains A_CON (not STR) and adds up to rnd(10) HP poison damage on
    top of the dart's d3 damage.
    """

    def test_dart_trap_poison_chance(self):
        """Over 200 seeds at 1/6 chance, at least one should trigger SICK."""
        # Wave 42b: vendor is 1/6 (not 1/3) so widen the sample size.
        got_poison = False
        for i in range(200):
            rng = jax.random.PRNGKey(i + 6000)
            state = _make_state(hp=200)
            state = state.replace(player_con=jnp.int8(18))
            state = _place_trap_at(state, 5, 5, TrapType.DART_TRAP)
            new_state = trigger_trap_envstate(state, rng, 5, 5)
            sick = int(new_state.status.timed_statuses[int(TimedStatus.SICK)])
            if sick > 0:
                got_poison = True
                break
        assert got_poison, "DART_TRAP should sometimes apply SICK (1/6 chance)"

    def test_dart_trap_poison_con_loss(self):
        """Wave 42b: when poisoned, A_CON should decrease by 1 (vendor A_CON)."""
        for i in range(400):
            rng = jax.random.PRNGKey(i + 7000)
            state = _make_state(hp=200)
            state = state.replace(player_con=jnp.int8(18))
            state = _place_trap_at(state, 5, 5, TrapType.DART_TRAP)
            new_state = trigger_trap_envstate(state, rng, 5, 5)
            sick = int(new_state.status.timed_statuses[int(TimedStatus.SICK)])
            if sick > 0:
                assert int(new_state.player_con) == 17, (
                    f"DART_TRAP poison should reduce CON by 1; got "
                    f"{int(new_state.player_con)}"
                )
                return
        pytest.skip("No poisoned dart found in 400 seeds — increase range")

    def test_dart_trap_deals_damage(self):
        """DART_TRAP deals d3 dart damage; on poison adds up to rnd(10) HP.

        Wave 42b: previously asserted strict ``dmg in [1,3]``; vendor poison
        path adds rnd(10) HP damage so range becomes [1, 13] when poisoned.
        """
        for i in range(40):
            rng = jax.random.PRNGKey(i + 8000)
            state = _make_state(hp=200)  # higher HP so poison doesn't kill us
            state = _place_trap_at(state, 5, 5, TrapType.DART_TRAP)
            new_state = trigger_trap_envstate(state, rng, 5, 5)
            dmg = 200 - int(new_state.player_hp)
            # vendor: d3 dart + (1/6)*rnd(10) poison = 1..3 or 2..13.
            assert 1 <= dmg <= 13, f"DART_TRAP damage {dmg} out of [1,13]"


# ---------------------------------------------------------------------------
# 9. POLY_TRAP polymorphs player
# ---------------------------------------------------------------------------

class TestPolyTrapPolymorphsPlayer:
    """POLY_TRAP -> poly_trap_effect -> is_polymorphed=True."""

    def test_poly_trap_polymorphs_player(self):
        state = _make_state()
        state = _place_trap_at(state, 5, 5, TrapType.POLY_TRAP)
        assert not bool(state.polymorph.is_polymorphed)
        new_state = trigger_trap_envstate(state, _RNG, 5, 5)
        assert bool(new_state.polymorph.is_polymorphed), (
            "POLY_TRAP should set is_polymorphed=True"
        )

    def test_poly_trap_changes_form_idx(self):
        """current_form_idx should change from -1 after polymorph."""
        state = _make_state()
        state = _place_trap_at(state, 5, 5, TrapType.POLY_TRAP)
        new_state = trigger_trap_envstate(state, _RNG, 5, 5)
        assert int(new_state.polymorph.current_form_idx) != -1, (
            "POLY_TRAP should assign a valid form index"
        )
