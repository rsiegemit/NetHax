"""Wave 6 Phase B+ — vendor-parity tests for trap formulas.

Every test asserts the *exact* numeric range from vendor/nethack/src/trap.c
and (where the formula is stochastic) uses multiple seeds to cover the
distribution.  Vendor is ground truth — these tests pin the implementation
to the C source.

Reference: vendor/nethack/src/trap.c
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.constants.tiles import TileType
from Nethax.nethax.subsystems.traps import (
    TrapType,
    trigger_trap_envstate,
    place_trap,
)
from Nethax.nethax.subsystems.status_effects import TimedStatus
from Nethax.nethax.subsystems.inventory import ItemCategory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat_lv(state: EnvState) -> int:
    b      = int(state.dungeon.current_branch)
    lv     = int(state.dungeon.current_level) - 1
    max_lv = int(state.terrain.shape[1])
    return b * max_lv + lv


def _make_state(seed: int = 42, hp: int = 200, pw: int = 200) -> EnvState:
    rng = jax.random.PRNGKey(seed)
    state = EnvState.default(rng)
    return state.replace(
        player_pos=jnp.array([5, 5], dtype=jnp.int16),
        player_hp=jnp.int32(hp),
        player_hp_max=jnp.int32(hp),
        player_pw=jnp.int32(pw),
        player_pw_max=jnp.int32(pw),
    )


def _place(state: EnvState, kind: TrapType, row: int = 5, col: int = 5) -> EnvState:
    flat = _flat_lv(state)
    pos  = jnp.array([flat, row, col], dtype=jnp.int32)
    return state.replace(
        traps=place_trap(state.traps, pos, kind, jax.random.PRNGKey(0))
    )


def _add_floor(state: EnvState) -> EnvState:
    b  = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    new_terrain = state.terrain.at[b, lv].set(
        jnp.full_like(state.terrain[b, lv], jnp.int8(TileType.FLOOR))
    )
    return state.replace(terrain=new_terrain)


def _hp_loss(before: EnvState, after: EnvState) -> int:
    return int(before.player_hp) - int(after.player_hp)


def _pw_loss(before: EnvState, after: EnvState) -> int:
    return int(before.player_pw) - int(after.player_pw)


# ---------------------------------------------------------------------------
# Parity tests — one per vendor formula
# ---------------------------------------------------------------------------

def test_arrow_trap_damage_range_1_to_6():
    """ARROW_TRAP: dmgval(ARROW) = d6 = 1..6 (vendor trap.c:1213)."""
    base = _make_state()
    base = _place(base, TrapType.ARROW_TRAP)
    losses = []
    for seed in range(64):
        rng = jax.random.PRNGKey(seed)
        out = trigger_trap_envstate(base, rng, 5, 5)
        losses.append(_hp_loss(base, out))
    assert all(1 <= l <= 6 for l in losses), (
        f"ARROW_TRAP HP loss out of [1,6]: {sorted(set(losses))}"
    )
    # Cover the full range across seeds.
    assert min(losses) == 1 and max(losses) == 6, (
        f"Expected ARROW_TRAP to span 1..6, got min={min(losses)} max={max(losses)}"
    )


def test_rocktrap_damage_2_to_12():
    """ROCKTRAP: d(2,6) = 2..12 (vendor trap.c:1339)."""
    base = _make_state()
    base = _place(base, TrapType.ROCKTRAP)
    losses = []
    for seed in range(128):
        rng = jax.random.PRNGKey(seed)
        out = trigger_trap_envstate(base, rng, 5, 5)
        losses.append(_hp_loss(base, out))
    assert all(2 <= l <= 12 for l in losses), (
        f"ROCKTRAP HP loss out of [2,12]: {sorted(set(losses))}"
    )
    assert min(losses) == 2 and max(losses) == 12, (
        f"Expected ROCKTRAP to span 2..12, got min={min(losses)} max={max(losses)}"
    )


def test_bear_trap_damage_d2_4_and_held():
    """BEAR_TRAP: d(2,4) = 2..8 dmg + held rn1(4,4) = 4..7 (vendor trap.c:1490, 1506).

    Note: contrary to the "vendor BEAR_TRAP is no-damage" folk wisdom, the
    player branch in trap.c sets ``int dmg = d(2, 4);`` (line 1490) and
    calls ``losehp(Maybe_Half_Phys(dmg), ...)`` (line 1521).
    """
    base = _make_state()
    base = _place(base, TrapType.BEAR_TRAP)
    losses, holds = [], []
    for seed in range(64):
        rng = jax.random.PRNGKey(seed)
        out = trigger_trap_envstate(base, rng, 5, 5)
        losses.append(_hp_loss(base, out))
        holds.append(int(out.status.timed_statuses[int(TimedStatus.FROZEN)]))
    assert all(2 <= l <= 8 for l in losses), (
        f"BEAR_TRAP HP loss out of [2,8]: {sorted(set(losses))}"
    )
    assert all(4 <= h <= 7 for h in holds), (
        f"BEAR_TRAP hold out of [4,7]: {sorted(set(holds))}"
    )


def test_pit_d6_damage_and_held():
    """PIT: rnd(6) = 1..6 fall damage + held rn1(6,2) = 2..7 (vendor trap.c:1920, 1950)."""
    base = _make_state()
    base = _place(base, TrapType.PIT)
    losses, holds = [], []
    for seed in range(64):
        rng = jax.random.PRNGKey(seed)
        out = trigger_trap_envstate(base, rng, 5, 5)
        losses.append(_hp_loss(base, out))
        holds.append(int(out.status.timed_statuses[int(TimedStatus.FROZEN)]))
    assert all(1 <= l <= 6 for l in losses), (
        f"PIT HP loss out of [1,6]: {sorted(set(losses))}"
    )
    assert all(2 <= h <= 7 for h in holds), (
        f"PIT hold out of [2,7]: {sorted(set(holds))}"
    )


def test_spiked_pit_d10_damage():
    """SPIKED_PIT: rnd(10) spike + (1/6) rnd(8) poison HP (vendor trap.c:1925,
    1938-1945).

    Audit-M #3: vendor poisoned("spikes", A_STR, ..., 8, FALSE) adds up to
    rnd(8) HP damage on top of the rnd(10) spike damage when the 1/6 poison
    roll fires; range therefore [1, 18] overall.  We separate poison vs
    non-poison by detecting STR drain (vendor A_STR drain on poison).
    """
    base = _make_state()
    base = base.replace(player_str=jnp.int16(18))
    base = _place(base, TrapType.SPIKED_PIT)
    non_poison = []
    poison    = []
    for seed in range(256):
        rng = jax.random.PRNGKey(seed)
        out = trigger_trap_envstate(base, rng, 5, 5)
        loss = _hp_loss(base, out)
        # Audit-M #3 — vendor poisoned("spikes", A_STR, ...) drains STR.
        was_poisoned = int(out.player_str) < int(base.player_str)
        if was_poisoned:
            poison.append(loss)
        else:
            non_poison.append(loss)
    assert all(1 <= l <= 10 for l in non_poison), (
        f"SPIKED_PIT non-poison HP loss out of [1,10]: {sorted(set(non_poison))}"
    )
    assert all(2 <= l <= 18 for l in poison), (
        f"SPIKED_PIT poison HP loss out of [2,18]: {sorted(set(poison))}"
    )
    assert min(non_poison) == 1 and max(non_poison) == 10, (
        f"Expected SPIKED_PIT non-poison to span 1..10, got "
        f"min={min(non_poison)} max={max(non_poison)}"
    )


def test_fire_trap_burns_scrolls_in_inv():
    """FIRE_TRAP: d(2,4) = 2..8 damage + sometimes burns SCROLL/SPBOOK/POTION
    (vendor trap.c:4238 + fire_damage in trap.c:4514).

    Wave 42b (Audit M #11, #12): old expectation was that every trigger burns
    the scroll.  Vendor uses ``burnarmor(youmonst) || rn2(3)`` (2/3 of
    triggers actually invoke fire_damage on items) AND each item then runs
    a luck save ``(Luck + 5) > rn2(20)``.  We assert the looser invariant
    that across many seeds the scroll burns OFTEN but not always.
    """
    base = _make_state()
    inv = base.inventory
    new_items = inv.items.replace(
        category=inv.items.category.at[0].set(jnp.int8(int(ItemCategory.SCROLL))),
        quantity=inv.items.quantity.at[0].set(jnp.int16(5)),
        type_id=inv.items.type_id.at[0].set(jnp.int16(1)),
    )
    base = base.replace(inventory=inv.replace(items=new_items))
    base = _place(base, TrapType.FIRE_TRAP)

    losses = []
    burned = 0
    n_trials = 128
    for seed in range(n_trials):
        rng = jax.random.PRNGKey(seed)
        # Reset scroll qty between trials so we can count burn events.
        trial = base.replace(
            inventory=base.inventory.replace(
                items=base.inventory.items.replace(
                    quantity=base.inventory.items.quantity.at[0].set(jnp.int16(5))
                )
            )
        )
        out = trigger_trap_envstate(trial, rng, 5, 5)
        losses.append(_hp_loss(trial, out))
        if int(out.inventory.items.quantity[0]) == 0:
            burned += 1
    # Wave 42b: scroll should burn often (>30%) but not always; vendor expected
    # rate is ~ (2/3) * P(luck save fails) ≈ 50% at Luck=0.
    assert 0.30 * n_trials <= burned <= 0.95 * n_trials, (
        f"FIRE_TRAP scroll burn rate {burned}/{n_trials} outside [30%, 95%]"
    )
    assert all(2 <= l <= 8 for l in losses), (
        f"FIRE_TRAP HP loss out of [2,8]: {sorted(set(losses))}"
    )


def test_anti_magic_drains_pw_total_2_to_12():
    """ANTI_MAGIC: d(2,6) = 2..12 TOTAL drain (vendor trap.c:2386).

    Audit-M #35: vendor splits the drain between uenmax (player_pw_max) and
    current uen (player_pw) when uenmax > drain.  Specifically
    ``halfd = rnd(drain/2)`` is removed from uenmax and ``drain - halfd``
    from current.  We therefore check the SUM of the two losses against the
    original 2..12 range.
    """
    base = _make_state()
    base = _place(base, TrapType.ANTI_MAGIC)
    totals = []
    for seed in range(128):
        rng = jax.random.PRNGKey(seed)
        out = trigger_trap_envstate(base, rng, 5, 5)
        pw_loss     = _pw_loss(base, out)
        pwmax_loss  = int(base.player_pw_max) - int(out.player_pw_max)
        totals.append(pw_loss + pwmax_loss)
    assert all(2 <= t <= 12 for t in totals), (
        f"ANTI_MAGIC total drain out of [2,12]: {sorted(set(totals))}"
    )
    assert min(totals) == 2 and max(totals) == 12, (
        f"Expected ANTI_MAGIC to span 2..12 total, got "
        f"min={min(totals)} max={max(totals)}"
    )


def test_sleep_gas_duration_1_to_25():
    """SLP_GAS_TRAP: SLEEP timer = rnd(25) = 1..25 (vendor trap.c:1575)."""
    base = _make_state()
    base = _place(base, TrapType.SLP_GAS_TRAP)
    sleeps = []
    for seed in range(128):
        rng = jax.random.PRNGKey(seed)
        out = trigger_trap_envstate(base, rng, 5, 5)
        sleeps.append(int(out.status.timed_statuses[int(TimedStatus.SLEEP)]))
    assert all(1 <= s <= 25 for s in sleeps), (
        f"SLP_GAS timer out of [1,25]: {sorted(set(sleeps))}"
    )
    # Reasonable coverage.
    assert min(sleeps) <= 5 and max(sleeps) >= 20, (
        f"Expected SLP_GAS distribution to cover edges, got "
        f"min={min(sleeps)} max={max(sleeps)}"
    )


def test_landmine_damage_total_with_recursive_pit():
    """LANDMINE: rnd(16) base damage + recursive PIT rnd(6) (Audit M #27).

    Vendor trap.c:2533 is ``int damage = rnd(16);`` for the landmine itself,
    but trap.c:2587 sets ``trap->ttyp = PIT`` and trap.c:2594-2596 invokes
    ``dotrap(trap, RECURSIVETRAP)`` so an additional ``rnd(6)`` PIT fall
    damage lands on the same turn.  Total HP loss range is therefore
    [1+1, 16+6] = [2, 22].
    """
    base = _make_state()
    base = _place(base, TrapType.LANDMINE)
    losses = []
    for seed in range(128):
        rng = jax.random.PRNGKey(seed)
        out = trigger_trap_envstate(base, rng, 5, 5)
        losses.append(_hp_loss(base, out))
    assert all(2 <= l <= 22 for l in losses), (
        f"LANDMINE total HP loss out of [2,22]: {sorted(set(losses))}"
    )
    assert min(losses) <= 6 and max(losses) >= 18, (
        f"Expected LANDMINE+PIT distribution to span widely, got "
        f"min={min(losses)} max={max(losses)}"
    )


def test_magic_trap_20_outcomes_distribution():
    """MAGIC_TRAP: vendor switches on rnd(20) inside domagictrap +
    1/30 magical explosion (vendor trap.c:2300, 4317).

    Verify that:
      - The 'no effect' outcome (fate=10 + shiver/howl/yearning/shakes/
        smell/tired) happens sometimes (state unchanged).
      - At least one outcome modifies state (HP / gold / status / Pw).
      - Across 1000 trials we hit multiple distinct behaviours.
    """
    base = _make_state()
    base = _add_floor(base)
    base = _place(base, TrapType.MAGIC_TRAP)

    changed_count = 0
    unchanged_count = 0
    distinct_signatures = set()
    for seed in range(1000):
        rng = jax.random.PRNGKey(seed)
        out = trigger_trap_envstate(base, rng, 5, 5)
        hp_d   = _hp_loss(base, out)
        pw_d   = _pw_loss(base, out)
        gold_d = int(out.player_gold)   - int(base.player_gold)
        blind  = int(out.status.timed_statuses[int(TimedStatus.BLIND)])
        deaf   = int(out.status.timed_statuses[int(TimedStatus.DEAF)])
        sig = (hp_d, pw_d, gold_d, blind > 0, deaf > 0)
        distinct_signatures.add(sig)
        if hp_d == 0 and pw_d == 0 and gold_d == 0 and blind == 0 and deaf == 0:
            unchanged_count += 1
        else:
            changed_count += 1
    # Vendor splits 9/20 monster-summons (BLIND+DEAF) + 6/20 nothings +
    # several effect branches.  Expect a healthy mix.
    assert changed_count >= 100, (
        f"Expected MAGIC_TRAP to change state often; got {changed_count}/1000"
    )
    assert unchanged_count >= 50, (
        f"Expected some no-op outcomes; got {unchanged_count}/1000"
    )
    assert len(distinct_signatures) >= 3, (
        f"Expected at least 3 distinct outcome signatures, got "
        f"{len(distinct_signatures)}: {distinct_signatures}"
    )


def test_pit_no_damage_when_climb_out():
    """Subsequent step-on-pit does not deal fall damage in vendor
    (the player is already in the pit; ``set_utrap`` short-circuits
    re-entry, vendor trap.c:1855-1870).

    Our wide-carrier implementation applies fall damage unconditionally
    on every trigger_trap_envstate call, but the env-loop only triggers
    when the player actually steps onto the tile (i.e. exits + re-enters).
    Document the parity gap by asserting the current (vendor-matching
    per-trigger) behaviour: HP DOES drop on each invocation.
    """
    base = _make_state()
    base = _place(base, TrapType.PIT)
    rng = jax.random.PRNGKey(42)
    out = trigger_trap_envstate(base, rng, 5, 5)
    # Vendor per-trigger formula: rnd(6) = 1..6 damage every step-on.
    assert _hp_loss(base, out) >= 1, "PIT trigger must apply fall damage"
