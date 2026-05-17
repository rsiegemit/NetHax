"""Parity tests — AD_WERE lycanthropy infection via monster_attack_player.

Vendor reference: vendor/nethack/src/uhitm.c::mhitm_ad_were (line 4265);
  when a were-creature lands a hit on the player and u.ulycn == NON_PM and
  !Protection_from_shape_changers, set_ulycn(monsndx(pa)) is called
  (src/were.c::set_ulycn, line 234).

Three cases:
  1. Hit lands → lycanthropy_form set, lycanthropy_timer > 0.
  2. Miss      → no change.
  3. Already lycanthropic → form unchanged.
"""

import jax
import jax.numpy as jnp
import pytest

# MONSTERS[21] = werewolf  (AT_BITE, AD_WERE, 2, 6)
_WEREWOLF_IDX: int = 21

_RNG = jax.random.PRNGKey(0)


def _make_werewolf_state(*, force_hit: bool, player_hp: int = 30):
    """Return (state, monster_slot=0) with a werewolf injected adjacent.

    ``force_hit``  — True:  set player AC very positive  → tmp always > rnd(20)
                   False: set player AC very negative   → tmp<=0→clamped to 1,
                           rnd(20) always >=1 → never hits.
    """
    from Nethax.nethax.env import NethaxEnv
    from Nethax.nethax.constants import TileType

    env = NethaxEnv()
    state, _ = env.reset(_RNG)

    # Fix player HP so the monster can't accidentally kill the player on the
    # first hit (which sets done and may short-circuit things in some paths).
    state = state.replace(
        player_hp=jnp.int32(player_hp),
        player_hp_max=jnp.int32(max(int(state.player_hp_max), player_hp)),
    )

    # Carve open floor around player so the tile check passes.
    p_row = int(state.player_pos[0])
    p_col = int(state.player_pos[1])
    branch = int(state.dungeon.current_branch)
    level_idx = int(state.dungeon.current_level) - 1
    terrain = state.terrain
    for c in range(max(0, p_col - 1), min(terrain.shape[3], p_col + 3)):
        terrain = terrain.at[branch, level_idx, p_row, c].set(
            jnp.int8(TileType.FLOOR)
        )
    state = state.replace(terrain=terrain)

    # Place a single werewolf in slot 0; zero out all other slots.
    mai = state.monster_ai
    n = mai.alive.shape[0]
    mai = mai.replace(alive=jnp.zeros((n,), dtype=mai.alive.dtype))

    mai = mai.replace(
        alive=mai.alive.at[0].set(True),
        hp=mai.hp.at[0].set(jnp.int32(50)),
        hp_max=mai.hp_max.at[0].set(jnp.int32(50)),
        pos=mai.pos.at[0].set(
            jnp.array([p_row, p_col + 1], dtype=jnp.int16)
        ),
        # entry_idx = 21 (werewolf) — used by _MONSTER_PRIMARY_ADTYP_TABLE
        entry_idx=mai.entry_idx.at[0].set(jnp.int16(_WEREWOLF_IDX)),
        # attack dice: 2d6 for werewolf
        attack_dice_n=mai.attack_dice_n.at[0].set(jnp.int8(2)),
        attack_dice_sides=mai.attack_dice_sides.at[0].set(jnp.int8(6)),
        # Large positive AC → high tmp → guaranteed hit.
        # Large negative AC → tmp clamps to 1 → rnd(20)>=1 → always miss.
        ac=mai.ac.at[0].set(jnp.int8(0)),  # unused for player; see below
    )
    state = state.replace(monster_ai=mai)

    # Adjust *player* AC via intrinsics to force hit/miss.
    # tmp = player_ac + 10 + mlev.  mlev ≈ hp_max//4 = 12.
    # For guaranteed hit: player_ac = 99 → tmp ≈ 121 > rnd(20).
    # For guaranteed miss: player_ac = -99 → raw_tmp ≈ -77 → clamped to 1;
    #   rnd(20) in [1..20] so 1 > [1..20] is False → always miss.
    if force_hit:
        # Default AC=10, mlev≈12 → tmp=32 > rnd(20) always. No change needed.
        pass
    else:
        # Guaranteed miss: make player AC very negative so tmp clamps to 1,
        # which never beats rnd(1..20).
        # combat.compute_ac sums worn_armor_ac_bonus over 7 slots; set all
        # to -10 each → bonus = +70 → armor_ac = 10 - 70 = -60;
        # tmp = -60 + 10 + 12 = -38, clamped to 1 → never hits.
        # AC formula: uac = 10 - sum(worn_armor_ac_bonus).
        # Set each slot to +10 → sum=+70 → AC = 10-70 = -60.
        # tmp = -60 + 10 + mlev(≈12) = -38, clamped to 1 → never > rnd(1..20).
        from Nethax.nethax.subsystems.inventory import N_ARMOR_SLOTS
        inv = state.inventory
        inv = inv.replace(
            worn_armor_ac_bonus=jnp.full(
                (N_ARMOR_SLOTS,), 10, dtype=jnp.int8
            )
        )
        state = state.replace(inventory=inv)

    return state


def test_werewolf_bite_infects():
    """A werewolf hit infects the (non-lycanthropic) player.

    Vendor ref: uhitm.c::mhitm_ad_were (line 4279): set_ulycn(monsndx(pa)).
    """
    from Nethax.nethax.subsystems.combat import monster_attack_player

    state = _make_werewolf_state(force_hit=True)

    # Confirm precondition: no lycanthropy yet.
    assert int(state.polymorph.lycanthropy_form) < 0

    new_state, _dmg = monster_attack_player(state, _RNG, jnp.int32(0))

    assert int(new_state.polymorph.lycanthropy_form) == _WEREWOLF_IDX, (
        f"Expected lycanthropy_form={_WEREWOLF_IDX}, "
        f"got {int(new_state.polymorph.lycanthropy_form)}"
    )
    # trigger_lycanthropy sets poly_timer (= _LYCANTHROPY_FORM_DURATION = 20)
    # and is_polymorphed=True; lycanthropy_timer is the separate were-form
    # auto-trigger countdown, not the transformation duration.
    assert bool(new_state.polymorph.is_polymorphed), (
        "Player should be polymorphed into were-form after infection"
    )
    assert int(new_state.polymorph.poly_timer) > 0, (
        "poly_timer should be positive after were-form infection"
    )


def test_were_bite_misses_no_infect():
    """A missed were-creature attack does NOT infect.

    Vendor ref: uhitm.c::mhitm_ad_were (line 4279): infection only on hit.
    """
    from Nethax.nethax.subsystems.combat import monster_attack_player

    state = _make_werewolf_state(force_hit=False)

    assert int(state.polymorph.lycanthropy_form) < 0

    new_state, dmg = monster_attack_player(state, _RNG, jnp.int32(0))

    assert int(new_state.polymorph.lycanthropy_form) < 0, (
        "Miss should not set lycanthropy_form"
    )
    assert int(new_state.polymorph.lycanthropy_timer) == 0, (
        "Miss should not set lycanthropy_timer"
    )


def test_already_infected_no_change():
    """A second were-bite does NOT change an already-lycanthropic player's form.

    Vendor ref: uhitm.c::mhitm_ad_were (line 4279): gate ``u.ulycn == NON_PM``.
    """
    from Nethax.nethax.subsystems.combat import monster_attack_player

    # Pre-infect player with werejackal (idx=15).
    _WEREJACKAL_IDX: int = 15
    state = _make_werewolf_state(force_hit=True)
    state = state.replace(
        polymorph=state.polymorph.replace(
            lycanthropy_form=jnp.int8(_WEREJACKAL_IDX),
            lycanthropy_timer=jnp.int16(100),
        )
    )

    new_state, _dmg = monster_attack_player(state, _RNG, jnp.int32(0))

    assert int(new_state.polymorph.lycanthropy_form) == _WEREJACKAL_IDX, (
        "lycanthropy_form should remain werejackal; werewolf should not overwrite"
    )
