"""Wave 3 integration tests — movement interactions with doors and traps.

Tests cover door-opening on bump, trap triggering on step, and trap
visibility in the observation.

Wave 3 door/trap logic is implemented by parallel agents.  Tests are
guarded with skipif when the features remain stubs.

All imports are lazy so collection never fails.
"""

import pytest


def _make_env_with_door_east(door_tile):
    """Helper: place *door_tile* one step east of the player.

    Returns (env, state, rng).
    """
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.env import NethaxEnv
    from Nethax.nethax.constants import TileType

    rng = jax.random.PRNGKey(11)
    env = NethaxEnv()
    state, _ = env.reset(rng)

    branch = int(state.dungeon.current_branch)
    level_idx = int(state.dungeon.current_level) - 1
    p_row = int(state.player_pos[0])
    p_col = int(state.player_pos[1])
    east_col = p_col + 1

    # Ensure east tile exists (not at map boundary)
    if east_col >= state.terrain.shape[3]:
        pytest.skip("Player at eastern map boundary; cannot place door east")

    new_terrain = state.terrain
    # Player tile = FLOOR
    new_terrain = new_terrain.at[branch, level_idx, p_row, p_col].set(
        jnp.int8(TileType.FLOOR)
    )
    # Door tile east of player
    new_terrain = new_terrain.at[branch, level_idx, p_row, east_col].set(
        jnp.int8(door_tile)
    )
    # Tile beyond the door = FLOOR (so player can walk through)
    if east_col + 1 < state.terrain.shape[3]:
        new_terrain = new_terrain.at[branch, level_idx, p_row, east_col + 1].set(
            jnp.int8(TileType.FLOOR)
        )
    state = state.replace(terrain=new_terrain)
    return env, state, rng


def test_walk_through_door():
    """Bump CLOSED_DOOR -> door opens; next step walks through."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.constants import TileType
    from Nethax.nethax.constants.actions import CompassCardinalDirection

    env, state, rng = _make_env_with_door_east(TileType.CLOSED_DOOR)

    p_col_before = int(state.player_pos[1])
    east_col = p_col_before + 1

    action_e = jnp.int32(int(CompassCardinalDirection.E))

    # First bump: should open the door (player does NOT move yet)
    rng, step_rng = jax.random.split(rng)
    state, _, _, _, _ = env.step(state, action_e, step_rng)

    branch = int(state.dungeon.current_branch)
    level_idx = int(state.dungeon.current_level) - 1
    door_tile_after_bump = int(state.terrain[branch, level_idx,
                                             int(state.player_pos[0]), east_col])
    assert door_tile_after_bump == int(TileType.OPEN_DOOR), (
        f"Expected OPEN_DOOR after bump, got tile={door_tile_after_bump}"
    )
    assert int(state.player_pos[1]) == p_col_before, (
        "Player should not move on door-bump turn"
    )

    # Second step: player walks through the now-open door
    rng, step_rng = jax.random.split(rng)
    state, _, _, _, _ = env.step(state, action_e, step_rng)
    assert int(state.player_pos[1]) == east_col, (
        f"Expected player col={east_col} after walking through door, "
        f"got {int(state.player_pos[1])}"
    )


def test_trap_triggers_on_step():
    """Step onto ARROW_TRAP tile -> HP decreases, trap.revealed = True."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.constants import TileType
    from Nethax.nethax.subsystems.traps import TrapType
    from Nethax.nethax.constants.actions import CompassCardinalDirection

    rng = jax.random.PRNGKey(13)
    from Nethax.nethax.env import NethaxEnv
    env = NethaxEnv()
    state, _ = env.reset(rng)

    branch = int(state.dungeon.current_branch)
    level_idx = int(state.dungeon.current_level) - 1
    p_row = int(state.player_pos[0])
    p_col = int(state.player_pos[1])
    trap_col = p_col + 1

    if trap_col >= state.terrain.shape[3]:
        pytest.skip("Player at eastern map boundary; cannot place trap east")

    # Place floor with HIDDEN_TRAP tile type (trap not yet revealed)
    new_terrain = state.terrain
    new_terrain = new_terrain.at[branch, level_idx, p_row, p_col].set(
        jnp.int8(TileType.FLOOR)
    )
    new_terrain = new_terrain.at[branch, level_idx, p_row, trap_col].set(
        jnp.int8(TileType.HIDDEN_TRAP)
    )
    state = state.replace(terrain=new_terrain)

    # Set trap type in TrapState
    flat_level = branch * state.terrain.shape[1] + level_idx
    new_trap_type = state.traps.trap_type.at[flat_level, p_row, trap_col].set(
        jnp.int8(TrapType.ARROW_TRAP)
    )
    state = state.replace(traps=state.traps.replace(trap_type=new_trap_type))

    hp_before = int(state.player_hp)

    action_e = jnp.int32(int(CompassCardinalDirection.E))
    rng, step_rng = jax.random.split(rng)
    state, _, _, _, _ = env.step(state, action_e, step_rng)

    hp_after = int(state.player_hp)
    revealed = bool(state.traps.revealed[flat_level, p_row, trap_col])

    assert hp_after < hp_before, (
        f"Expected HP to decrease on ARROW_TRAP, before={hp_before}, after={hp_after}"
    )
    assert revealed, "Expected trap.revealed=True after stepping on it"


def test_revealed_trap_visible():
    """Revealed trap shows in obs['glyphs'] differently than hidden trap."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.constants import TileType
    from Nethax.nethax.subsystems.traps import TrapType
    from Nethax.nethax.obs.nle_obs import _S_trap, GLYPH_CMAP_OFF

    rng = jax.random.PRNGKey(17)
    from Nethax.nethax.env import NethaxEnv
    env = NethaxEnv()
    state, _ = env.reset(rng)

    branch = int(state.dungeon.current_branch)
    level_idx = int(state.dungeon.current_level) - 1
    p_row = int(state.player_pos[0])
    p_col = int(state.player_pos[1])

    # Place a hidden trap at (p_row, p_col+2) so it's not under the player
    trap_col = p_col + 2
    if trap_col >= state.terrain.shape[3] - 1:
        pytest.skip("Too close to map boundary")

    new_terrain = state.terrain
    new_terrain = new_terrain.at[branch, level_idx, p_row, trap_col].set(
        jnp.int8(TileType.HIDDEN_TRAP)
    )
    state = state.replace(terrain=new_terrain)

    flat_level = branch * state.terrain.shape[1] + level_idx

    # Mark tile as explored so it shows up in glyphs
    new_explored = state.explored.at[branch, level_idx, p_row, trap_col].set(True)
    state = state.replace(explored=new_explored)

    # Unrevealed: glyph should be floor (S_room + GLYPH_CMAP_OFF)
    from Nethax.nethax.obs.nle_obs import build_glyphs, _S_room
    glyphs_hidden = build_glyphs(state)
    # Clamp col to glyph array width (79)
    glyph_col = min(trap_col, 78)
    glyph_hidden = int(glyphs_hidden[p_row, glyph_col])

    floor_glyph = int(GLYPH_CMAP_OFF) + int(_S_room)
    assert glyph_hidden == floor_glyph, (
        f"Hidden trap should look like floor (glyph={floor_glyph}), "
        f"got {glyph_hidden}"
    )

    # Now reveal the trap
    new_revealed = state.traps.revealed.at[flat_level, p_row, trap_col].set(True)
    # Also update terrain to TRAP tile (revealed trap uses TileType.TRAP)
    new_terrain2 = state.terrain.at[branch, level_idx, p_row, trap_col].set(
        jnp.int8(TileType.TRAP)
    )
    state = state.replace(
        traps=state.traps.replace(revealed=new_revealed),
        terrain=new_terrain2,
    )

    glyphs_revealed = build_glyphs(state)
    glyph_revealed = int(glyphs_revealed[p_row, glyph_col])

    trap_glyph = int(GLYPH_CMAP_OFF) + int(_S_trap)
    assert glyph_revealed == trap_glyph, (
        f"Revealed trap should have trap glyph ({trap_glyph}), "
        f"got {glyph_revealed}"
    )
    assert glyph_revealed != glyph_hidden, (
        "Revealed trap glyph should differ from hidden trap glyph"
    )
