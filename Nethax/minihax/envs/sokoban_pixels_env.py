"""Gymnax-compatible pixel environment for Sokoban (Tier 4)."""
import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple, Optional

from Nethax.environment_base.environment_bases import EnvironmentNoAutoReset
from Nethax.minihax.constants import NUM_ACTIONS_TIER4, TILE_SIZE
from Nethax.minihax.game_logic.sokoban import sokoban_step
from Nethax.minihax.primitives.movement import check_stair_goal
from Nethax.minihax.states import SokobanState, EnvParams, SokobanStaticParams


class SokobanPixelsEnv(EnvironmentNoAutoReset):
    """Base class for all Sokoban pixel environments.

    Subclasses must implement:
        - world_gen(rng, params, static_params) -> SokobanState
        - name property
    """

    def __init__(self, static_params: Optional[SokobanStaticParams] = None):
        super().__init__()
        if static_params is None:
            static_params = SokobanStaticParams()
        self.static_params = static_params

        # Load tiles eagerly — must be concrete before JIT tracing
        from Nethax.tiles.renderer import load_tiles
        self._tiles_array = load_tiles()

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
        self, rng: jax.Array, state: SokobanState, action: int, params: EnvParams
    ) -> Tuple[jax.Array, SokobanState, float, bool, dict]:
        # Split RNG for next step
        rng, rng_step = jax.random.split(rng)

        # Execute step
        new_state = sokoban_step(rng_step, state, action, params, self.static_params)

        # Reward matching original MiniHack Sokoban:
        # Win: +1.0, Death/Timeout: 0.0, Running: -0.001 per step + 0.1 per pit filled
        actually_won = (new_state.pits_remaining == 0) & check_stair_goal(
            new_state.player_position, new_state.downstair_position
        )
        pits_filled = (state.pits_remaining - new_state.pits_remaining).astype(jnp.float32)
        running_reward = -0.001 + pits_filled * 0.1
        dead_or_timeout = new_state.terminal & ~actually_won
        reward = jnp.where(actually_won, 1.0, jnp.where(dead_or_timeout, 0.0, running_reward))

        done = new_state.terminal
        info = {
            "pits_remaining": new_state.pits_remaining,
            "timestep": new_state.timestep,
            "won": actually_won,
            "discount": self.discount(new_state, params),
        }

        return (
            lax.stop_gradient(self.get_obs(new_state)),
            lax.stop_gradient(new_state),
            reward,
            done,
            info,
        )

    def reset_env(
        self, rng: jax.Array, params: EnvParams
    ) -> Tuple[jax.Array, SokobanState]:
        rng, rng_gen = jax.random.split(rng)
        state = self.world_gen(rng_gen, params, self.static_params)
        return self.get_obs(state), state

    def world_gen(self, rng: jax.Array, params: EnvParams, static_params: SokobanStaticParams) -> SokobanState:
        """Generate initial state. Must be implemented by subclasses."""
        raise NotImplementedError

    def get_obs(self, state: SokobanState) -> jax.Array:
        from Nethax.minihax.pixel_renderer import render_pixels_no_monsters
        return render_pixels_no_monsters(state, self.static_params, self._tiles_array)

    def is_terminal(self, state: SokobanState, params: EnvParams) -> bool:
        return state.terminal

    @property
    def num_actions(self) -> int:
        return NUM_ACTIONS_TIER4

    def action_space(self, params=None):
        from gymnax.environments.spaces import Discrete
        return Discrete(NUM_ACTIONS_TIER4)

    def observation_space(self, params=None):
        from gymnax.environments.spaces import Box
        h = self.static_params.map_height * TILE_SIZE
        w = self.static_params.map_width * TILE_SIZE
        return Box(0, 255, (h, w, 3), dtype=jnp.uint8)


# ============================================================================
# Individual Sokoban pixel environment classes
# ============================================================================

class Soko1aPixelsEnv(SokobanPixelsEnv):
    def world_gen(self, rng, params, static_params):
        from Nethax.minihax.world_gen.sokoban import make_soko1a
        tile_map, player_pos, stair_pos, pits_remaining = make_soko1a(rng, static_params)
        from Nethax.minihax.primitives.visibility import compute_visible
        visible_map = compute_visible(player_pos, tile_map, static_params.map_height, static_params.map_width)
        return SokobanState(
            map=tile_map,
            player_position=player_pos,
            downstair_position=stair_pos,
            pits_remaining=pits_remaining,
            seen_map=visible_map,
            visible_map=visible_map,
            timestep=0,
            terminal=False,
            state_rng=rng,
        )

    @property
    def name(self) -> str:
        return "Minihax-Soko1a-Pixels-v0"


class Soko1bPixelsEnv(SokobanPixelsEnv):
    def world_gen(self, rng, params, static_params):
        from Nethax.minihax.world_gen.sokoban import make_soko1b
        tile_map, player_pos, stair_pos, pits_remaining = make_soko1b(rng, static_params)
        from Nethax.minihax.primitives.visibility import compute_visible
        visible_map = compute_visible(player_pos, tile_map, static_params.map_height, static_params.map_width)
        return SokobanState(
            map=tile_map,
            player_position=player_pos,
            downstair_position=stair_pos,
            pits_remaining=pits_remaining,
            seen_map=visible_map,
            visible_map=visible_map,
            timestep=0,
            terminal=False,
            state_rng=rng,
        )

    @property
    def name(self) -> str:
        return "Minihax-Soko1b-Pixels-v0"


class Soko2aPixelsEnv(SokobanPixelsEnv):
    def world_gen(self, rng, params, static_params):
        from Nethax.minihax.world_gen.sokoban import make_soko2a
        tile_map, player_pos, stair_pos, pits_remaining = make_soko2a(rng, static_params)
        from Nethax.minihax.primitives.visibility import compute_visible
        visible_map = compute_visible(player_pos, tile_map, static_params.map_height, static_params.map_width)
        return SokobanState(
            map=tile_map,
            player_position=player_pos,
            downstair_position=stair_pos,
            pits_remaining=pits_remaining,
            seen_map=visible_map,
            visible_map=visible_map,
            timestep=0,
            terminal=False,
            state_rng=rng,
        )

    @property
    def name(self) -> str:
        return "Minihax-Soko2a-Pixels-v0"


class Soko2bPixelsEnv(SokobanPixelsEnv):
    def world_gen(self, rng, params, static_params):
        from Nethax.minihax.world_gen.sokoban import make_soko2b
        tile_map, player_pos, stair_pos, pits_remaining = make_soko2b(rng, static_params)
        from Nethax.minihax.primitives.visibility import compute_visible
        visible_map = compute_visible(player_pos, tile_map, static_params.map_height, static_params.map_width)
        return SokobanState(
            map=tile_map,
            player_position=player_pos,
            downstair_position=stair_pos,
            pits_remaining=pits_remaining,
            seen_map=visible_map,
            visible_map=visible_map,
            timestep=0,
            terminal=False,
            state_rng=rng,
        )

    @property
    def name(self) -> str:
        return "Minihax-Soko2b-Pixels-v0"


class Soko3aPixelsEnv(SokobanPixelsEnv):
    def world_gen(self, rng, params, static_params):
        from Nethax.minihax.world_gen.sokoban import make_soko3a
        tile_map, player_pos, stair_pos, pits_remaining = make_soko3a(rng, static_params)
        from Nethax.minihax.primitives.visibility import compute_visible
        visible_map = compute_visible(player_pos, tile_map, static_params.map_height, static_params.map_width)
        return SokobanState(
            map=tile_map,
            player_position=player_pos,
            downstair_position=stair_pos,
            pits_remaining=pits_remaining,
            seen_map=visible_map,
            visible_map=visible_map,
            timestep=0,
            terminal=False,
            state_rng=rng,
        )

    @property
    def name(self) -> str:
        return "Minihax-Soko3a-Pixels-v0"


class Soko3bPixelsEnv(SokobanPixelsEnv):
    def world_gen(self, rng, params, static_params):
        from Nethax.minihax.world_gen.sokoban import make_soko3b
        tile_map, player_pos, stair_pos, pits_remaining = make_soko3b(rng, static_params)
        from Nethax.minihax.primitives.visibility import compute_visible
        visible_map = compute_visible(player_pos, tile_map, static_params.map_height, static_params.map_width)
        return SokobanState(
            map=tile_map,
            player_position=player_pos,
            downstair_position=stair_pos,
            pits_remaining=pits_remaining,
            seen_map=visible_map,
            visible_map=visible_map,
            timestep=0,
            terminal=False,
            state_rng=rng,
        )

    @property
    def name(self) -> str:
        return "Minihax-Soko3b-Pixels-v0"


class Soko4aPixelsEnv(SokobanPixelsEnv):
    def world_gen(self, rng, params, static_params):
        from Nethax.minihax.world_gen.sokoban import make_soko4a
        tile_map, player_pos, stair_pos, pits_remaining = make_soko4a(rng, static_params)
        from Nethax.minihax.primitives.visibility import compute_visible
        visible_map = compute_visible(player_pos, tile_map, static_params.map_height, static_params.map_width)
        return SokobanState(
            map=tile_map,
            player_position=player_pos,
            downstair_position=stair_pos,
            pits_remaining=pits_remaining,
            seen_map=visible_map,
            visible_map=visible_map,
            timestep=0,
            terminal=False,
            state_rng=rng,
        )

    @property
    def name(self) -> str:
        return "Minihax-Soko4a-Pixels-v0"


class Soko4bPixelsEnv(SokobanPixelsEnv):
    def world_gen(self, rng, params, static_params):
        from Nethax.minihax.world_gen.sokoban import make_soko4b
        tile_map, player_pos, stair_pos, pits_remaining = make_soko4b(rng, static_params)
        from Nethax.minihax.primitives.visibility import compute_visible
        visible_map = compute_visible(player_pos, tile_map, static_params.map_height, static_params.map_width)
        return SokobanState(
            map=tile_map,
            player_position=player_pos,
            downstair_position=stair_pos,
            pits_remaining=pits_remaining,
            seen_map=visible_map,
            visible_map=visible_map,
            timestep=0,
            terminal=False,
            state_rng=rng,
        )

    @property
    def name(self) -> str:
        return "Minihax-Soko4b-Pixels-v0"
