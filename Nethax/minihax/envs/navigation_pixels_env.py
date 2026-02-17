"""Gymnax-compatible pixel environment for Navigation (Tier 1).

Pixel-rendered versions of all Tier 1 navigation environments (mazes, corridors).
"""
import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple, Optional, Callable

from Nethax.environment_base.environment_bases import EnvironmentNoAutoReset
from Nethax.minihax.constants import NUM_ACTIONS_TIER1, NUM_ACTIONS_TIER1_EXPLORE, TILE_SIZE, ItemType
from Nethax.minihax.game_logic.navigation import navigation_step
from Nethax.minihax.primitives.movement import check_stair_goal
from Nethax.minihax.states import NavigationState, EnvParams, NavigationStaticParams


class NavigationPixelsEnv(EnvironmentNoAutoReset):
    """Base class for all Tier 1 navigation pixel environments.

    Subclasses set a world_gen function that produces NavigationState.
    Step logic uses navigation_step (movement + stair goal check).
    Observation uses render_pixels_no_monsters for pixel rendering.
    """

    def __init__(self, env_name: str, world_gen_fn: Callable,
                 static_params: Optional[NavigationStaticParams] = None):
        super().__init__()
        self._env_name = env_name
        self._world_gen_fn = world_gen_fn
        if static_params is None:
            static_params = NavigationStaticParams()
        self.static_params = static_params

        # Load tiles eagerly — must be concrete before JIT tracing
        from Nethax.tiles.renderer import load_tiles
        self._tiles_array = load_tiles()

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
        self, rng: jax.Array, state: NavigationState, action: int, params: EnvParams
    ) -> Tuple[jax.Array, NavigationState, float, bool, dict]:
        rng, rng_step = jax.random.split(rng)

        new_state = navigation_step(rng_step, state, action, params, self.static_params)

        # Reward: +1 on successful completion (terminal and not timed out)
        won = new_state.terminal & (new_state.timestep < params.max_timesteps)
        reward = jnp.where(won, 1.0, 0.0)

        # Frozen step penalty (NetHack turn consumption rules)
        is_move = action < 8
        moved = jnp.any(new_state.player_position != state.player_position)
        map_changed = jnp.any(new_state.map != state.map)
        turn_consumed = ((is_move & moved)
                         | ((action == 8) & map_changed)
                         | (action == 9)
                         | (action == 10))
        frozen = ~turn_consumed
        reward = reward + jnp.where(frozen, -0.01, 0.0)

        done = new_state.terminal
        info = {
            "timestep": new_state.timestep,
            "won": won,
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
    ) -> Tuple[jax.Array, NavigationState]:
        rng, rng_gen = jax.random.split(rng)
        state = self._world_gen_fn(rng_gen, params, self.static_params)
        return self.get_obs(state), state

    def get_obs(self, state: NavigationState) -> jax.Array:
        from Nethax.minihax.pixel_renderer import render_pixels_no_monsters
        return render_pixels_no_monsters(state, self.static_params, self._tiles_array)

    def is_terminal(self, state: NavigationState, params: EnvParams) -> bool:
        return state.terminal

    @property
    def num_actions(self) -> int:
        return NUM_ACTIONS_TIER1

    def action_space(self, params=None):
        from gymnax.environments.spaces import Discrete
        return Discrete(NUM_ACTIONS_TIER1)

    def observation_space(self, params=None):
        from gymnax.environments.spaces import Box
        h = self.static_params.map_height * TILE_SIZE
        w = self.static_params.map_width * TILE_SIZE
        return Box(0, 255, (h, w, 3), dtype=jnp.uint8)

    @property
    def name(self) -> str:
        return f"Minihax-{self._env_name}-Pixels-v0"


# ============================================================================
# Individual Navigation pixel environment classes
# ============================================================================

class MazewalkPixelsEnv(NavigationPixelsEnv):
    def __init__(self):
        from Nethax.minihax.world_gen.mazewalk_envs import generate_mazewalk
        super().__init__("Mazewalk", generate_mazewalk)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(max_timesteps=1000)


class ExploreMazeNavPixelsEnv(NavigationPixelsEnv):
    """Base for ExploreMaze pixel envs: repeatable stair reward + apple eating, timeout-only terminal.

    Original MiniHack ExploreMaze:
      - custom_reward_fn: +1 per step while on stair (repeatable, not terminal)
      - +0.5 per apple eaten (EAT action while on apple tile)
      - max_episode_steps=500
      - Never terminates on task success, only timeout

    Action space: Tier 1 base (0-10) + EAT(11) = 12 actions.
    """

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(max_timesteps=500)

    @property
    def num_actions(self) -> int:
        return NUM_ACTIONS_TIER1_EXPLORE

    def action_space(self, params=None):
        from gymnax.environments.spaces import Discrete
        return Discrete(NUM_ACTIONS_TIER1_EXPLORE)

    def step_env(
        self, rng: jax.Array, state: NavigationState, action: int, params: EnvParams
    ) -> Tuple[jax.Array, NavigationState, float, bool, dict]:
        rng, rng_step = jax.random.split(rng)

        new_state = navigation_step(rng_step, state, action, params, self.static_params)

        # Override terminal: timeout only (ExploreMaze never ends on stair)
        new_timestep = new_state.timestep
        timeout = new_timestep >= params.max_timesteps
        new_state = new_state.replace(terminal=timeout)

        # --- EAT action: eat apple at player position ---
        is_eat = (action == 11)  # EAT action in Tier 1 ExploreMaze
        player_pos = new_state.player_position
        gi = state.ground_items  # use pre-step ground items
        at_player = ((gi.position[:, 0] == player_pos[0]) &
                     (gi.position[:, 1] == player_pos[1]) &
                     gi.mask &
                     (gi.type_id == ItemType.APPLE))
        any_apple = jnp.any(at_player)
        ate = is_eat & any_apple
        # Remove first matching apple
        first_idx = jnp.argmax(at_player)
        new_mask = gi.mask.at[first_idx].set(jnp.where(ate, False, gi.mask[first_idx]))
        new_gi = gi.replace(mask=new_mask)
        new_state = new_state.replace(ground_items=new_gi)

        # --- Reward ---
        # +0.5 per apple eaten + 1.0 per step on stair (repeatable)
        on_stair = check_stair_goal(new_state.player_position, state.downstair_position)
        eat_reward = jnp.where(ate, 0.5, 0.0)
        stair_reward = jnp.where(on_stair, 1.0, 0.0)
        reward = eat_reward + stair_reward

        # Frozen step penalty (NetHack turn consumption rules)
        is_move = action < 8
        moved = jnp.any(new_state.player_position != state.player_position)
        map_changed = jnp.any(new_state.map != state.map)
        turn_consumed = ((is_move & moved)
                         | ((action == 8) & map_changed)
                         | (action == 9)
                         | (action == 10)
                         | ate)
        frozen = ~turn_consumed
        reward = reward + jnp.where(frozen, -0.01, 0.0)

        done = new_state.terminal
        info = {
            "timestep": new_state.timestep,
            "won": on_stair,
            "discount": self.discount(new_state, params),
        }

        return (
            lax.stop_gradient(self.get_obs(new_state)),
            lax.stop_gradient(new_state),
            reward,
            done,
            info,
        )


class ExploreMazeEasyPixelsEnv(ExploreMazeNavPixelsEnv):
    def __init__(self):
        from Nethax.minihax.world_gen.mazewalk_envs import generate_explore_maze_easy
        super().__init__("ExploreMazeEasy", generate_explore_maze_easy)


class ExploreMazeEasyPremappedPixelsEnv(ExploreMazeNavPixelsEnv):
    def __init__(self):
        from Nethax.minihax.world_gen.mazewalk_envs import generate_explore_maze_easy_premapped
        super().__init__("ExploreMazeEasyPremapped", generate_explore_maze_easy_premapped)


class ExploreMazeHardPixelsEnv(ExploreMazeNavPixelsEnv):
    def __init__(self):
        from Nethax.minihax.world_gen.mazewalk_envs import generate_explore_maze_hard
        super().__init__("ExploreMazeHard", generate_explore_maze_hard)


class ExploreMazeHardPremappedPixelsEnv(ExploreMazeNavPixelsEnv):
    def __init__(self):
        from Nethax.minihax.world_gen.mazewalk_envs import generate_explore_maze_hard_premapped
        super().__init__("ExploreMazeHardPremapped", generate_explore_maze_hard_premapped)


# ============================================================================
# Corridor pixel environments (RANDOM_CORRIDORS)
# max_episode_steps=1000 matching MiniHack MiniHackCorridor
# ============================================================================

class CorridorPixelsEnvBase(NavigationPixelsEnv):
    """Base for corridor pixel envs with max_timesteps=1000 (matching MiniHack)."""
    @property
    def default_params(self) -> EnvParams:
        return EnvParams(max_timesteps=1000)


class Corridor2PixelsEnv(CorridorPixelsEnvBase):
    def __init__(self):
        from Nethax.minihax.world_gen.corridor import generate_corridor2
        super().__init__("Corridor2", generate_corridor2)


class Corridor3PixelsEnv(CorridorPixelsEnvBase):
    def __init__(self):
        from Nethax.minihax.world_gen.corridor import generate_corridor3
        super().__init__("Corridor3", generate_corridor3)


class Corridor5PixelsEnv(CorridorPixelsEnvBase):
    def __init__(self):
        from Nethax.minihax.world_gen.corridor import generate_corridor5
        super().__init__("Corridor5", generate_corridor5)


class Corridor8PixelsEnv(CorridorPixelsEnvBase):
    def __init__(self):
        from Nethax.minihax.world_gen.corridor import generate_corridor8
        super().__init__("Corridor8", generate_corridor8)


class Corridor10PixelsEnv(CorridorPixelsEnvBase):
    def __init__(self):
        from Nethax.minihax.world_gen.corridor import generate_corridor10
        super().__init__("Corridor10", generate_corridor10)
