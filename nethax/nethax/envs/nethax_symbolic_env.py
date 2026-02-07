import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import spaces, environment
from typing import Tuple, Optional

from nethax.nethax.envs.common import log_achievements_to_info
from nethax.environment_base.environment_bases import EnvironmentNoAutoReset
from nethax.nethax.constants import *
from nethax.nethax.game_logic import nethax_step, is_game_over
from nethax.nethax.nethax_state import EnvState, EnvParams, StaticEnvParams
from nethax.nethax.renderer import render_nethax_symbolic
from nethax.nethax.world_gen.world_gen import generate_world


def get_map_obs_shape(static_params=None):
    if static_params is None:
        static_params = StaticEnvParams()
    map_h, map_w = static_params.map_size
    # Full map: tile one-hot + player position + monster map
    return map_h * map_w * NUM_TILE_TYPES + map_h * map_w + map_h * map_w


def get_stats_obs_shape():
    """Player stats + status effects + intrinsics + hunger + inventory summary + identification."""
    stats = 8
    status = 6
    intrinsics = NUM_INTRINSICS
    hunger = len(HungerState)
    inv_summary = len(ItemCategory)
    id_knowledge = NUM_POTION_TYPES + NUM_SCROLL_TYPES
    return stats + status + intrinsics + hunger + inv_summary + id_knowledge


class NethaxSymbolicEnvNoAutoReset(EnvironmentNoAutoReset):
    def __init__(self, static_env_params: Optional[StaticEnvParams] = None):
        super().__init__()

        if static_env_params is None:
            static_env_params = self.default_static_params()
        self.static_env_params = static_env_params

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @staticmethod
    def default_static_params() -> StaticEnvParams:
        return StaticEnvParams()

    def step_env(
        self, rng: jax.Array, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[jax.Array, EnvState, float, bool, dict]:
        state, reward = nethax_step(rng, state, action, params, self.static_env_params)

        done = self.is_terminal(state, params)
        info = log_achievements_to_info(state, done)
        info["discount"] = self.discount(state, params)

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def reset_env(
        self, rng: jax.Array, params: EnvParams
    ) -> Tuple[jax.Array, EnvState]:
        rng, _rng = jax.random.split(rng)
        state = generate_world(_rng, params, self.static_env_params)

        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> jax.Array:
        return render_nethax_symbolic(state, self.static_env_params)

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        return is_game_over(state, params, self.static_env_params)

    @property
    def name(self) -> str:
        return "Nethax-Symbolic-NoAutoReset-v1"

    @property
    def num_actions(self) -> int:
        return NUM_ACTIONS

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        return spaces.Discrete(NUM_ACTIONS)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        map_obs_shape = get_map_obs_shape(self.static_env_params)
        stats_obs_shape = get_stats_obs_shape()
        obs_shape = map_obs_shape + stats_obs_shape

        return spaces.Box(
            0.0,
            1.0,
            (obs_shape,),
            dtype=jnp.float32,
        )


class NethaxSymbolicEnv(environment.Environment):
    def __init__(self, static_env_params: Optional[StaticEnvParams] = None):
        super().__init__()

        if static_env_params is None:
            static_env_params = self.default_static_params()
        self.static_env_params = static_env_params

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @staticmethod
    def default_static_params() -> StaticEnvParams:
        return StaticEnvParams()

    def step_env(
        self, rng: jax.Array, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[jax.Array, EnvState, float, bool, dict]:
        state, reward = nethax_step(rng, state, action, params, self.static_env_params)

        done = self.is_terminal(state, params)
        info = log_achievements_to_info(state, done)
        info["discount"] = self.discount(state, params)

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def reset_env(
        self, rng: jax.Array, params: EnvParams
    ) -> Tuple[jax.Array, EnvState]:
        rng, _rng = jax.random.split(rng)
        state = generate_world(_rng, params, self.static_env_params)

        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> jax.Array:
        return render_nethax_symbolic(state, self.static_env_params)

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        return is_game_over(state, params, self.static_env_params)

    @property
    def name(self) -> str:
        return "Nethax-Symbolic-v1"

    @property
    def num_actions(self) -> int:
        return NUM_ACTIONS

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        return spaces.Discrete(NUM_ACTIONS)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        map_obs_shape = get_map_obs_shape(self.static_env_params)
        stats_obs_shape = get_stats_obs_shape()
        obs_shape = map_obs_shape + stats_obs_shape

        return spaces.Box(
            0.0,
            1.0,
            (obs_shape,),
            dtype=jnp.float32,
        )
