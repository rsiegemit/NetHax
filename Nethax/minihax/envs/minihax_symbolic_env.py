"""Gymnax-compatible symbolic environment for ZombieHorde."""
import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple, Optional

from Nethax.environment_base.environment_bases import EnvironmentNoAutoReset
from Nethax.minihax.constants import NUM_ACTIONS, NUM_TILE_TYPES
from Nethax.minihax.game_logic.zombie_horde import minihax_step, is_game_over
from Nethax.minihax.minihax_state import EnvState, EnvParams, StaticEnvParams
from Nethax.minihax.renderer import render_minihax_symbolic
from Nethax.minihax.world_gen.zombie_horde import generate_zombie_horde
from Nethax.minihax.envs.common import log_zombie_info


def get_obs_shape(static_params=None):
    if static_params is None:
        static_params = StaticEnvParams()
    map_h = static_params.map_height
    map_w = static_params.map_width
    # map_onehot + player_pos + monster_presence + seen_map + visible_map + stats
    map_obs = map_h * map_w * NUM_TILE_TYPES + map_h * map_w + map_h * map_w + 2 * map_h * map_w
    stats_obs = 8
    return map_obs + stats_obs


class MinihaxZombieHordeSymbolicEnv(EnvironmentNoAutoReset):
    def __init__(self, static_env_params: Optional[StaticEnvParams] = None):
        super().__init__()
        if static_env_params is None:
            static_env_params = StaticEnvParams()
        self.static_env_params = static_env_params

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
        self, rng: jax.Array, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[jax.Array, EnvState, float, bool, dict]:
        state, reward = minihax_step(rng, state, action, params, self.static_env_params)
        done = self.is_terminal(state, params)
        info = log_zombie_info(state, done)
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
        state = generate_zombie_horde(_rng, params, self.static_env_params)
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> jax.Array:
        return render_minihax_symbolic(state, self.static_env_params)

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        return is_game_over(state, params, self.static_env_params)

    @property
    def name(self) -> str:
        return "Minihax-ZombieHorde-Symbolic-v0"

    @property
    def num_actions(self) -> int:
        return NUM_ACTIONS

    def action_space(self, params=None):
        from gymnax.environments.spaces import Discrete
        return Discrete(NUM_ACTIONS)

    def observation_space(self, params=None):
        from gymnax.environments.spaces import Box
        obs_size = get_obs_shape(self.static_env_params)
        return Box(0.0, 1.0, (obs_size,), dtype=jnp.float32)
