"""Gymnax-compatible NLE-style environment for ZombieHorde."""
import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple, Optional

from Nethax.environment_base.environment_bases import EnvironmentNoAutoReset
from Nethax.minihax.constants import NUM_ACTIONS
from Nethax.minihax.game_logic.zombie_horde import minihax_step, is_game_over
from Nethax.minihax.minihax_state import EnvState, EnvParams, StaticEnvParams
from Nethax.minihax.world_gen.zombie_horde import generate_zombie_horde
from Nethax.minihax.envs.common import log_zombie_info


class MinihaxZombieHordeNLEEnv(EnvironmentNoAutoReset):
    def __init__(self, static_env_params: Optional[StaticEnvParams] = None,
                 crop_size: int = 9):
        super().__init__()
        if static_env_params is None:
            static_env_params = StaticEnvParams()
        self.static_env_params = static_env_params
        self.crop_size = crop_size

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
        self, rng: jax.Array, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[jax.Array, EnvState, float, bool, dict]:
        state, reward = minihax_step(rng, state, action, params, self.static_env_params)

        # Store prev_action in state for observation
        state = state.replace(prev_action=action)

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

    def get_obs(self, state: EnvState) -> dict:
        from Nethax.minihax.nle_obs import render_nle_zombie_horde
        return render_nle_zombie_horde(state, self.static_env_params, self.crop_size, prev_action=state.prev_action)

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        return is_game_over(state, params, self.static_env_params)

    @property
    def name(self) -> str:
        return "Minihax-ZombieHorde-NLE-v0"

    @property
    def num_actions(self) -> int:
        return NUM_ACTIONS

    def action_space(self, params=None):
        from gymnax.environments.spaces import Discrete
        return Discrete(NUM_ACTIONS)

    def observation_space(self, params=None):
        from Nethax.minihax.nle_obs import nle_observation_space
        map_h = self.static_env_params.map_height
        map_w = self.static_env_params.map_width
        return nle_observation_space(map_h, map_w, self.crop_size)
