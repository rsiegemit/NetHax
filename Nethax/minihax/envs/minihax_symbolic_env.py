"""Gymnax-compatible symbolic environment for ZombieHorde."""
import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple, Optional

from Nethax.environment_base.environment_bases import EnvironmentNoAutoReset
from Nethax.minihax.constants import NUM_ACTIONS, NUM_TILE_TYPES, NUM_ITEM_TYPES
from Nethax.minihax.game_logic.zombie_horde import minihax_step, is_game_over
from Nethax.minihax.states import CombatState, CombatStaticParams, EnvParams
from Nethax.minihax.renderer import render_combat_symbolic
from Nethax.minihax.world_gen.zombie_horde import generate_zombie_horde
from Nethax.minihax.envs.common import log_zombie_info


def get_obs_shape(static_params=None):
    if static_params is None:
        static_params = CombatStaticParams(has_temple=True)
    map_h = static_params.map_height
    map_w = static_params.map_width
    max_items = static_params.max_items
    # map_onehot + player + stair + monsters + traps + seen + visible + inventory + stats
    obs_size = (map_h * map_w * NUM_TILE_TYPES +
                map_h * map_w +      # player
                map_h * map_w +      # stair
                map_h * map_w +      # monsters
                map_h * map_w +      # traps
                2 * map_h * map_w +  # seen_map + visible_map
                max_items +          # inventory
                17)                  # stats
    return obs_size


class MinihaxZombieHordeSymbolicEnv(EnvironmentNoAutoReset):
    def __init__(self, static_env_params: Optional[CombatStaticParams] = None):
        super().__init__()
        if static_env_params is None:
            static_env_params = CombatStaticParams(has_temple=True)
        self.static_env_params = static_env_params

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
        self, rng: jax.Array, state: CombatState, action: int, params: EnvParams
    ) -> Tuple[jax.Array, CombatState, float, bool, dict]:
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
    ) -> Tuple[jax.Array, CombatState]:
        rng, _rng = jax.random.split(rng)
        state = generate_zombie_horde(_rng, params, self.static_env_params)
        return self.get_obs(state), state

    def get_obs(self, state: CombatState) -> jax.Array:
        return render_combat_symbolic(state, self.static_env_params)

    def is_terminal(self, state: CombatState, params: EnvParams) -> bool:
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
