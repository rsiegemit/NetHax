"""Gymnax-compatible pixel environment for ZombieHorde."""
import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple, Optional

from Nethax.environment_base.environment_bases import EnvironmentNoAutoReset
from Nethax.minihax.constants import NUM_ACTIONS, TILE_SIZE
from Nethax.minihax.game_logic.zombie_horde import minihax_step, is_game_over
from Nethax.minihax.states import CombatState, CombatStaticParams, EnvParams
from Nethax.minihax.world_gen.zombie_horde import generate_zombie_horde
from Nethax.minihax.envs.common import log_zombie_info


class MinihaxZombieHordePixelsEnv(EnvironmentNoAutoReset):
    """Pixel-rendered ZombieHorde environment using NetHack tile sprites.

    Observation: [map_h * 16, map_w * 16, 3] RGB uint8 image.
    Requires tiles.npy to be generated via tiles.convert_tiles.
    """
    def __init__(self, static_env_params: Optional[CombatStaticParams] = None):
        super().__init__()
        if static_env_params is None:
            static_env_params = CombatStaticParams(has_temple=True)
        self.static_env_params = static_env_params

        # Load tiles eagerly — must be concrete before JIT tracing
        from Nethax.tiles.renderer import load_tiles
        self._tiles_array = load_tiles()

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
        from Nethax.minihax.pixel_renderer import render_pixels_with_monsters
        return render_pixels_with_monsters(
            state, self.static_env_params, self._tiles_array,
            self.static_env_params.max_monsters,
        )

    def is_terminal(self, state: CombatState, params: EnvParams) -> bool:
        return is_game_over(state, params, self.static_env_params)

    @property
    def name(self) -> str:
        return "Minihax-ZombieHorde-Pixels-v0"

    @property
    def num_actions(self) -> int:
        return NUM_ACTIONS

    def action_space(self, params=None):
        from gymnax.environments.spaces import Discrete
        return Discrete(NUM_ACTIONS)

    def observation_space(self, params=None):
        from gymnax.environments.spaces import Box
        sp = self.static_env_params
        h = sp.map_height * TILE_SIZE
        w = sp.map_width * TILE_SIZE
        return Box(0, 255, (h, w, 3), dtype=jnp.uint8)
