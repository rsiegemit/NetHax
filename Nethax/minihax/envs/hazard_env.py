"""Gymnax-compatible environment for Hazard (Tier 2).

Parametric env class for all Tier 2 hazard environments (lava, items, doors,
simple monsters). Follows the pattern from navigation_env.py.
"""
import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple, Optional, Callable

from Nethax.environment_base.environment_bases import EnvironmentNoAutoReset
from Nethax.minihax.constants import NUM_ACTIONS_TIER2, NUM_TILE_TYPES
from Nethax.minihax.game_logic.hazard import hazard_step
from Nethax.minihax.states import HazardState, EnvParams, HazardStaticParams


class HazardEnv(EnvironmentNoAutoReset):
    """Base class for all Tier 2 hazard environments.

    Subclasses set a world_gen function that produces HazardState.
    Step logic uses hazard_step (movement + lava + items + simple monsters + doors).
    Observation uses render_hazard_symbolic.
    """

    def __init__(self, env_name: str, world_gen_fn: Callable,
                 static_params: Optional[HazardStaticParams] = None):
        super().__init__()
        self._env_name = env_name
        self._world_gen_fn = world_gen_fn
        if static_params is None:
            static_params = HazardStaticParams()
        self.static_params = static_params

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
        self, rng: jax.Array, state: HazardState, action: int, params: EnvParams
    ) -> Tuple[jax.Array, HazardState, float, bool, dict]:
        rng, rng_step = jax.random.split(rng)

        new_state, reward = hazard_step(
            rng_step, state, action, params, self.static_params
        )

        # Win = terminal and not timed out and not dead
        won = new_state.terminal & (new_state.timestep < params.max_timesteps) & (new_state.player_stats.hp > 0)

        done = new_state.terminal
        info = {
            "timestep": new_state.timestep,
            "won": won,
            "player_hp": new_state.player_stats.hp,
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
    ) -> Tuple[jax.Array, HazardState]:
        rng, rng_gen = jax.random.split(rng)
        state = self._world_gen_fn(rng_gen, params, self.static_params)
        return self.get_obs(state), state

    def get_obs(self, state: HazardState) -> jax.Array:
        from Nethax.minihax.renderer import render_hazard_symbolic
        return render_hazard_symbolic(state, self.static_params)

    def is_terminal(self, state: HazardState, params: EnvParams) -> bool:
        return state.terminal

    @property
    def num_actions(self) -> int:
        return NUM_ACTIONS_TIER2

    def action_space(self, params=None):
        from gymnax.environments.spaces import Discrete
        return Discrete(NUM_ACTIONS_TIER2)

    def observation_space(self, params=None):
        from gymnax.environments.spaces import Box
        map_h = self.static_params.map_height
        map_w = self.static_params.map_width
        max_m = self.static_params.max_monsters
        max_items = self.static_params.max_items
        # map_onehot + player_onehot + stair_onehot + monster_presence + seen + visible + inventory + stats
        obs_size = (map_h * map_w * NUM_TILE_TYPES +
                    map_h * map_w +
                    map_h * map_w +
                    map_h * map_w +
                    2 * map_h * map_w +
                    max_items +
                    14)
        return Box(0.0, 1.0, (obs_size,), dtype=jnp.float32)

    @property
    def name(self) -> str:
        return f"Minihax-{self._env_name}-v0"


# ============================================================================
# Individual Hazard environment classes
# ============================================================================

class LavaCrossingEnv(HazardEnv):
    def __init__(self):
        from Nethax.minihax.world_gen.lava_crossing import generate_lava_crossing
        super().__init__(
            "LavaCrossing",
            generate_lava_crossing,
            HazardStaticParams(map_height=10, map_width=15, max_monsters=1,
                               max_items=3, max_ground_items=3),
        )


class HideNSeekEnv(HazardEnv):
    def __init__(self):
        from Nethax.minihax.world_gen.hidenseek import generate_hidenseek
        super().__init__(
            "HideNSeek",
            generate_hidenseek,
            HazardStaticParams(map_height=10, map_width=12, max_monsters=2,
                               max_items=3, max_ground_items=3),
        )


class HideNSeekBigEnv(HazardEnv):
    def __init__(self):
        from Nethax.minihax.world_gen.hidenseek import generate_hidenseek_big
        super().__init__(
            "HideNSeekBig",
            generate_hidenseek_big,
            HazardStaticParams(map_height=16, map_width=16, max_monsters=2,
                               max_items=3, max_ground_items=3),
        )


class HideNSeekLavaEnv(HazardEnv):
    def __init__(self):
        from Nethax.minihax.world_gen.hidenseek import generate_hidenseek_lava
        super().__init__(
            "HideNSeekLava",
            generate_hidenseek_lava,
            HazardStaticParams(map_height=10, map_width=12, max_monsters=2,
                               max_items=3, max_ground_items=3),
        )


class HideNSeekMappedEnv(HazardEnv):
    def __init__(self):
        from Nethax.minihax.world_gen.hidenseek import generate_hidenseek_mapped
        super().__init__(
            "HideNSeekMapped",
            generate_hidenseek_mapped,
            HazardStaticParams(map_height=10, map_width=12, max_monsters=2,
                               max_items=3, max_ground_items=3),
        )


class QuestEasyEnv(HazardEnv):
    def __init__(self):
        from Nethax.minihax.world_gen.quest_easy import generate_quest_easy
        super().__init__(
            "QuestEasy",
            generate_quest_easy,
            HazardStaticParams(map_height=10, map_width=30, max_monsters=3,
                               max_items=3, max_ground_items=5),
        )


class QuestMediumEnv(HazardEnv):
    def __init__(self):
        from Nethax.minihax.world_gen.quest_medium import generate_quest_medium
        super().__init__(
            "QuestMedium",
            generate_quest_medium,
            HazardStaticParams(map_height=10, map_width=38, max_monsters=6,
                               max_items=3, max_ground_items=5),
        )


class LockedDoorEnv(HazardEnv):
    def __init__(self):
        from Nethax.minihax.world_gen.locked_door import generate_locked_door
        super().__init__(
            "LockedDoor",
            generate_locked_door,
            HazardStaticParams(map_height=10, map_width=15, max_monsters=1,
                               max_items=3, max_ground_items=3),
        )


class LockedDoorFixedEnv(HazardEnv):
    def __init__(self):
        from Nethax.minihax.world_gen.locked_door import generate_locked_door_fixed
        super().__init__(
            "LockedDoorFixed",
            generate_locked_door_fixed,
            HazardStaticParams(map_height=10, map_width=15, max_monsters=1,
                               max_items=3, max_ground_items=3),
        )


class TreasureDashEnv(HazardEnv):
    def __init__(self):
        from Nethax.minihax.world_gen.treasure_dash import generate_treasure_dash
        super().__init__(
            "TreasureDash",
            generate_treasure_dash,
            HazardStaticParams(map_height=5, map_width=75, max_monsters=1,
                               max_items=1, max_ground_items=24),
        )

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(max_timesteps=40)

    @property
    def num_actions(self) -> int:
        return 9  # 8 compass + search

    def action_space(self, params=None):
        from gymnax.environments.spaces import Discrete
        return Discrete(9)

    def step_env(self, rng, state, action, params):
        from Nethax.minihax.game_logic.treasure_dash import treasure_dash_step
        rng, rng_step = jax.random.split(rng)
        new_state, reward = treasure_dash_step(
            rng_step, state, action, params, self.static_params
        )
        done = new_state.terminal
        won = done & (new_state.timestep < params.max_timesteps)
        info = {
            "timestep": new_state.timestep,
            "won": won,
            "player_hp": new_state.player_stats.hp,
            "discount": self.discount(new_state, params),
        }
        return (
            lax.stop_gradient(self.get_obs(new_state)),
            lax.stop_gradient(new_state),
            reward,
            done,
            info,
        )
