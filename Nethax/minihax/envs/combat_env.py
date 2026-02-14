"""Gymnax-compatible environment for Combat (Tier 3).

Parametric env class for all Tier 3 combat environments (quest, memento,
key_and_door, closed_door, chest). Follows the pattern from hazard_env.py.

All Tier 3 envs share the same CombatStaticParams to enable a single JIT trace.
"""
import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple, Optional, Callable

from Nethax.environment_base.environment_bases import EnvironmentNoAutoReset
from Nethax.minihax.constants import NUM_ACTIONS_TIER3, NUM_TILE_TYPES
from Nethax.minihax.game_logic.combat import combat_step
from Nethax.minihax.states import CombatState, EnvParams, CombatStaticParams


class CombatEnv(EnvironmentNoAutoReset):
    """Base class for all Tier 3 combat environments.

    Subclasses set a world_gen function that produces CombatState.
    Step logic uses combat_step (movement + bump attack + traps + items + doors
    + terrain + full monster AI + monster attacks + goal check).
    Observation uses render_combat_symbolic.
    """

    def __init__(self, env_name: str, world_gen_fn: Callable,
                 static_params: Optional[CombatStaticParams] = None):
        super().__init__()
        self._env_name = env_name
        self._world_gen_fn = world_gen_fn
        if static_params is None:
            static_params = CombatStaticParams()
        self.static_params = static_params

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
        self, rng: jax.Array, state: CombatState, action: int, params: EnvParams
    ) -> Tuple[jax.Array, CombatState, float, bool, dict]:
        rng, rng_step = jax.random.split(rng)

        new_state, reward = combat_step(
            rng_step, state, action, params, self.static_params
        )

        # Win = terminal and not timed out and not dead
        won = new_state.terminal & (new_state.timestep < params.max_timesteps) & (new_state.player_stats.hp > 0)

        done = new_state.terminal
        info = {
            "timestep": new_state.timestep,
            "won": won,
            "player_hp": new_state.player_stats.hp,
            "score": new_state.player_stats.score,
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
    ) -> Tuple[jax.Array, CombatState]:
        rng, rng_gen = jax.random.split(rng)
        state = self._world_gen_fn(rng_gen, params, self.static_params)
        return self.get_obs(state), state

    def get_obs(self, state: CombatState) -> jax.Array:
        from Nethax.minihax.renderer import render_combat_symbolic
        return render_combat_symbolic(state, self.static_params)

    def is_terminal(self, state: CombatState, params: EnvParams) -> bool:
        return state.terminal

    @property
    def num_actions(self) -> int:
        return NUM_ACTIONS_TIER3

    def action_space(self, params=None):
        from gymnax.environments.spaces import Discrete
        return Discrete(NUM_ACTIONS_TIER3)

    def observation_space(self, params=None):
        from gymnax.environments.spaces import Box
        from Nethax.minihax.constants import NUM_ITEM_TYPES
        map_h = self.static_params.map_height
        map_w = self.static_params.map_width
        max_items = self.static_params.max_items
        # map_onehot + player + stair + monsters + traps + seen + visible + inventory + stats
        obs_size = (map_h * map_w * NUM_TILE_TYPES +
                    map_h * map_w +      # player
                    map_h * map_w +      # stair
                    map_h * map_w +      # monsters
                    map_h * map_w +      # traps
                    2 * map_h * map_w +  # seen_map + visible_map
                    max_items +          # inventory
                    17)                  # stats
        return Box(0.0, 1.0, (obs_size,), dtype=jnp.float32)

    @property
    def name(self) -> str:
        return f"Minihax-{self._env_name}-v0"


# ============================================================================
# Individual Combat environment classes
# ============================================================================

class QuestEnv(CombatEnv):
    def __init__(self):
        from Nethax.minihax.world_gen.quest import generate_quest
        super().__init__("Quest", generate_quest)


class QuestHardEnv(CombatEnv):
    def __init__(self):
        from Nethax.minihax.world_gen.quest_hard import generate_quest_hard
        super().__init__("QuestHard", generate_quest_hard,
                         static_params=CombatStaticParams(map_height=21))


class KeyAndDoorEnv(CombatEnv):
    def __init__(self):
        from Nethax.minihax.world_gen.key_and_door import generate_key_and_door
        super().__init__("KeyAndDoor", generate_key_and_door)


class KeyAndDoorTmpEnv(CombatEnv):
    def __init__(self):
        from Nethax.minihax.world_gen.key_and_door import generate_key_and_door_tmp
        super().__init__("KeyAndDoorTmp", generate_key_and_door_tmp)


class ClosedDoorEnv(CombatEnv):
    def __init__(self):
        from Nethax.minihax.world_gen.closed_door import generate_closed_door
        super().__init__("ClosedDoor", generate_closed_door)


class ChestEnv(CombatEnv):
    def __init__(self):
        from Nethax.minihax.world_gen.chest import generate_chest
        super().__init__("Chest", generate_chest)


class MementoEasyEnv(CombatEnv):
    def __init__(self):
        from Nethax.minihax.world_gen.memento import generate_memento_easy
        super().__init__("MementoEasy", generate_memento_easy,
                         static_params=CombatStaticParams(goal_type=1, goal_monster_idx=1))


class MementoShortEnv(CombatEnv):
    def __init__(self):
        from Nethax.minihax.world_gen.memento import generate_memento_short
        super().__init__("MementoShort", generate_memento_short,
                         static_params=CombatStaticParams(goal_type=1, goal_monster_idx=1))


class MementoHardEnv(CombatEnv):
    def __init__(self):
        from Nethax.minihax.world_gen.memento import generate_memento_hard
        super().__init__("MementoHard", generate_memento_hard,
                         static_params=CombatStaticParams(goal_type=1, goal_monster_idx=2))
