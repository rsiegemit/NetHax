"""Gymnax-compatible NLE-style environment for Combat (Tier 3).

Parametric env class for all Tier 3 combat environments (quest, memento,
key_and_door, closed_door, chest). Follows the pattern from combat_env.py
but uses NLE-style dict observations.

All Tier 3 envs share the same CombatStaticParams to enable a single JIT trace.
"""
import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple, Optional, Callable

from Nethax.environment_base.environment_bases import EnvironmentNoAutoReset
from Nethax.minihax.constants import NUM_ACTIONS_TIER3
from Nethax.minihax.game_logic.combat import combat_step
from Nethax.minihax.states import CombatState, EnvParams, CombatStaticParams


class CombatNLEEnv(EnvironmentNoAutoReset):
    """Base class for all Tier 3 combat NLE environments.

    Subclasses set a world_gen function that produces CombatState.
    Step logic uses combat_step (movement + bump attack + traps + items + doors
    + terrain + full monster AI + monster attacks + goal check).
    Observation uses render_nle_combat.
    """

    def __init__(self, env_name: str, world_gen_fn: Callable,
                 static_params: Optional[CombatStaticParams] = None,
                 crop_size: int = 9):
        super().__init__()
        self._env_name = env_name
        self._world_gen_fn = world_gen_fn
        if static_params is None:
            static_params = CombatStaticParams()
        self.static_params = static_params
        self.crop_size = crop_size

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
        won = new_state.terminal & (new_state.timestep < params.max_timesteps) & (new_state.player_hp > 0)

        done = new_state.terminal
        info = {
            "timestep": new_state.timestep,
            "won": won,
            "player_hp": new_state.player_hp,
            "score": new_state.score,
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

    def get_obs(self, state: CombatState) -> dict:
        from Nethax.minihax.nle_obs import render_nle_combat
        return render_nle_combat(state, self.static_params, self.crop_size)

    def is_terminal(self, state: CombatState, params: EnvParams) -> bool:
        return state.terminal

    @property
    def num_actions(self) -> int:
        return NUM_ACTIONS_TIER3

    def action_space(self, params=None):
        from gymnax.environments.spaces import Discrete
        return Discrete(NUM_ACTIONS_TIER3)

    def observation_space(self, params=None):
        from Nethax.minihax.nle_obs import nle_observation_space
        map_h = self.static_params.map_height
        map_w = self.static_params.map_width
        return nle_observation_space(map_h, map_w, self.crop_size)

    @property
    def name(self) -> str:
        return f"Minihax-{self._env_name}-NLE-v0"


# ============================================================================
# Individual Combat NLE environment classes
# ============================================================================

class QuestNLEEnv(CombatNLEEnv):
    def __init__(self):
        from Nethax.minihax.world_gen.quest import generate_quest
        super().__init__("Quest", generate_quest)


class QuestHardNLEEnv(CombatNLEEnv):
    def __init__(self):
        from Nethax.minihax.world_gen.quest_hard import generate_quest_hard
        super().__init__("QuestHard", generate_quest_hard,
                         static_params=CombatStaticParams(map_height=21))


class KeyAndDoorNLEEnv(CombatNLEEnv):
    def __init__(self):
        from Nethax.minihax.world_gen.key_and_door import generate_key_and_door
        super().__init__("KeyAndDoor", generate_key_and_door)


class KeyAndDoorTmpNLEEnv(CombatNLEEnv):
    def __init__(self):
        from Nethax.minihax.world_gen.key_and_door import generate_key_and_door_tmp
        super().__init__("KeyAndDoorTmp", generate_key_and_door_tmp)


class ClosedDoorNLEEnv(CombatNLEEnv):
    def __init__(self):
        from Nethax.minihax.world_gen.closed_door import generate_closed_door
        super().__init__("ClosedDoor", generate_closed_door)


class ChestNLEEnv(CombatNLEEnv):
    def __init__(self):
        from Nethax.minihax.world_gen.chest import generate_chest
        super().__init__("Chest", generate_chest)


class MementoEasyNLEEnv(CombatNLEEnv):
    def __init__(self):
        from Nethax.minihax.world_gen.memento import generate_memento_easy
        super().__init__("MementoEasy", generate_memento_easy,
                         static_params=CombatStaticParams(goal_type=1, goal_monster_idx=1))


class MementoShortNLEEnv(CombatNLEEnv):
    def __init__(self):
        from Nethax.minihax.world_gen.memento import generate_memento_short
        super().__init__("MementoShort", generate_memento_short,
                         static_params=CombatStaticParams(goal_type=1, goal_monster_idx=1))


class MementoHardNLEEnv(CombatNLEEnv):
    def __init__(self):
        from Nethax.minihax.world_gen.memento import generate_memento_hard
        super().__init__("MementoHard", generate_memento_hard,
                         static_params=CombatStaticParams(goal_type=1, goal_monster_idx=2))
