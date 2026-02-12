"""Gymnax-compatible pixel environment for Combat (Tier 3).

Pixel-rendered versions of all Tier 3 combat environments (quest, memento,
key_and_door, closed_door, chest).
"""
import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple, Optional, Callable

from Nethax.environment_base.environment_bases import EnvironmentNoAutoReset
from Nethax.minihax.constants import NUM_ACTIONS_TIER3, TILE_SIZE
from Nethax.minihax.game_logic.combat import combat_step
from Nethax.minihax.states import CombatState, EnvParams, CombatStaticParams


class CombatPixelsEnv(EnvironmentNoAutoReset):
    """Base class for all Tier 3 combat pixel environments.

    Subclasses set a world_gen function that produces CombatState.
    Step logic uses combat_step (movement + bump attack + traps + items + doors
    + terrain + full monster AI + monster attacks + goal check).
    Observation uses render_pixels_with_monsters for pixel rendering.
    """

    def __init__(self, env_name: str, world_gen_fn: Callable,
                 static_params: Optional[CombatStaticParams] = None):
        super().__init__()
        self._env_name = env_name
        self._world_gen_fn = world_gen_fn
        if static_params is None:
            static_params = CombatStaticParams()
        self.static_params = static_params

        # Load tiles eagerly — must be concrete before JIT tracing
        from Nethax.tiles.renderer import load_tiles
        self._tiles_array = load_tiles()

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

    def get_obs(self, state: CombatState, wizard_mode=False) -> jax.Array:
        from Nethax.minihax.pixel_renderer import render_pixels_with_monsters
        traps = state.traps if wizard_mode else None
        return render_pixels_with_monsters(
            state, self.static_params, self._tiles_array, self.static_params.max_monsters,
            wizard_mode=wizard_mode, traps=traps,
        )

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
        h = self.static_params.map_height * TILE_SIZE
        w = self.static_params.map_width * TILE_SIZE
        return Box(0, 255, (h, w, 3), dtype=jnp.uint8)

    @property
    def name(self) -> str:
        return f"Minihax-{self._env_name}-Pixels-v0"


# ============================================================================
# Individual Combat pixel environment classes
# ============================================================================

class QuestPixelsEnv(CombatPixelsEnv):
    def __init__(self):
        from Nethax.minihax.world_gen.quest import generate_quest
        super().__init__("Quest", generate_quest)


class QuestHardPixelsEnv(CombatPixelsEnv):
    def __init__(self):
        from Nethax.minihax.world_gen.quest_hard import generate_quest_hard
        from Nethax.minihax.states import CombatStaticParams
        super().__init__("QuestHard", generate_quest_hard,
                         static_params=CombatStaticParams(map_height=21))


class KeyAndDoorPixelsEnv(CombatPixelsEnv):
    def __init__(self):
        from Nethax.minihax.world_gen.key_and_door import generate_key_and_door
        super().__init__("KeyAndDoor", generate_key_and_door)


class KeyAndDoorTmpPixelsEnv(CombatPixelsEnv):
    def __init__(self):
        from Nethax.minihax.world_gen.key_and_door import generate_key_and_door_tmp
        super().__init__("KeyAndDoorTmp", generate_key_and_door_tmp)


class ClosedDoorPixelsEnv(CombatPixelsEnv):
    def __init__(self):
        from Nethax.minihax.world_gen.closed_door import generate_closed_door
        super().__init__("ClosedDoor", generate_closed_door)


class ChestPixelsEnv(CombatPixelsEnv):
    def __init__(self):
        from Nethax.minihax.world_gen.chest import generate_chest
        super().__init__("Chest", generate_chest)


class MementoEasyPixelsEnv(CombatPixelsEnv):
    def __init__(self):
        from Nethax.minihax.world_gen.memento import generate_memento_easy
        super().__init__("MementoEasy", generate_memento_easy,
                         static_params=CombatStaticParams(goal_type=1, goal_monster_idx=1))


class MementoShortPixelsEnv(CombatPixelsEnv):
    def __init__(self):
        from Nethax.minihax.world_gen.memento import generate_memento_short
        super().__init__("MementoShort", generate_memento_short,
                         static_params=CombatStaticParams(goal_type=1, goal_monster_idx=1))


class MementoHardPixelsEnv(CombatPixelsEnv):
    def __init__(self):
        from Nethax.minihax.world_gen.memento import generate_memento_hard
        super().__init__("MementoHard", generate_memento_hard,
                         static_params=CombatStaticParams(goal_type=1, goal_monster_idx=2))
