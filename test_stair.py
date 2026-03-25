"""Test whether stepping on stair ends the episode."""
from Nethax.nethax_env import make_minihax_env_from_name
from Nethax.minihax.states import EnvParams
import jax
import jax.numpy as jnp

env = make_minihax_env_from_name('Minihax-Corridor2-v0')
params = env.default_params
print(f"auto_descend={params.auto_descend}")

rng = jax.random.PRNGKey(42)
obs, state = env.reset(rng, params)
print(f"Player: {state.player_position}")
print(f"Stair:  {state.downstair_position}")

# Manually place player next to stair and step onto it
# First, find direction from player to stair
dr = int(state.downstair_position[0] - state.player_position[0])
dc = int(state.downstair_position[1] - state.player_position[1])
print(f"Delta to stair: dr={dr}, dc={dc}")

# Teleport: replace player_position to be adjacent to stair
adj_pos = jnp.array([state.downstair_position[0] - 1, state.downstair_position[1]], dtype=jnp.int32)
state_adj = state.replace(player_position=adj_pos)
print(f"Placed player at {adj_pos} (one north of stair)")

# Move south (action 4) to step onto stair
obs2, state2, reward, done, info = env.step(jax.random.PRNGKey(0), state_adj, 4, params)
print(f"After moving south onto stair:")
print(f"  Player pos: {state2.player_position}")
print(f"  Terminal: {state2.terminal}")
print(f"  Done: {done}")
print(f"  Reward: {float(reward):.4f}")
print(f"  Won: {info['won']}")
