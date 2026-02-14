"""Test that AC improves on level-up via do_melee_attack."""
import jax, jax.numpy as jnp
from Nethax.nethax_env import make_minihax_env_from_name
from Nethax.minihax.primitives.combat import do_melee_attack

env = make_minihax_env_from_name('Minihax-ZombieHorde-Symbolic-v0')
params = env.default_params
static_params = env.static_env_params

rng = jax.random.PRNGKey(0)
rng, reset_rng = jax.random.split(rng)
obs, state = env.reset(reset_rng, params)

print('Start: level=%d AC=%d HP=%d XP=%d' % (state.player_xp_level, state.player_ac, state.player_hp, state.player_xp))

# Repeatedly attack the first alive zombie until we get level-ups
prev_level = int(state.player_xp_level)
for i in range(200):
    # Find first alive hostile monster
    alive = state.monsters.mask
    target_idx = jnp.argmax(alive)
    target_pos = state.monsters.position[target_idx]

    rng, attack_rng = jax.random.split(rng)
    state = do_melee_attack(attack_rng, state, target_pos, static_params)

    cur_level = int(state.player_xp_level)
    if cur_level != prev_level:
        expected_ac = 10 - (cur_level - 1)
        actual_ac = int(state.player_ac)
        ok = "OK" if actual_ac == expected_ac else "MISMATCH"
        print('Attack %d: LEVEL %d->%d AC=%d expected=%d kills=%d %s' % (
            i, prev_level, cur_level, actual_ac, expected_ac, state.monsters_killed, ok))
        prev_level = cur_level

    if int(state.monsters_killed) >= 5:
        break

print('Final: level=%d AC=%d kills=%d XP=%d' % (
    state.player_xp_level, state.player_ac, state.monsters_killed, state.player_xp))
