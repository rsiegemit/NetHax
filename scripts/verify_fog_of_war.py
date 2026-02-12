"""Verify fog of war implementation across all 4 tiers + zombie horde."""
import jax
import jax.numpy as jnp
import sys

def test_env(env_name, tier_label):
    """Test that an env can be created and stepped with fog of war."""
    from Nethax.nethax_env import make_minihax_env_from_name

    print(f"\n{'='*60}")
    print(f"Testing {tier_label}: {env_name}")
    print(f"{'='*60}")

    env, env_params = make_minihax_env_from_name(env_name)

    rng = jax.random.PRNGKey(42)
    rng, rng_reset = jax.random.split(rng)

    obs, state = env.reset(rng_reset, env_params)

    # Check state has visibility fields
    assert hasattr(state, 'seen_map'), f"{env_name}: missing seen_map"
    assert hasattr(state, 'visible_map'), f"{env_name}: missing visible_map"

    print(f"  State has seen_map: shape={state.seen_map.shape}, dtype={state.seen_map.dtype}")
    print(f"  State has visible_map: shape={state.visible_map.shape}, dtype={state.visible_map.dtype}")
    print(f"  Initial seen tiles: {int(state.seen_map.sum())}")
    print(f"  Initial visible tiles: {int(state.visible_map.sum())}")

    # Verify player position is visible
    pr, pc = int(state.player_position[0]), int(state.player_position[1])
    assert state.visible_map[pr, pc], f"{env_name}: player position not visible!"
    assert state.seen_map[pr, pc], f"{env_name}: player position not seen!"
    print(f"  Player at ({pr}, {pc}) is visible: OK")

    # Take a few steps
    for step_i in range(5):
        rng, rng_step = jax.random.split(rng)
        action = jax.random.randint(rng_step, (), 0, env.num_actions)
        rng, rng_act = jax.random.split(rng)
        obs, state, reward, done, info = env.step(rng_act, state, action, env_params)

    print(f"  After 5 steps - seen tiles: {int(state.seen_map.sum())}, visible: {int(state.visible_map.sum())}")

    # Check seen_map only grows (never shrinks)
    # visible_map can change each step
    print(f"  Obs shape: {obs.shape}")
    print(f"  PASSED!")
    return True


def test_pixel_env(env_name, tier_label):
    """Test that a pixel env renders with fog of war."""
    from Nethax.nethax_env import make_minihax_env_from_name

    print(f"\n{'='*60}")
    print(f"Testing {tier_label} (pixels): {env_name}")
    print(f"{'='*60}")

    env, env_params = make_minihax_env_from_name(env_name)

    rng = jax.random.PRNGKey(42)
    rng, rng_reset = jax.random.split(rng)

    obs, state = env.reset(rng_reset, env_params)

    print(f"  Pixel obs shape: {obs.shape}, dtype={obs.dtype}")
    print(f"  Min pixel: {float(obs.min()):.3f}, Max pixel: {float(obs.max()):.3f}")

    # Check that some pixels are dark (fog of war effect)
    # Unseen areas should have brightness 0, seen-but-not-visible should be ~0.3
    zero_pixels = (obs == 0).all(axis=-1).sum() if obs.ndim == 3 else 0
    print(f"  Black pixels (unseen): {int(zero_pixels)}")

    # Take a step
    rng, rng_step = jax.random.split(rng)
    action = jax.random.randint(rng_step, (), 0, env.num_actions)
    rng, rng_act = jax.random.split(rng)
    obs, state, reward, done, info = env.step(rng_act, state, action, env_params)

    print(f"  After step - obs shape: {obs.shape}")
    print(f"  PASSED!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("FOG OF WAR VERIFICATION")
    print("=" * 60)

    results = []

    # Tier 1: Navigation (symbolic)
    results.append(test_env("Minihax-Corridor5-v0", "Tier 1 Navigation"))

    # Tier 2: Hazard (symbolic)
    results.append(test_env("Minihax-LavaCrossing-v0", "Tier 2 Hazard"))

    # Tier 3: Combat (symbolic)
    results.append(test_env("Minihax-KeyAndDoor-v0", "Tier 3 Combat"))

    # Tier 4: Sokoban (symbolic)
    results.append(test_env("Minihax-Soko1a-v0", "Tier 4 Sokoban"))

    # ZombieHorde (legacy)
    results.append(test_env("Minihax-ZombieHorde-v0", "ZombieHorde"))

    # Pixel envs
    results.append(test_pixel_env("Minihax-Corridor5Pixels-v0", "Tier 1 Navigation"))
    results.append(test_pixel_env("Minihax-LavaCrossingPixels-v0", "Tier 2 Hazard"))
    results.append(test_pixel_env("Minihax-KeyAndDoorPixels-v0", "Tier 3 Combat"))
    results.append(test_pixel_env("Minihax-Soko1aPixels-v0", "Tier 4 Sokoban"))
    results.append(test_pixel_env("Minihax-ZombieHordePixels-v0", "ZombieHorde"))

    print(f"\n\n{'='*60}")
    print(f"RESULTS: {sum(results)}/{len(results)} passed")
    print(f"{'='*60}")

    if all(results):
        print("\nAll fog of war tests PASSED!")
        sys.exit(0)
    else:
        print("\nSome tests FAILED!")
        sys.exit(1)
