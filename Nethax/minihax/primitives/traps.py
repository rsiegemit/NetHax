"""Trap primitives for Tier 3."""
import jax
import jax.numpy as jnp


def check_traps(player_pos, traps, rng):
    """Check if player stepped on an active trap.

    Board trap (type_id=1): makes noise to wake nearby monsters, no damage.
    Pit trap (type_id=2): 1d6 damage.

    Args:
        player_pos: jnp.ndarray [2]
        traps: Traps struct (position, type_id, triggered, mask)
        rng: JAX PRNG key

    Returns:
        hp_delta: int -- HP change (negative = damage)
        new_traps: Traps with triggered flags updated
        noise: bool -- True if board trap triggered (wakes nearby monsters)
    """
    # Check each trap
    pos_r = traps.position[:, 0] == player_pos[0]
    pos_c = traps.position[:, 1] == player_pos[1]
    on_trap = pos_r & pos_c & traps.mask & jnp.logical_not(traps.triggered)

    any_trap = jnp.any(on_trap)
    trap_idx = jnp.argmax(on_trap)
    safe_idx = jnp.where(any_trap, trap_idx, 0)

    trap_type = traps.type_id[safe_idx]
    is_board = (trap_type == 1)  # Squeaky board = noise only
    is_pit = (trap_type == 2)    # Pit trap = 1d6 damage

    # Board trap: no damage, just noise to wake nearby monsters
    noise = any_trap & is_board

    # Pit trap: 1d6 damage
    rng, rng_pit = jax.random.split(rng)
    pit_dmg = jnp.where(any_trap & is_pit, jax.random.randint(rng_pit, (), 1, 7), 0)

    hp_delta = -pit_dmg  # Board does 0, pit does 1d6

    # Mark trap as triggered
    new_triggered = traps.triggered.at[safe_idx].set(
        jnp.where(any_trap, True, traps.triggered[safe_idx])
    )
    new_traps = traps.replace(triggered=new_triggered)

    return hp_delta, new_traps, noise
