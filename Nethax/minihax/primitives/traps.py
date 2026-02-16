"""Trap primitives for Tier 3."""
import jax
import jax.numpy as jnp


def check_traps(player_pos, traps, rng):
    """Check if player stepped on an active trap.

    Board trap (type_id=1): makes noise to wake nearby monsters, no damage.
    Pit trap (type_id=2): 1d6 damage.

    Args:
        player_pos: jnp.ndarray [2]
        traps: Traps struct (position, type_id, triggered, hidden, mask)
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

    # Also reveal hidden trap on trigger
    new_hidden = new_traps.hidden.at[safe_idx].set(
        jnp.where(any_trap, False, new_traps.hidden[safe_idx])
    )
    new_traps = new_traps.replace(hidden=new_hidden)

    return hp_delta, new_traps, noise


def search_traps(rng, player_pos, traps):
    """SEARCH action: attempt to reveal hidden traps within Chebyshev distance 1.

    Each hidden trap in range has 1/5 chance of being revealed.

    Args:
        rng: JAX PRNG key
        player_pos: jnp.ndarray [2]
        traps: Traps struct (with hidden field)

    Returns:
        new_traps: Traps with some hidden flags cleared
    """
    max_traps = traps.position.shape[0]
    rngs = jax.random.split(rng, max_traps)

    # For each trap: check range, check hidden, roll 1/5
    dr = jnp.abs(traps.position[:, 0] - player_pos[0])
    dc = jnp.abs(traps.position[:, 1] - player_pos[1])
    in_range = (dr <= 1) & (dc <= 1)
    can_reveal = traps.mask & traps.hidden & in_range

    # Roll 1/5 chance per trap
    rolls = jax.vmap(lambda r: jax.random.randint(r, (), 0, 5))(rngs)
    revealed = can_reveal & (rolls == 0)

    new_hidden = jnp.where(revealed, False, traps.hidden)
    return traps.replace(hidden=new_hidden)
