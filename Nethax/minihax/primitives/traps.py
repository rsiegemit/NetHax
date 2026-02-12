"""Trap primitives for Tier 3."""
import jax.numpy as jnp


def check_traps(player_pos, traps):
    """Check if player stepped on an active trap.

    Board trap (type_id=1): instant death -- set HP to 0 via massive damage.
    Pit trap (type_id=2): reserved for future use.

    Args:
        player_pos: jnp.ndarray [2]
        traps: Traps struct (position, type_id, triggered, mask)

    Returns:
        hp_delta: int -- HP change (negative = damage, -9999 for board trap)
        new_traps: Traps with triggered flags updated
    """
    # Check each trap
    pos_r = traps.position[:, 0] == player_pos[0]
    pos_c = traps.position[:, 1] == player_pos[1]
    on_trap = pos_r & pos_c & traps.mask & jnp.logical_not(traps.triggered)

    any_trap = jnp.any(on_trap)
    trap_idx = jnp.argmax(on_trap)
    safe_idx = jnp.where(any_trap, trap_idx, 0)

    trap_type = traps.type_id[safe_idx]
    is_board = (trap_type == 1)  # Board trap = instant death

    # Board trap: massive damage to guarantee death
    hp_delta = jnp.where(any_trap & is_board, -9999, 0)

    # Mark trap as triggered
    new_triggered = traps.triggered.at[safe_idx].set(
        jnp.where(any_trap, True, traps.triggered[safe_idx])
    )
    new_traps = traps.replace(triggered=new_triggered)

    return hp_delta, new_traps
