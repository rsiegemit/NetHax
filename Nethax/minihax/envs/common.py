"""Common utilities for minihax environments."""


def log_zombie_info(state, done):
    """Log zombie kills to info dict."""
    return {
        "monsters_killed": state.monsters_killed,
        "score": state.score,
        "returned_episode_returns": state.score * done,
        "returned_episode_lengths": state.timestep * done,
    }
