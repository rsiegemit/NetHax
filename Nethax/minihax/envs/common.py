"""Common utilities for minihax environments."""


def log_zombie_info(state, done):
    """Log zombie kills to info dict."""
    return {
        "monsters_killed": state.player_stats.monsters_killed,
        "score": state.player_stats.score,
        "returned_episode_returns": state.player_stats.score * done,
        "returned_episode_lengths": state.timestep * done,
    }
