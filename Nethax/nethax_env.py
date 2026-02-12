def make_nethax_env_from_name(name: str, auto_reset: bool):
    if auto_reset:
        if name == "Nethax-Symbolic-v1" or name == "Nethax-Symbolic-AutoReset-v1":
            from nethax.nethax.envs.nethax_symbolic_env import NethaxSymbolicEnv
            return NethaxSymbolicEnv()
    else:
        if name == "Nethax-Symbolic-v1":
            from nethax.nethax.envs.nethax_symbolic_env import NethaxSymbolicEnvNoAutoReset
            return NethaxSymbolicEnvNoAutoReset()

    raise ValueError(f"Unknown nethax environment: {name}")


def make_nethax_env_from_params(symbolic: bool, auto_reset: bool):
    if symbolic:
        if auto_reset:
            from nethax.nethax.envs.nethax_symbolic_env import NethaxSymbolicEnv
            return NethaxSymbolicEnv()
        else:
            from nethax.nethax.envs.nethax_symbolic_env import NethaxSymbolicEnvNoAutoReset
            return NethaxSymbolicEnvNoAutoReset()

    raise ValueError("Only symbolic observation mode is currently supported")


def make_minihax_env_from_name(name: str):
    if name == "Minihax-ZombieHorde-Symbolic-v0":
        from Nethax.minihax.envs.minihax_symbolic_env import MinihaxZombieHordeSymbolicEnv
        return MinihaxZombieHordeSymbolicEnv()
    elif name == "Minihax-ZombieHorde-Pixels-v0":
        from Nethax.minihax.envs.minihax_pixels_env import MinihaxZombieHordePixelsEnv
        return MinihaxZombieHordePixelsEnv()

    # Sokoban environments (Tier 4)
    elif name == "Minihax-Soko1a-v0":
        from Nethax.minihax.envs.sokoban_env import Soko1aEnv
        return Soko1aEnv()
    elif name == "Minihax-Soko1b-v0":
        from Nethax.minihax.envs.sokoban_env import Soko1bEnv
        return Soko1bEnv()
    elif name == "Minihax-Soko2a-v0":
        from Nethax.minihax.envs.sokoban_env import Soko2aEnv
        return Soko2aEnv()
    elif name == "Minihax-Soko2b-v0":
        from Nethax.minihax.envs.sokoban_env import Soko2bEnv
        return Soko2bEnv()
    elif name == "Minihax-Soko3a-v0":
        from Nethax.minihax.envs.sokoban_env import Soko3aEnv
        return Soko3aEnv()
    elif name == "Minihax-Soko3b-v0":
        from Nethax.minihax.envs.sokoban_env import Soko3bEnv
        return Soko3bEnv()
    elif name == "Minihax-Soko4a-v0":
        from Nethax.minihax.envs.sokoban_env import Soko4aEnv
        return Soko4aEnv()
    elif name == "Minihax-Soko4b-v0":
        from Nethax.minihax.envs.sokoban_env import Soko4bEnv
        return Soko4bEnv()

    # Navigation environments (Tier 1) - Maze
    elif name == "Minihax-Mazewalk-v0":
        from Nethax.minihax.envs.navigation_env import MazewalkEnv
        return MazewalkEnv()
    elif name == "Minihax-ExploreMazeEasy-v0":
        from Nethax.minihax.envs.navigation_env import ExploreMazeEasyEnv
        return ExploreMazeEasyEnv()
    elif name == "Minihax-ExploreMazeEasyPremapped-v0":
        from Nethax.minihax.envs.navigation_env import ExploreMazeEasyPremappedEnv
        return ExploreMazeEasyPremappedEnv()
    elif name == "Minihax-ExploreMazeHard-v0":
        from Nethax.minihax.envs.navigation_env import ExploreMazeHardEnv
        return ExploreMazeHardEnv()
    elif name == "Minihax-ExploreMazeHardPremapped-v0":
        from Nethax.minihax.envs.navigation_env import ExploreMazeHardPremappedEnv
        return ExploreMazeHardPremappedEnv()

    # Navigation environments (Tier 1) - Corridor
    elif name == "Minihax-Corridor2-v0":
        from Nethax.minihax.envs.navigation_env import Corridor2Env
        return Corridor2Env()
    elif name == "Minihax-Corridor3-v0":
        from Nethax.minihax.envs.navigation_env import Corridor3Env
        return Corridor3Env()
    elif name == "Minihax-Corridor5-v0":
        from Nethax.minihax.envs.navigation_env import Corridor5Env
        return Corridor5Env()
    elif name == "Minihax-Corridor8-v0":
        from Nethax.minihax.envs.navigation_env import Corridor8Env
        return Corridor8Env()
    elif name == "Minihax-Corridor10-v0":
        from Nethax.minihax.envs.navigation_env import Corridor10Env
        return Corridor10Env()

    # Hazard environments (Tier 2)
    elif name == "Minihax-LavaCrossing-v0":
        from Nethax.minihax.envs.hazard_env import LavaCrossingEnv
        return LavaCrossingEnv()
    elif name == "Minihax-HideNSeek-v0":
        from Nethax.minihax.envs.hazard_env import HideNSeekEnv
        return HideNSeekEnv()
    elif name == "Minihax-HideNSeekBig-v0":
        from Nethax.minihax.envs.hazard_env import HideNSeekBigEnv
        return HideNSeekBigEnv()
    elif name == "Minihax-HideNSeekLava-v0":
        from Nethax.minihax.envs.hazard_env import HideNSeekLavaEnv
        return HideNSeekLavaEnv()
    elif name == "Minihax-HideNSeekMapped-v0":
        from Nethax.minihax.envs.hazard_env import HideNSeekMappedEnv
        return HideNSeekMappedEnv()
    elif name == "Minihax-QuestEasy-v0":
        from Nethax.minihax.envs.hazard_env import QuestEasyEnv
        return QuestEasyEnv()
    elif name == "Minihax-QuestMedium-v0":
        from Nethax.minihax.envs.hazard_env import QuestMediumEnv
        return QuestMediumEnv()
    elif name == "Minihax-LockedDoor-v0":
        from Nethax.minihax.envs.hazard_env import LockedDoorEnv
        return LockedDoorEnv()
    elif name == "Minihax-LockedDoorFixed-v0":
        from Nethax.minihax.envs.hazard_env import LockedDoorFixedEnv
        return LockedDoorFixedEnv()
    elif name == "Minihax-TreasureDash-v0":
        from Nethax.minihax.envs.hazard_env import TreasureDashEnv
        return TreasureDashEnv()

    # Combat environments (Tier 3)
    elif name == "Minihax-Quest-v0":
        from Nethax.minihax.envs.combat_env import QuestEnv
        return QuestEnv()
    elif name == "Minihax-QuestHard-v0":
        from Nethax.minihax.envs.combat_env import QuestHardEnv
        return QuestHardEnv()
    elif name == "Minihax-KeyAndDoor-v0":
        from Nethax.minihax.envs.combat_env import KeyAndDoorEnv
        return KeyAndDoorEnv()
    elif name == "Minihax-KeyAndDoorTmp-v0":
        from Nethax.minihax.envs.combat_env import KeyAndDoorTmpEnv
        return KeyAndDoorTmpEnv()
    elif name == "Minihax-ClosedDoor-v0":
        from Nethax.minihax.envs.combat_env import ClosedDoorEnv
        return ClosedDoorEnv()
    elif name == "Minihax-Chest-v0":
        from Nethax.minihax.envs.combat_env import ChestEnv
        return ChestEnv()
    elif name == "Minihax-MementoEasy-v0":
        from Nethax.minihax.envs.combat_env import MementoEasyEnv
        return MementoEasyEnv()
    elif name == "Minihax-MementoShort-v0":
        from Nethax.minihax.envs.combat_env import MementoShortEnv
        return MementoShortEnv()
    elif name == "Minihax-MementoHard-v0":
        from Nethax.minihax.envs.combat_env import MementoHardEnv
        return MementoHardEnv()

    # ========================================================================
    # Pixel environments
    # ========================================================================

    # Navigation pixel environments (Tier 1) - Maze
    elif name == "Minihax-Mazewalk-Pixels-v0":
        from Nethax.minihax.envs.navigation_pixels_env import MazewalkPixelsEnv
        return MazewalkPixelsEnv()
    elif name == "Minihax-ExploreMazeEasy-Pixels-v0":
        from Nethax.minihax.envs.navigation_pixels_env import ExploreMazeEasyPixelsEnv
        return ExploreMazeEasyPixelsEnv()
    elif name == "Minihax-ExploreMazeEasyPremapped-Pixels-v0":
        from Nethax.minihax.envs.navigation_pixels_env import ExploreMazeEasyPremappedPixelsEnv
        return ExploreMazeEasyPremappedPixelsEnv()
    elif name == "Minihax-ExploreMazeHard-Pixels-v0":
        from Nethax.minihax.envs.navigation_pixels_env import ExploreMazeHardPixelsEnv
        return ExploreMazeHardPixelsEnv()
    elif name == "Minihax-ExploreMazeHardPremapped-Pixels-v0":
        from Nethax.minihax.envs.navigation_pixels_env import ExploreMazeHardPremappedPixelsEnv
        return ExploreMazeHardPremappedPixelsEnv()

    # Navigation pixel environments (Tier 1) - Corridor
    elif name == "Minihax-Corridor2-Pixels-v0":
        from Nethax.minihax.envs.navigation_pixels_env import Corridor2PixelsEnv
        return Corridor2PixelsEnv()
    elif name == "Minihax-Corridor3-Pixels-v0":
        from Nethax.minihax.envs.navigation_pixels_env import Corridor3PixelsEnv
        return Corridor3PixelsEnv()
    elif name == "Minihax-Corridor5-Pixels-v0":
        from Nethax.minihax.envs.navigation_pixels_env import Corridor5PixelsEnv
        return Corridor5PixelsEnv()
    elif name == "Minihax-Corridor8-Pixels-v0":
        from Nethax.minihax.envs.navigation_pixels_env import Corridor8PixelsEnv
        return Corridor8PixelsEnv()
    elif name == "Minihax-Corridor10-Pixels-v0":
        from Nethax.minihax.envs.navigation_pixels_env import Corridor10PixelsEnv
        return Corridor10PixelsEnv()

    # Hazard pixel environments (Tier 2)
    elif name == "Minihax-LavaCrossing-Pixels-v0":
        from Nethax.minihax.envs.hazard_pixels_env import LavaCrossingPixelsEnv
        return LavaCrossingPixelsEnv()
    elif name == "Minihax-HideNSeek-Pixels-v0":
        from Nethax.minihax.envs.hazard_pixels_env import HideNSeekPixelsEnv
        return HideNSeekPixelsEnv()
    elif name == "Minihax-HideNSeekBig-Pixels-v0":
        from Nethax.minihax.envs.hazard_pixels_env import HideNSeekBigPixelsEnv
        return HideNSeekBigPixelsEnv()
    elif name == "Minihax-HideNSeekLava-Pixels-v0":
        from Nethax.minihax.envs.hazard_pixels_env import HideNSeekLavaPixelsEnv
        return HideNSeekLavaPixelsEnv()
    elif name == "Minihax-HideNSeekMapped-Pixels-v0":
        from Nethax.minihax.envs.hazard_pixels_env import HideNSeekMappedPixelsEnv
        return HideNSeekMappedPixelsEnv()
    elif name == "Minihax-QuestEasy-Pixels-v0":
        from Nethax.minihax.envs.hazard_pixels_env import QuestEasyPixelsEnv
        return QuestEasyPixelsEnv()
    elif name == "Minihax-QuestMedium-Pixels-v0":
        from Nethax.minihax.envs.hazard_pixels_env import QuestMediumPixelsEnv
        return QuestMediumPixelsEnv()
    elif name == "Minihax-LockedDoor-Pixels-v0":
        from Nethax.minihax.envs.hazard_pixels_env import LockedDoorPixelsEnv
        return LockedDoorPixelsEnv()
    elif name == "Minihax-LockedDoorFixed-Pixels-v0":
        from Nethax.minihax.envs.hazard_pixels_env import LockedDoorFixedPixelsEnv
        return LockedDoorFixedPixelsEnv()
    elif name == "Minihax-TreasureDash-Pixels-v0":
        from Nethax.minihax.envs.hazard_pixels_env import TreasureDashPixelsEnv
        return TreasureDashPixelsEnv()

    # Combat pixel environments (Tier 3)
    elif name == "Minihax-Quest-Pixels-v0":
        from Nethax.minihax.envs.combat_pixels_env import QuestPixelsEnv
        return QuestPixelsEnv()
    elif name == "Minihax-QuestHard-Pixels-v0":
        from Nethax.minihax.envs.combat_pixels_env import QuestHardPixelsEnv
        return QuestHardPixelsEnv()
    elif name == "Minihax-KeyAndDoor-Pixels-v0":
        from Nethax.minihax.envs.combat_pixels_env import KeyAndDoorPixelsEnv
        return KeyAndDoorPixelsEnv()
    elif name == "Minihax-KeyAndDoorTmp-Pixels-v0":
        from Nethax.minihax.envs.combat_pixels_env import KeyAndDoorTmpPixelsEnv
        return KeyAndDoorTmpPixelsEnv()
    elif name == "Minihax-ClosedDoor-Pixels-v0":
        from Nethax.minihax.envs.combat_pixels_env import ClosedDoorPixelsEnv
        return ClosedDoorPixelsEnv()
    elif name == "Minihax-Chest-Pixels-v0":
        from Nethax.minihax.envs.combat_pixels_env import ChestPixelsEnv
        return ChestPixelsEnv()
    elif name == "Minihax-MementoEasy-Pixels-v0":
        from Nethax.minihax.envs.combat_pixels_env import MementoEasyPixelsEnv
        return MementoEasyPixelsEnv()
    elif name == "Minihax-MementoShort-Pixels-v0":
        from Nethax.minihax.envs.combat_pixels_env import MementoShortPixelsEnv
        return MementoShortPixelsEnv()
    elif name == "Minihax-MementoHard-Pixels-v0":
        from Nethax.minihax.envs.combat_pixels_env import MementoHardPixelsEnv
        return MementoHardPixelsEnv()

    # Sokoban pixel environments (Tier 4)
    elif name == "Minihax-Soko1a-Pixels-v0":
        from Nethax.minihax.envs.sokoban_pixels_env import Soko1aPixelsEnv
        return Soko1aPixelsEnv()
    elif name == "Minihax-Soko1b-Pixels-v0":
        from Nethax.minihax.envs.sokoban_pixels_env import Soko1bPixelsEnv
        return Soko1bPixelsEnv()
    elif name == "Minihax-Soko2a-Pixels-v0":
        from Nethax.minihax.envs.sokoban_pixels_env import Soko2aPixelsEnv
        return Soko2aPixelsEnv()
    elif name == "Minihax-Soko2b-Pixels-v0":
        from Nethax.minihax.envs.sokoban_pixels_env import Soko2bPixelsEnv
        return Soko2bPixelsEnv()
    elif name == "Minihax-Soko3a-Pixels-v0":
        from Nethax.minihax.envs.sokoban_pixels_env import Soko3aPixelsEnv
        return Soko3aPixelsEnv()
    elif name == "Minihax-Soko3b-Pixels-v0":
        from Nethax.minihax.envs.sokoban_pixels_env import Soko3bPixelsEnv
        return Soko3bPixelsEnv()
    elif name == "Minihax-Soko4a-Pixels-v0":
        from Nethax.minihax.envs.sokoban_pixels_env import Soko4aPixelsEnv
        return Soko4aPixelsEnv()
    elif name == "Minihax-Soko4b-Pixels-v0":
        from Nethax.minihax.envs.sokoban_pixels_env import Soko4bPixelsEnv
        return Soko4bPixelsEnv()

    # ========================================================================
    # NLE-style dict observation environments
    # ========================================================================

    # ZombieHorde NLE
    elif name == "Minihax-ZombieHorde-NLE-v0":
        from Nethax.minihax.envs.minihax_nle_env import MinihaxZombieHordeNLEEnv
        return MinihaxZombieHordeNLEEnv()

    # Navigation NLE environments (Tier 1) - Maze
    elif name == "Minihax-Mazewalk-NLE-v0":
        from Nethax.minihax.envs.navigation_nle_env import MazewalkNLEEnv
        return MazewalkNLEEnv()
    elif name == "Minihax-ExploreMazeEasy-NLE-v0":
        from Nethax.minihax.envs.navigation_nle_env import ExploreMazeEasyNLEEnv
        return ExploreMazeEasyNLEEnv()
    elif name == "Minihax-ExploreMazeEasyPremapped-NLE-v0":
        from Nethax.minihax.envs.navigation_nle_env import ExploreMazeEasyPremappedNLEEnv
        return ExploreMazeEasyPremappedNLEEnv()
    elif name == "Minihax-ExploreMazeHard-NLE-v0":
        from Nethax.minihax.envs.navigation_nle_env import ExploreMazeHardNLEEnv
        return ExploreMazeHardNLEEnv()
    elif name == "Minihax-ExploreMazeHardPremapped-NLE-v0":
        from Nethax.minihax.envs.navigation_nle_env import ExploreMazeHardPremappedNLEEnv
        return ExploreMazeHardPremappedNLEEnv()

    # Navigation NLE environments (Tier 1) - Corridor
    elif name == "Minihax-Corridor2-NLE-v0":
        from Nethax.minihax.envs.navigation_nle_env import Corridor2NLEEnv
        return Corridor2NLEEnv()
    elif name == "Minihax-Corridor3-NLE-v0":
        from Nethax.minihax.envs.navigation_nle_env import Corridor3NLEEnv
        return Corridor3NLEEnv()
    elif name == "Minihax-Corridor5-NLE-v0":
        from Nethax.minihax.envs.navigation_nle_env import Corridor5NLEEnv
        return Corridor5NLEEnv()
    elif name == "Minihax-Corridor8-NLE-v0":
        from Nethax.minihax.envs.navigation_nle_env import Corridor8NLEEnv
        return Corridor8NLEEnv()
    elif name == "Minihax-Corridor10-NLE-v0":
        from Nethax.minihax.envs.navigation_nle_env import Corridor10NLEEnv
        return Corridor10NLEEnv()

    # Hazard NLE environments (Tier 2)
    elif name == "Minihax-LavaCrossing-NLE-v0":
        from Nethax.minihax.envs.hazard_nle_env import LavaCrossingNLEEnv
        return LavaCrossingNLEEnv()
    elif name == "Minihax-HideNSeek-NLE-v0":
        from Nethax.minihax.envs.hazard_nle_env import HideNSeekNLEEnv
        return HideNSeekNLEEnv()
    elif name == "Minihax-HideNSeekBig-NLE-v0":
        from Nethax.minihax.envs.hazard_nle_env import HideNSeekBigNLEEnv
        return HideNSeekBigNLEEnv()
    elif name == "Minihax-HideNSeekLava-NLE-v0":
        from Nethax.minihax.envs.hazard_nle_env import HideNSeekLavaNLEEnv
        return HideNSeekLavaNLEEnv()
    elif name == "Minihax-HideNSeekMapped-NLE-v0":
        from Nethax.minihax.envs.hazard_nle_env import HideNSeekMappedNLEEnv
        return HideNSeekMappedNLEEnv()
    elif name == "Minihax-QuestEasy-NLE-v0":
        from Nethax.minihax.envs.hazard_nle_env import QuestEasyNLEEnv
        return QuestEasyNLEEnv()
    elif name == "Minihax-QuestMedium-NLE-v0":
        from Nethax.minihax.envs.hazard_nle_env import QuestMediumNLEEnv
        return QuestMediumNLEEnv()
    elif name == "Minihax-LockedDoor-NLE-v0":
        from Nethax.minihax.envs.hazard_nle_env import LockedDoorNLEEnv
        return LockedDoorNLEEnv()
    elif name == "Minihax-LockedDoorFixed-NLE-v0":
        from Nethax.minihax.envs.hazard_nle_env import LockedDoorFixedNLEEnv
        return LockedDoorFixedNLEEnv()
    elif name == "Minihax-TreasureDash-NLE-v0":
        from Nethax.minihax.envs.hazard_nle_env import TreasureDashNLEEnv
        return TreasureDashNLEEnv()

    # Combat NLE environments (Tier 3)
    elif name == "Minihax-Quest-NLE-v0":
        from Nethax.minihax.envs.combat_nle_env import QuestNLEEnv
        return QuestNLEEnv()
    elif name == "Minihax-QuestHard-NLE-v0":
        from Nethax.minihax.envs.combat_nle_env import QuestHardNLEEnv
        return QuestHardNLEEnv()
    elif name == "Minihax-KeyAndDoor-NLE-v0":
        from Nethax.minihax.envs.combat_nle_env import KeyAndDoorNLEEnv
        return KeyAndDoorNLEEnv()
    elif name == "Minihax-KeyAndDoorTmp-NLE-v0":
        from Nethax.minihax.envs.combat_nle_env import KeyAndDoorTmpNLEEnv
        return KeyAndDoorTmpNLEEnv()
    elif name == "Minihax-ClosedDoor-NLE-v0":
        from Nethax.minihax.envs.combat_nle_env import ClosedDoorNLEEnv
        return ClosedDoorNLEEnv()
    elif name == "Minihax-Chest-NLE-v0":
        from Nethax.minihax.envs.combat_nle_env import ChestNLEEnv
        return ChestNLEEnv()
    elif name == "Minihax-MementoEasy-NLE-v0":
        from Nethax.minihax.envs.combat_nle_env import MementoEasyNLEEnv
        return MementoEasyNLEEnv()
    elif name == "Minihax-MementoShort-NLE-v0":
        from Nethax.minihax.envs.combat_nle_env import MementoShortNLEEnv
        return MementoShortNLEEnv()
    elif name == "Minihax-MementoHard-NLE-v0":
        from Nethax.minihax.envs.combat_nle_env import MementoHardNLEEnv
        return MementoHardNLEEnv()

    # Sokoban NLE environments (Tier 4)
    elif name == "Minihax-Soko1a-NLE-v0":
        from Nethax.minihax.envs.sokoban_nle_env import Soko1aNLEEnv
        return Soko1aNLEEnv()
    elif name == "Minihax-Soko1b-NLE-v0":
        from Nethax.minihax.envs.sokoban_nle_env import Soko1bNLEEnv
        return Soko1bNLEEnv()
    elif name == "Minihax-Soko2a-NLE-v0":
        from Nethax.minihax.envs.sokoban_nle_env import Soko2aNLEEnv
        return Soko2aNLEEnv()
    elif name == "Minihax-Soko2b-NLE-v0":
        from Nethax.minihax.envs.sokoban_nle_env import Soko2bNLEEnv
        return Soko2bNLEEnv()
    elif name == "Minihax-Soko3a-NLE-v0":
        from Nethax.minihax.envs.sokoban_nle_env import Soko3aNLEEnv
        return Soko3aNLEEnv()
    elif name == "Minihax-Soko3b-NLE-v0":
        from Nethax.minihax.envs.sokoban_nle_env import Soko3bNLEEnv
        return Soko3bNLEEnv()
    elif name == "Minihax-Soko4a-NLE-v0":
        from Nethax.minihax.envs.sokoban_nle_env import Soko4aNLEEnv
        return Soko4aNLEEnv()
    elif name == "Minihax-Soko4b-NLE-v0":
        from Nethax.minihax.envs.sokoban_nle_env import Soko4bNLEEnv
        return Soko4bNLEEnv()

    raise ValueError(f"Unknown minihax environment: {name}")
