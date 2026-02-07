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
