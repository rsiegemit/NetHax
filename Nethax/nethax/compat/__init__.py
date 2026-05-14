"""NLE compatibility shims for Nethax.

Wave 5 Phase 5 — exposes wrappers that adapt the JAX-native ``NethaxEnv``
to be a drop-in replacement for ``nle.nethack.Nethack`` and related
classes from the original Facebook NLE C bindings.
"""
from Nethax.nethax.compat.nle_shim import NLECompat

__all__ = ["NLECompat"]
