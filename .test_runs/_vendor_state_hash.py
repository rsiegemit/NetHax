"""Hash the vendor-mode (byte-parity) reset state — for current-vs-pristine
byte-neutrality check. Vendor path must be unchanged by the vec/gate work."""
import hashlib, numpy as np, jax
from Nethax.nethax.parity_mode import set_parity_mode, ParityMode, use_vendor_rng
set_parity_mode(ParityMode.NLE_BYTEPARITY)
from Nethax.minihax.minihax_env import MinihaxEnv
assert use_vendor_rng(), "must be vendor mode"
env = MinihaxEnv("MiniHack-Room-Monster-15x15-v0")
h = hashlib.sha256()
for seed in range(3):
    s,_ = env.reset(jax.random.key(seed)); jax.block_until_ready(s)
    for path, a in jax.tree_util.tree_leaves_with_path(s):
        if jax.numpy.issubdtype(a.dtype, jax.dtypes.prng_key):
            a = jax.random.key_data(a)
        h.update(np.asarray(a).tobytes())
print("VENDOR_RESET_HASH", h.hexdigest()[:32])
