"""Flax SymbolicGlyphNet encoder for MiniHax NLE-style observations.

Port of SOL's PyTorch SymbolicGlyphNet with MiniHax adaptations:
- Compact glyph space (50 glyphs, not NLE's 5991)
- Pre-computed glyphs_crop (no Crop module)
- No message encoder (messages are zeros)
- Per-tier blstats normalization
"""
import jax
import jax.numpy as jnp
from flax import linen as nn

from Nethax.minihax.nle_obs import NUM_GLYPHS, DEFAULT_CROP_SIZE, BLSTATS_SIZE


class SymbolicGlyphNet(nn.Module):
    """Encodes NLE-style dict observations into a flat feature vector.

    Architecture matches SOL's SymbolicGlyphNet:
      1. Glyph crop embedding -> Conv2d pipeline -> flatten
      2. BLStats normalize -> clip[-5,5] -> MLP -> concat raw
      3. Optional prev_action one-hot

    Operates on single observations (no batch dim). Use jax.vmap for batching.

    Args:
        num_glyphs: Size of glyph vocabulary (default 50 for MiniHax).
        glyph_edim: Glyph embedding dimension (default 64, same as SOL).
        crop_dim: Crop window size (default 9, MiniHack default).
        num_actions: Action space size for prev_action one-hot.
        use_prev_action: Whether to include prev_action in output.
        blstats_norm: Tuple of 27 floats for per-element blstats normalization.
    """
    num_glyphs: int = NUM_GLYPHS
    glyph_edim: int = 64
    crop_dim: int = DEFAULT_CROP_SIZE
    num_actions: int = 8
    use_prev_action: bool = True
    blstats_norm: tuple = (0.0,) * BLSTATS_SIZE

    @nn.compact
    def __call__(self, obs_dict):
        edim = self.glyph_edim
        k_dim = 2 * edim

        # --- Crop glyph embedding ---
        # glyphs_crop: (crop_dim, crop_dim) uint8
        glyphs_crop = obs_dict["glyphs_crop"]
        glyph_embed = nn.Embed(
            num_embeddings=self.num_glyphs,
            features=edim,
        )(glyphs_crop)  # (crop_dim, crop_dim, edim)

        # Conv pipeline (Flax Conv expects (H, W, C), use VALID padding to match PyTorch)
        x = nn.Conv(features=k_dim, kernel_size=(3, 3), strides=(2, 2), padding='VALID')(glyph_embed)
        x = nn.elu(x)
        x = nn.Conv(features=2 * k_dim, kernel_size=(3, 3), strides=(2, 2), padding='VALID')(x)
        x = nn.elu(x)
        crop_flat = x.reshape(-1)

        # --- BLStats encoder ---
        # Matches SOL: normalize -> clip -> MLP(27->128->128) -> concat raw
        blstats = obs_dict["blstats"].astype(jnp.float32)
        norm_vec = jnp.array(self.blstats_norm)
        norm_bls = jnp.clip(blstats * norm_vec, -5.0, 5.0)

        bl_h = nn.Dense(features=128)(norm_bls)
        bl_h = nn.elu(bl_h)
        bl_h = nn.Dense(features=128)(bl_h)
        bl_h = nn.elu(bl_h)
        bl_out = jnp.concatenate([bl_h, norm_bls])  # (155,)

        # --- No message encoder (messages are all zeros in MiniHax) ---

        # --- Combine encodings ---
        encodings = [crop_flat, bl_out]

        if self.use_prev_action:
            prev_action = obs_dict["prev_actions"]
            encodings.append(jax.nn.one_hot(prev_action, self.num_actions))

        return jnp.concatenate(encodings)

    def output_size(self) -> int:
        """Compute output feature vector size."""
        # crop_conv output: with crop_dim=9, kernel=3, stride=2
        #   after conv1: (9-3)//2 + 1 = 4 -> (4, 4, 2*edim)
        #   after conv2: (4-3)//2 + 1 = 1 -> (1, 1, 4*edim)
        crop_out = 4 * self.glyph_edim
        bl_out = 128 + BLSTATS_SIZE  # MLP output + raw normalized
        total = crop_out + bl_out
        if self.use_prev_action:
            total += self.num_actions
        return total
