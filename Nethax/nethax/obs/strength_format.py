"""Vendor-canonical player strength formatting.

Vendor source: ``vendor/nethack/src/botl.c::get_strength_str`` lines 20-37.

Player strength is stored internally in the range ``[3, 125]``.  Vendor's
``get_strength_str`` (botl.c:20-37) emits, exactly:

* ``[3, 18]``    -> ``"3"`` .. ``"18"``       (``%-1d`` of st)
* ``[19, 117]``  -> ``"18/01"`` .. ``"18/99"`` (``18/%02d`` of ``st - 18``)
* ``118``        -> ``"18/**"``               (``STR18(100)``: maxed-out 18/100)
* ``[119, 125]`` -> ``"19"`` .. ``"25"``      (``%2d`` of ``st - 100``)

The two display ranges (3-17 and 19-25) overlap conceptually with the
``BL_STR25`` clamped value, but the ``BL_STR125`` internal value carries
the percentile-strength range ``18/XX`` so we format directly off the
125-scale.

This module ships a pure-Python helper plus a JIT-compatible byte renderer
used by ``nle_obs._build_status_row1``.
"""

from __future__ import annotations

import jax.numpy as jnp


__all__ = ["format_strength", "STRENGTH_FIELD_WIDTH", "render_strength_bytes"]


# Fixed column width allotted to the strength field in the JIT-rendered
# status row.  Vendor's ``"St:%s"`` is variable-width; we right-pad with
# spaces so downstream offsets stay deterministic across player_str ranges.
STRENGTH_FIELD_WIDTH: int = 5  # widest case is "18/00" / "18/**"


def format_strength(player_str: int) -> str:
    """Return the vendor-canonical ``get_strength_str`` rendering.

    Args:
        player_str: internal STR125 value (range [3, 125] in normal play).

    Returns:
        The same byte sequence that ``vendor botl.c::get_strength_str``
        would emit for that strength, e.g. ``"18/00"``, ``"18/**"``,
        ``"3"``, or ``"25"``.

    Cite: vendor/nethack/src/botl.c::get_strength_str (lines 20-37).
    """
    st = int(player_str)
    if st > 18:
        if st > 118:           # STR18(100) == 118 in vendor macro
            return f"{st - 100}"
        if st < 118:
            return f"18/{st - 18:02d}"
        return "18/**"
    return f"{st}"


# ---------------------------------------------------------------------------
# JIT-pure byte renderer used by the status row.
#
# Renders ``player_str`` to a fixed-width uint8[STRENGTH_FIELD_WIDTH] buffer
# right-padded with spaces.  Output layout examples (' '=space):
#   3     -> "3    "
#   17    -> "17   "
#   18    -> "18   "
#   19    -> "18/01"
#   50    -> "18/32"
#   117   -> "18/99"
#   118   -> "18/**"
#   119   -> "19   "
#   125   -> "25   "
# ---------------------------------------------------------------------------

_SPACE = jnp.uint8(ord(" "))
_DIGIT0 = jnp.uint8(ord("0"))
_SLASH = jnp.uint8(ord("/"))
_STAR = jnp.uint8(ord("*"))
_ONE = jnp.uint8(ord("1"))
_EIGHT = jnp.uint8(ord("8"))


def render_strength_bytes(player_str) -> jnp.ndarray:
    """JIT-pure 5-byte renderer for the vendor strength string.

    Args:
        player_str: scalar int (any int dtype) — the STR125 value.

    Returns:
        uint8[STRENGTH_FIELD_WIDTH] right-padded with ASCII spaces.
    """
    st = jnp.int32(player_str)

    # --- Path A: st <= 18 -> "%d" (1 or 2 digits) right-padded ---
    # Mid-range only ever holds st in [3..18] in normal play.  Vendor uses
    # "%-1d" which prints all digits with no padding, so we just emit the
    # digits LSB-first and right-pad with spaces.
    tens_a = jnp.where(st >= 10, _DIGIT0 + (st // 10).astype(jnp.uint8), _SPACE)
    ones_a = (_DIGIT0 + (st % 10).astype(jnp.uint8))
    # Shape: 1-digit -> "<d>    ", 2-digit -> "<dd>   "
    one_digit_a = jnp.stack([ones_a, _SPACE, _SPACE, _SPACE, _SPACE])
    two_digit_a = jnp.stack([tens_a, ones_a, _SPACE, _SPACE, _SPACE])
    bytes_a = jnp.where(st >= 10, two_digit_a, one_digit_a)

    # --- Path B: 18 < st < 118 -> "18/XX" ---
    over18 = jnp.maximum(st - 18, jnp.int32(0))                # 0..99
    pct_tens = _DIGIT0 + (over18 // 10).astype(jnp.uint8)
    pct_ones = _DIGIT0 + (over18 % 10).astype(jnp.uint8)
    bytes_b = jnp.stack([_ONE, _EIGHT, _SLASH, pct_tens, pct_ones])

    # --- Path C: st == 118 -> "18/**" ---
    bytes_c = jnp.stack([_ONE, _EIGHT, _SLASH, _STAR, _STAR])

    # --- Path D: st > 118 -> "%d" of (st - 100), right-padded ---
    high = jnp.maximum(st - 100, jnp.int32(0))                  # 19..25 typical
    high_tens = jnp.where(high >= 10,
                          _DIGIT0 + (high // 10).astype(jnp.uint8),
                          _SPACE)
    high_ones = _DIGIT0 + (high % 10).astype(jnp.uint8)
    one_digit_d = jnp.stack([high_ones, _SPACE, _SPACE, _SPACE, _SPACE])
    two_digit_d = jnp.stack([high_tens, high_ones, _SPACE, _SPACE, _SPACE])
    bytes_d = jnp.where(high >= 10, two_digit_d, one_digit_d)

    # Compose: pick path by st bucket.
    is_low = st <= jnp.int32(18)
    is_max18 = st == jnp.int32(118)
    is_high = st > jnp.int32(118)

    result = jnp.where(is_low, bytes_a,
              jnp.where(is_max18, bytes_c,
               jnp.where(is_high, bytes_d, bytes_b)))
    return result.astype(jnp.uint8)
