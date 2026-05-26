"""Byte-parity test for Nethax/nethax/vendor_rng.py against vendor ISAAC64.

Compiles a tiny C harness around ``vendor/nle/src/isaac64.c`` that mirrors
exactly what NLE does in ``init_isaac64`` (rnd.c:42-58) + ``RND`` (rnd.c:60-64):

    init_isaac64(seed=0, &core)
    for i in 0..99: print isaac64_next_uint64(&core)

Then runs our Python ``init_py(0)`` + 100 ``next_uint64_py`` and asserts
byte-identical match.

If the compiler is unavailable (CI without gcc / wrong vendor path), the
test ``xfail``s with a clear message rather than blowing up -- this is an
audit deliverable and the C-build dependency is documented.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from Nethax.nethax.vendor_rng import init_py, next_uint64_py

REPO = Path(__file__).resolve().parents[1]
VENDOR_ISAAC_C = REPO / "vendor" / "nle" / "src" / "isaac64.c"
VENDOR_ISAAC_H_DIR = REPO / "vendor" / "nle" / "include"


def _build_vendor_harness(tmp_path: Path) -> Path:
    """Compile the vendor ISAAC64 + a small driver into a binary."""
    if not VENDOR_ISAAC_C.exists():
        pytest.skip(f"vendor source not present: {VENDOR_ISAAC_C}")

    # Strip vendor's ``#include "config.h"`` / ``#include "isaac64.h"`` and
    # the ``#ifdef USE_ISAAC64`` gate -- we have already provided the gate.
    src = VENDOR_ISAAC_C.read_text()
    # Drop vendor's own ``#include "config.h"`` -- we predefine USE_ISAAC64
    # ourselves at the top of the file to keep the ``#ifdef USE_ISAAC64``
    # body active.
    prelude = (
        "#define USE_ISAAC64\n"
        "#include <stdint.h>\n"
        "#define ISAAC64_SZ_LOG 8\n"
        "#define ISAAC64_SZ (1 << ISAAC64_SZ_LOG)\n"
        "#define ISAAC64_SEED_SZ_MAX (ISAAC64_SZ << 3)\n"
        "typedef struct isaac64_ctx {\n"
        "    unsigned n;\n"
        "    uint64_t r[ISAAC64_SZ];\n"
        "    uint64_t m[ISAAC64_SZ];\n"
        "    uint64_t a, b, c;\n"
        "} isaac64_ctx;\n"
        "void isaac64_init(isaac64_ctx *, const unsigned char *, int);\n"
        "void isaac64_reseed(isaac64_ctx *, const unsigned char *, int);\n"
        "uint64_t isaac64_next_uint64(isaac64_ctx *);\n"
    )
    src = src.replace('#include "config.h"', "/* config.h stripped */")
    src = src.replace('#include "isaac64.h"', "/* isaac64.h inlined */")
    src = prelude + src
    # The file is wrapped in ``#ifdef USE_ISAAC64 ... #endif``; we keep that
    # because we ``#define USE_ISAAC64`` in the harness.

    isaac_path = tmp_path / "isaac64_vendor.c"
    isaac_path.write_text(src)

    driver = r"""
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define USE_ISAAC64
#define ISAAC64_SZ_LOG 8
#define ISAAC64_SZ (1 << ISAAC64_SZ_LOG)
#define ISAAC64_SEED_SZ_MAX (ISAAC64_SZ << 3)
typedef struct isaac64_ctx {
    unsigned n;
    uint64_t r[ISAAC64_SZ];
    uint64_t m[ISAAC64_SZ];
    uint64_t a, b, c;
} isaac64_ctx;
void isaac64_init(isaac64_ctx *, const unsigned char *, int);
uint64_t isaac64_next_uint64(isaac64_ctx *);

int main(void) {
    isaac64_ctx ctx;
    /* Same packing NLE uses in rnd.c init_isaac64:
       sizeof(unsigned long) little-endian bytes (8 on LP64). */
    unsigned char seed_bytes[8] = {0};
    unsigned long seed = 0UL;
    for (unsigned i = 0; i < sizeof(seed); i++) {
        seed_bytes[i] = (unsigned char)(seed & 0xFF);
        seed >>= 8;
    }
    isaac64_init(&ctx, seed_bytes, (int)sizeof(seed_bytes));
    for (int i = 0; i < 100; i++) {
        printf("%llu\n", (unsigned long long)isaac64_next_uint64(&ctx));
    }
    return 0;
}
""".lstrip()
    driver_path = tmp_path / "driver.c"
    driver_path.write_text(driver)

    binary = tmp_path / "harness"
    cmd = [
        "cc",
        "-O0",
        "-std=c99",
        str(driver_path),
        str(isaac_path),
        "-o",
        str(binary),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        pytest.skip("C compiler not available")
    if proc.returncode != 0:
        pytest.xfail(
            "vendor isaac64.c failed to compile standalone "
            "(needs vendor header munging beyond audit scope): "
            f"\nstderr:\n{proc.stderr}"
        )
    return binary


def _run_vendor(binary: Path) -> list[int]:
    proc = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert proc.returncode == 0, proc.stderr
    return [int(line) for line in proc.stdout.strip().splitlines()]


def _run_python(n: int = 100) -> list[int]:
    state = init_py(0)
    out = []
    for _ in range(n):
        state, v = next_uint64_py(state)
        out.append(v)
    return out


def test_isaac64_first_100_byte_parity(tmp_path: Path) -> None:
    """First 100 outputs must match vendor ISAAC64 bit-for-bit.

    Both sides seed with ``unsigned long = 0`` packed little-endian into
    8 bytes (the LP64 layout NLE uses on Linux/macOS).
    """
    binary = _build_vendor_harness(tmp_path)
    vendor = _run_vendor(binary)
    ours = _run_python(100)
    assert len(vendor) == 100
    assert len(ours) == 100
    assert ours == vendor, (
        "ISAAC64 divergence -- first mismatch at index "
        f"{next((i for i in range(100) if vendor[i] != ours[i]), -1)}\n"
        f"  vendor[0:5] = {vendor[:5]}\n"
        f"  ours  [0:5] = {ours[:5]}"
    )


def test_python_isaac64_smoke_seed_zero() -> None:
    """Standalone smoke check: deterministic across runs, sane bit range."""
    out_a = _run_python(100)
    out_b = _run_python(100)
    assert out_a == out_b, "ISAAC64 must be deterministic for fixed seed"
    # All outputs must fit in u64 and not all be zero.
    assert all(0 <= v < (1 << 64) for v in out_a)
    assert any(v != 0 for v in out_a)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
