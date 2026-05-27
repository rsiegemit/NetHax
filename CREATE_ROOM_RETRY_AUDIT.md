# CREATE_ROOM Retry / RNG-draw Audit

**Scope:** `vendor/nle/src/sp_lev.c::create_room` (lines 1126–1292) vs
`Nethax/nethax/dungeon/create_room.py` (commit 595f902).

---

## 1. Vendor retry behaviour

### Loop structure

```c
do {
    /* ... random-path body ... */
} while (++trycnt <= 100 && !r1);
```

The loop **exits as soon as `r1 != NULL`** — i.e. the *first* attempt that
passes both the rect-fits test AND `check_room` terminates the loop.

### Typical attempt count per room (seed=0, freshly-stoned level)

| Room # | Expected attempts | Reason |
|--------|------------------|--------|
| 0–2    | 1                | Full rect pool, very likely any picked rect fits |
| 3–4    | 1–3              | Pool shrinking but still large rects available |
| 5–6    | 2–10             | Remaining rects smaller; fits-test fails more often |

On a freshly-stoned level `check_room` never draws `rn2(3)` (all cells are
STONE/0), so no hidden RNG there.

### RNG draws per *successful* attempt (random path, non-vault)

| Draw | Expression | Always? |
|------|------------|---------|
| D1   | `rnd_rect()` → `rn2(rect_cnt)` | Yes (once per attempt) |
| D2   | `rn2((hx-lx>28)?12:8)` for dx | Yes (once per attempt, non-vault) |
| D3   | `rn2(4)` for dy | Yes (once per attempt, non-vault) |
| D4   | `rn2(…)` for xabs | Only when rect fits (1195 passes) |
| D5   | `rn2(…)` for yabs | Same gate as D4 |
| D6   | `rn2(nroom)` | Only when outer gate fires (nroom>0, ly==0, hy>=ROWNO-1) |
| D7   | `rn1(3,2)` for yabs override | Only when D6 gate fires AND `rn2(nroom)==0` |

**Failed attempt cost:** D1+D2+D3 = 3 draws if the rect-fits test fails
immediately (line 1195); D1–D5 = 5 draws if the rect fits but `check_room`
rejects.

**Successful attempt cost:** 5 draws (D1–D5) normally; 6–7 when D6/D7 fire.

---

## 2. Our JAX implementation

```python
lax.fori_loop(0, _MAX_TRYCNT, _body, init)   # _MAX_TRYCNT = 100
```

The loop **always runs exactly 100 iterations**. The `done` flag gates the
body via `lax.cond(done, _skip_attempt, _do_attempt, ...)`:

```python
def _skip_attempt(state):
    r, p = state
    return AttemptResult(rng=r, pool=p, success=False, ...)  # NO draws
```

**Critical finding:** `_skip_attempt` returns `rng` unchanged — **zero RNG
draws** on iterations after the first success. The JAX implementation
therefore consumes the same number of draws as vendor for the *body*, but
only because `lax.cond` properly suppresses the untaken branch.

### Draw count comparison

| Scenario | Vendor draws | Our draws | Delta |
|----------|-------------|-----------|-------|
| Room succeeds on attempt 1 | 5–7 | 5–7 | **0** |
| Room succeeds on attempt 3 (2 failed) | 3+3+5 = 11–17 | same | **0** |
| Room fails all 100 | 300–500 | 300–500 | **0** |

The retry loop itself is **not** the source of the ~1682-draw deficit.

---

## 3. Where the real deficit comes from

The per-room draw count matches. The deficit must lie elsewhere in `mklev`:

1. **`makerooms` outer loop** — vendor calls `create_room` from a `while`
   loop gated on `rnd_rect()` returning non-NULL, so each *rejected* outer
   iteration consumes 1 `rnd_rect` draw before even entering `create_room`.
   Our port of that loop needs auditing.

2. **`makecorridors` / `dosdoor`** — corridor generation between rooms is
   RNG-heavy; any mismatch there dwarfs room-placement drift.

3. **`mklev` preamble** — `init_level`, `mineralize`, special object
   scattering, etc. all draw RNG before rooms are even placed.

---

## 4. Recommendation

**No change needed to `create_room_random`.** The `lax.cond(done, _skip, _do)`
pattern correctly suppresses draws on post-success iterations — vendor
equivalence is preserved. Do **not** add phantom draws to pad the loop.

The deficit investigation should move to the `makerooms` outer-loop port and
the post-room mklev phases.
