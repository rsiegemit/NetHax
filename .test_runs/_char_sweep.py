"""Role x Race x Alignment byte-parity sweep for the RESET inventory + @ glyph.

For every VALID NetHack role/race/alignment combination, build a fixed simple
env (Room-5x5) on both the vendor MiniHack side (character="rol-rac-ali-mal")
and the minihax side (bootstrap_character(role, race, align)) and byte-compare
the parts of the reset observation that depend ONLY on u_init / role_init and
NOT on the map:

    - hero @ glyph (the player-monster PM glyph at the hero cell)
    - inv_glyphs, inv_letters, inv_strs  (the starting inventory)

The map itself may still diverge if a role's ini_inv consumes a different
ISAAC64 draw count than vendor -- that is a SEPARATE (level-gen) concern; this
sweep isolates "are all races/classes/items rendered right".

Usage:
    NETHAX_EAGER=1 JAX_PLATFORMS=cpu .venv/bin/python .test_runs/_char_sweep.py
    ... [--role arc] [--env MiniHack-Room-5x5-v0] [--seed 0]
"""
from __future__ import annotations

import argparse

# Reuse the harness setup: gym shim, set_parity_mode(NLE_BYTEPARITY), imports.
import minihax_byteparity as H  # noqa: E402
import numpy as np  # noqa: E402
import jax  # noqa: E402

from Nethax.nethax.constants.roles import Role  # noqa: E402
from Nethax.nethax.constants.races import Race  # noqa: E402
from Nethax.minihax.level_generator import bootstrap_character  # noqa: E402
from Nethax.minihax.minihax_env import MinihaxEnv  # noqa: E402
from Nethax.nethax.obs.nle_obs import build_nle_observation  # noqa: E402

_ROLE_ABBR = {
    Role.ARCHEOLOGIST: "arc", Role.BARBARIAN: "bar", Role.CAVEMAN: "cav",
    Role.HEALER: "hea", Role.KNIGHT: "kni", Role.MONK: "mon",
    Role.PRIEST: "pri", Role.RANGER: "ran", Role.ROGUE: "rog",
    Role.SAMURAI: "sam", Role.TOURIST: "tou", Role.VALKYRIE: "val",
    Role.WIZARD: "wiz",
}
_RACE_ABBR = {
    Race.HUMAN: "hum", Race.ELF: "elf", Race.DWARF: "dwa",
    Race.GNOME: "gno", Race.ORC: "orc",
}
_ALIGN_ABBR = {0: "law", 1: "neu", 2: "cha"}

# Which alignments each role permits (roles.py comments).
_ROLE_ALIGNS = {
    Role.ARCHEOLOGIST: {0, 1}, Role.BARBARIAN: {1, 2}, Role.CAVEMAN: {0, 1},
    Role.HEALER: {1}, Role.KNIGHT: {0}, Role.MONK: {0, 1, 2},
    Role.PRIEST: {0, 1, 2}, Role.RANGER: {1, 2}, Role.ROGUE: {2},
    Role.SAMURAI: {0}, Role.TOURIST: {1}, Role.VALKYRIE: {0, 1},
    Role.WIZARD: {1, 2},
}
# Which races each role permits.
_ROLE_RACES = {
    Role.ARCHEOLOGIST: {Race.HUMAN, Race.DWARF, Race.GNOME},
    Role.BARBARIAN: {Race.HUMAN, Race.ORC},
    Role.CAVEMAN: {Race.HUMAN, Race.DWARF, Race.GNOME},
    Role.HEALER: {Race.HUMAN, Race.GNOME},
    Role.KNIGHT: {Race.HUMAN},
    Role.MONK: {Race.HUMAN},
    Role.PRIEST: {Race.HUMAN, Race.ELF},
    Role.RANGER: {Race.HUMAN, Race.ELF, Race.GNOME, Race.ORC},
    Role.ROGUE: {Race.HUMAN, Race.ORC},
    Role.SAMURAI: {Race.HUMAN},
    Role.TOURIST: {Race.HUMAN},
    Role.VALKYRIE: {Race.HUMAN, Race.DWARF},
    Role.WIZARD: {Race.HUMAN, Race.ELF, Race.GNOME, Race.ORC},
}
# Which alignments each race permits.
_RACE_ALIGNS = {
    Race.HUMAN: {0, 1, 2}, Race.ELF: {2}, Race.DWARF: {0},
    Race.GNOME: {1}, Race.ORC: {2},
}


def valid_combos():
    out = []
    for role in Role:
        for race in _ROLE_RACES[role]:
            aligns = _ROLE_ALIGNS[role] & _RACE_ALIGNS[race]
            for al in sorted(aligns):
                out.append((role, race, al))
    return out


def _hero_glyph(dump):
    y, x = dump["agent_yx"]
    g = dump["glyphs"]
    if 0 <= y < g.shape[0] and 0 <= x < g.shape[1]:
        return int(g[y, x])
    return -1


def vendor_dump_char(env_id, seed, charstr):
    cls = H._env_id_to_vendor_cls(env_id)
    H.random.seed(seed)
    env = cls(observation_keys=H._OBS_KEYS, character=charstr)
    env.seed(seed, seed, reseed=False)
    r = env.reset()
    obs = r[0] if isinstance(r, tuple) else r
    d = {
        "glyphs": np.asarray(obs["glyphs"]),
        "chars": np.asarray(obs["chars"]),
        "agent_yx": H._agent_yx_from_blstats(np.asarray(obs["blstats"])),
        "inv_glyphs": np.asarray(obs["inv_glyphs"]),
        "inv_letters": np.asarray(obs["inv_letters"]),
        "inv_strs": np.asarray(obs["inv_strs"]),
    }
    try:
        env.close()
    except Exception:
        pass
    return d


def minihax_dump_char(env_id, seed, role, race, al, gender=0):
    with bootstrap_character(role, race, al, gender):
        env = MinihaxEnv(env_id)
        state, _ = env.reset(jax.random.key(seed))
        obs = build_nle_observation(state)
    return {
        "glyphs": np.asarray(obs["glyphs"]),
        "chars": np.asarray(obs["chars"]),
        "agent_yx": H._agent_yx_from_blstats(np.asarray(obs["blstats"])),
        "inv_glyphs": np.asarray(obs["inv_glyphs"]),
        "inv_letters": np.asarray(obs["inv_letters"]),
        "inv_strs": np.asarray(obs["inv_strs"]),
    }


def cmp_inv(v, m):
    """Compare hero glyph + inventory only. Return None or short reason."""
    hv, hm = _hero_glyph(v), _hero_glyph(m)
    if hv != hm:
        return f"hero_glyph vendor={hv} minihax={hm}"
    for key in ("inv_glyphs", "inv_letters"):
        a, b = v[key], m[key]
        if a.shape != b.shape:
            return f"{key} shape {a.shape}!={b.shape}"
        ne = np.argwhere(a != b)
        if ne.size:
            i = int(ne[0, 0])
            return f"{key}[{i}] vendor={int(a[i])} minihax={int(b[i])}"
    a, b = v["inv_strs"], m["inv_strs"]
    if a.shape != b.shape:
        return f"inv_strs shape {a.shape}!={b.shape}"
    if not np.array_equal(a, b):
        ne = np.argwhere(a != b)
        i, j = int(ne[0, 0]), int(ne[0, 1])
        # decode the two rows to strings for readability
        vs = bytes(int(c) for c in a[i] if c).decode("latin1", "replace")
        ms = bytes(int(c) for c in b[i] if c).decode("latin1", "replace")
        return f"inv_strs[slot={i}] vendor={vs!r} minihax={ms!r}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="MiniHack-Room-5x5-v0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--role", default=None, help="3-letter abbrev to filter")
    ap.add_argument("--gender", default="both", choices=["mal", "fem", "both"],
                    help="Which gender(s) to sweep (default both).")
    ap.add_argument("--inv-only", action="store_true",
                    help="Compare only hero glyph + inventory (legacy). Default "
                         "compares the FULL reset obs: map glyphs/chars + agent "
                         "+ inventory (i.e. does NOT isolate inventory).")
    args = ap.parse_args()

    combos = valid_combos()
    if args.role:
        combos = [c for c in combos if _ROLE_ABBR[c[0]] == args.role.lower()]
    genders = ["mal", "fem"] if args.gender == "both" else [args.gender]

    def _compare(v, m):
        return cmp_inv(v, m) if args.inv_only else H.diff_dumps(v, m)

    _GENDER_INT = {"mal": 0, "fem": 1}

    npass = nfail = nerr = 0
    fails = []
    for (role, race, al) in combos:
        for g in genders:
            charstr = (f"{_ROLE_ABBR[role]}-{_RACE_ABBR[race]}-"
                       f"{_ALIGN_ABBR[al]}-{g}")
            try:
                v = vendor_dump_char(args.env, args.seed, charstr)
                m = minihax_dump_char(args.env, args.seed, role, race, al,
                                      _GENDER_INT[g])
            except Exception as e:
                nerr += 1
                print(f"  {charstr:20s} ERROR {type(e).__name__}: {e}")
                continue
            reason = _compare(v, m)
            if reason is None:
                npass += 1
                print(f"  {charstr:20s} PASS")
            else:
                nfail += 1
                fails.append((charstr, reason))
                print(f"  {charstr:20s} FAIL  {reason}")

    mode = "inv-only" if args.inv_only else "FULL-obs"
    print(f"\n[char-sweep] env={args.env} seed={args.seed} mode={mode} "
          f"gender={args.gender}  PASS={npass} FAIL={nfail} ERR={nerr}  "
          f"(of {len(combos) * len(genders)} role x race x gender)")
    if fails:
        print("[char-sweep] FAILS:")
        for cs, r in fails:
            print(f"    {cs:20s} {r}")


if __name__ == "__main__":
    main()
