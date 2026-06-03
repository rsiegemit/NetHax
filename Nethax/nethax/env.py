"""Top-level NetHack JAX environment — NLE-compatible step/reset interface.

Wave 1 status:
    Reset returns the default EnvState and an empty NLE observation.
    Step is a true no-op: returns the input state, zero reward, done=False.

In later waves:
    Wave 2 wires action dispatch (movement) and observation builders.
    Wave 3 turns combat/magic actions into real outcomes.
    Wave 4 connects monster AI and dungeon traversal.

Canonical reference: vendor/nle/nle/env/base.py for the API contract.
"""
from __future__ import annotations
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp

from Nethax.nethax.state import EnvState, StaticParams
from Nethax.nethax.obs.nle_obs import build_nle_observation
from Nethax.nethax.subsystems.action_dispatch import dispatch_action
from Nethax.nethax.subsystems.monster_ai import step as _monster_ai_step
from Nethax.nethax.subsystems.status_effects import step as _status_step
from Nethax.nethax.subsystems.status_effects import tick_hallu_expiry as _tick_hallu_expiry
from Nethax.nethax.subsystems.status_effects import tick_luck_drift as _tick_luck_drift
from Nethax.nethax.subsystems.status_effects import (
    tick_slime_cancels_stoning as _tick_slime_cancels_stoning,
)
from Nethax.nethax.subsystems.timer_queue import tick_timers as _tick_timer_queue
from Nethax.nethax.subsystems.occupation import tick_occupation as _tick_occupation
from Nethax.nethax.subsystems.ascension import maybe_ascend
from Nethax.nethax.subsystems.polymorph import step as _polymorph_step
from Nethax.nethax.subsystems.shop import shop_step as _shop_step
from Nethax.nethax.dungeon.branches import (
    generate_main_branch_l1,
    generate_main_branch_l1_with_features,
    consume_init_dungeons_draws,
    consume_init_dungeons_variable_draws,
)
from Nethax.nethax.dungeon.spawning import populate_level_with_monsters
from Nethax.nethax.constants.roles import Role
from Nethax.nethax.constants.races import Race
from Nethax.nethax.constants import TileType
from Nethax.nethax.subsystems.character import create_character, get_starting_pet
from Nethax.nethax.subsystems.skills import init_skills
from Nethax.nethax.subsystems.digging import dig_tick as _dig_tick
from Nethax.nethax.subsystems.messages import clear_message as _clear_message
from Nethax.nethax.subsystems.riding import tick_gallop as _tick_gallop
from Nethax.nethax.subsystems.riding import tick_saddle as _tick_saddle
from Nethax.nethax.subsystems.swallow import digest_tick as _digest_tick
from Nethax.nethax.subsystems.experience import newexplevel as _newexplevel
from Nethax.nethax.subsystems.regions import run_regions as _run_regions
from Nethax.nethax.subsystems.mplayer import (
    maybe_seed_astral_mplayers as _maybe_seed_astral_mplayers,
)
from Nethax.nethax.parity_mode import use_vendor_rng
from Nethax.nethax import vendor_rng as _vendor_rng
from Nethax.nethax.obs.glyph_shuffle import compute_descr_shuffle as _compute_descr_shuffle
from Nethax.nethax.nle_action_map import maybe_remap_action as _maybe_remap_action


class NethaxEnv:
    """Minimal NLE-style JAX env.

    Methods are Python-side (no JIT here) so children/tests can wrap them.
    The state pytree itself is fully jittable.
    """

    def __init__(self, static: StaticParams | None = None):
        self.static = static or StaticParams()
        self._step_jit = jax.jit(_step_impl)

    def reset(
        self,
        rng: jax.Array,
        role: "Role | None" = None,
        race: "Race | None" = None,
        alignment: int = 0,
        disp_seed: int | None = None,
    ) -> Tuple[EnvState, Dict[str, jax.Array]]:
        """Return (initial_state, initial_observation).

        Parameters
        ----------
        rng       : JAX PRNG key
        role      : Role enum value; defaults to VALKYRIE if None.
        race      : Race enum value; defaults to HUMAN if None.
        alignment : 0=lawful, 1=neutral, 2=chaotic; default 0.
        disp_seed : Optional explicit DISP-stream seed.  When ``None``
                    (the default), the DISP stream is seeded with the
                    same integer as CORE — matching NLE's validator
                    ``env.seed(seeds=(s, s))`` convention.
                    Cite: vendor/nle/src/nle.c:530-532
                          ``nle_set_seed(core, disp, reseed)`` seeds the
                          two ``rnglist[]`` entries independently.
        """
        if role is None:
            role = Role.VALKYRIE
        if race is None:
            race = Race.HUMAN

        rng_state, rng_level, rng_char, rng_monsters = jax.random.split(rng, 4)
        state = EnvState.default(rng=rng_state, static=self.static)

        # Vendor NetHack starts the move counter at 1, not 0 (the game begins
        # on turn 1).  ``decl.c:195`` initialises ``moves = 1L`` and the first
        # observation after newgame already reports moves==1; the per-step
        # ``svm.moves++`` (allmain.c:244) then advances it to 2, 3, ....  Our
        # ``EnvState.default`` seeds ``timestep=0``, so without this the
        # blstats BL_TIME field lags NLE by exactly 1 at every step.  Cite:
        # vendor/nle/src/decl.c:195.  ``timestep`` (the monotonic env-step
        # clock used for timer deadlines / hashes) and ``game_moves`` (the
        # vendor ``moves`` turn counter surfaced as BL_TIME) both start at 1.
        # They diverge thereafter: ``timestep`` advances every env step while
        # ``game_moves`` only advances on time-consuming actions (a blocked
        # wall-bump takes zero game time, so vendor ``moves`` does not tick).
        state = state.replace(timestep=jnp.int32(1), game_moves=jnp.int32(1))

        # NLE_BYTEPARITY: seed ISAAC64 from the host-side integer derived
        # from the incoming PRNGKey.  Vendor NLE seeds each rnglist[] entry
        # via ``init_isaac64(seed)`` where ``seed`` is the platform unsigned
        # long.  We collapse the two PRNGKey words via XOR so a 64-bit seed
        # round-trips deterministically.  Cite: vendor/nle/src/rnd.c
        # ``init_isaac64`` (lines 42-58) and hacklib.c::set_random (854-868).
        #
        # Once seeded, re-derive the dungeon-gen / character / monster sub-keys
        # from the ISAAC64 stream itself rather than from Threefry ``split``.
        # This is the SHIM step toward byte parity: the downstream consumers
        # still use ``jax.random.randint`` (so the dungeon-gen call graph and
        # JIT shape stay intact), but the PRNGKey they read is now a function
        # of the ISAAC64 output — so the layout, monster placement, and
        # character rolls become deterministic with respect to the vendor RNG
        # rather than Threefry.  A truly byte-exact path would replace each
        # ``randint`` with ``rn2`` and thread ``Isaac64State`` through every
        # call site; that requires a much larger refactor.  Cite:
        # vendor/nethack/src/rnd.c (rn2/rnd) under USE_ISAAC64 — every
        # vendor RNG draw bottoms out in ``isaac64_next_uint64() % x``.
        # Vendor-RNG scalar pre-draws — set defaults so the dungeon-gen kwargs
        # stay traceable on the non-vendor path.  These are overwritten below
        # from ISAAC64 when ``use_vendor_rng()`` is active.
        n_rooms_vendor: int = 8         # generate_rooms default
        n_monsters_vendor: int = 5      # populate_level_with_monsters default
        if use_vendor_rng():
            # Derive the ISAAC64 seed from the raw PRNGKey integer so it
            # matches NLE's ``env.seed(seeds=(s, s))`` byte-for-byte.  NLE
            # packs an ``unsigned long`` little-endian as the ISAAC64 seed
            # (vendor/nle/src/rnd.c::init_isaac64 lines 40-57), so for seed
            # ``s`` we need to pass exactly ``s`` to ``_vendor_rng.init``.
            #
            # JAX's ``jax.random.PRNGKey(s)`` returns the key ``[s_hi, s_lo]``
            # (two uint32 words, big-endian).  Reconstruct ``s`` as a uint64
            # by combining ``(hi << 32) | lo``.  Previously we used
            # ``jax.random.bits(rng, (), dtype=jnp.uint64)`` which hashes the
            # key (PRNGKey(0) hashed to 7719171245655871230 instead of 0),
            # leading to a completely different ISAAC64 stream from NLE's.
            # NOTE: ``_vendor_rng.init`` still runs reference Python ISAAC64
            # math, so this branch is not yet ``jax.vmap``-safe; under vmap
            # the ``int(seed_arr)`` below would raise ConcretizationTypeError.
            # The non-vendor default path (ParityMode.NLE) skips this branch
            # entirely and vmaps cleanly.
            seed_hi = jnp.uint64(rng[0])
            seed_lo = jnp.uint64(rng[1])
            seed_arr = (seed_hi << jnp.uint64(32)) | seed_lo
            # Use vmap-compatible JAX init.  Bit-exact with the legacy
            # _vendor_rng.init(int(seed_arr)) for any uint64 seed
            # (verified for seeds 0..999999 against Python reference).
            v_state = _vendor_rng.init_jax(seed_arr)

            # Seed the DISP stream (rnglist[DISP]) alongside CORE.  NLE's
            # ``nle_set_seed`` takes BOTH integers and seeds the two
            # ``rnglist[]`` entries via independent ``init_isaac64`` calls
            # — so DISP's stream is wholly independent of CORE's even when
            # both share the same numeric seed.  The validator default
            # ``env.seed(seeds=(s, s), reseed=False)`` (vendor
            # vendor/nle/nle/env/base.py:441) collapses both seeds to the
            # same integer; we mirror that here unless the caller passes
            # an explicit ``disp_seed`` override.
            # Cite: vendor/nle/src/nle.c:530-532 ``set_random(core, rn2)``
            #       and ``set_random(disp, rn2_on_display_rng)``.
            # disp_seed: only convert to traced uint64 if not already
            if disp_seed is None:
                disp_seed_val = seed_arr
            else:
                disp_seed_val = jnp.uint64(disp_seed)
            v_state_disp = _vendor_rng.init_jax(disp_seed_val)
            state = state.replace(vendor_rng_disp=v_state_disp)

            # Replay vendor ``init_objects()`` BEFORE any dungeon-gen
            # draws so the ISAAC64 stream stays byte-aligned with NLE.
            # Vendor sequence: set_random (RNG seed) → init_objects
            # (shuffle + GEM jitter + WAN_NOTHING coin) → role_init →
            # init_dungeons → u_init → mklev.  We must consume the
            # same ~200 ISAAC64 draws here.  Cite:
            # vendor/nle/src/allmain.c::newgame lines 585-627;
            # vendor/nle/src/o_init.c::init_objects (111-183).
            v_state, descr_idx = _compute_descr_shuffle(v_state)
            state = state.replace(descr_idx=descr_idx)

            # role_init (vendor/nle/src/role.c:2011-2137) sits between
            # init_objects and init_dungeons in allmain.c::newgame
            # (vendor/nle/src/allmain.c:606-608).  For every pre-specified
            # role/race/gender/alignment loadout used by Nethax callers it
            # consumes ZERO ISAAC64 draws because every conditional RNG path
            # is gated off.  Per-loadout audit:
            #
            # env.reset default = Valkyrie-Human-Lawful (env.py:97-100):
            #   - validrole/validrace/validalign all true (Valkyrie.allow at
            #     role.c:526 carries MH_HUMAN | ROLE_FEMALE | ROLE_LAWFUL)
            #   - validgend(Val,Hum,Male)=false → flags.female=!flags.female
            #     (role.c:2041-2042), deterministic flip, no rn2
            #   - quest-leader rn2(100) at role.c:2070 skipped: PM_NORN has
            #     M2_FEMALE → is_female(pm)=true (monst.c:2955)
            #   - quest-nemesis rn2(100) at role.c:2091 skipped: PM_LORD_SURTUR
            #     has M2_MALE → is_male(pm)=true (monst.c:3096)
            #   - pantheon randrole loop at role.c:2095-2099 skipped:
            #     Valkyrie.lgod = "Tyr" non-NULL (role.c:511)
            #
            # NLE-shim default = Monk-Human-Neutral-Male (nle_shim.py:78,
            # mirrors vendor/nle/nle/env/base.py "mon-hum-neu-mal"):
            #   - validrole/validrace/validgend/validalign all true (Monk.allow
            #     at role.c:261 carries MH_HUMAN|ROLE_MALE|ROLE_FEMALE|all 3 alignments)
            #   - quest-leader skipped: PM_GRAND_MASTER M2_MALE (monst.c:2902)
            #   - quest-nemesis skipped: PM_MASTER_KAEN  M2_MALE (monst.c:3045)
            #   - pantheon loop skipped: Monk.lgod = "Shan Lai Ching" (role.c:246)
            #
            # Prior-audit Rogue-Human-Chaotic-Male loadout (commit 2aa1252)
            # also lands at 0 draws via the same gating chain.
            #
            # Therefore no ISAAC64 byte-shift is emitted here.  If a future
            # caller passes a role whose lgod is NULL, or quest-leader/nemesis
            # carries neither M2_MALE/FEMALE/NEUTER (so rn2(100) fires), this
            # audit must be revisited and explicit vendor_rng draws inserted
            # at this site to stay byte-aligned with allmain.c::newgame.

            # PARITY FIX: derive the three Threefry sub-keys from the input
            # PRNGKey via jax.random.split, NOT from the ISAAC64 stream.
            # Previously this called ``_vendor_draw_prngkey`` thrice, each of
            # which consumed one raw uint64 from ``v_state`` (ISAAC64 CORE).
            # Vendor C (vendor/nle/src/allmain.c:604-625, role.c:role_init)
            # makes ZERO ISAAC64 draws between init_objects and init_dungeons
            # for any deterministic role/race/alignment loadout, so those 3
            # uint64s shifted the entire ISAAC64 stream by 3 words —
            # producing the first divergence at op#196 (rn2(100)) in
            # /tmp/vendor_rnd_trace_v2.txt vs /tmp/nethax_rnd_trace_jit.txt.
            # Cite: vendor/nle/src/allmain.c:604-615; vendor/nle/src/role.c
            # role_init for the deterministic-loadout RNG-zero audit at
            # env.py:178-212.
            rng_level, rng_char, rng_monsters = jax.random.split(rng, 3)

            # NOTE (Phase 3 — MKLEV_PORT_PLAN.md §1.4):
            #   Previously this block pre-drew rn2(5) + rn2(mc_upper) here
            #   to derive ``n_rooms_vendor`` (5..9) and ``n_monsters_vendor``
            #   for downstream generate_rooms / populate_level_with_monsters
            #   calls.  Vendor C makes NO such pre-draws -- makerooms
            #   (mklev.c:229) drives room count implicitly via
            #   rnd_rect()-exhaustion of the rect pool, and the monster
            #   count rnd((nroom>>1)+1) (mklev.c:804) is drawn AFTER the
            #   per-room feature pass.  Pre-drawing them here shifted every
            #   downstream ISAAC64 byte by 16 (two uint64 outputs), causing
            #   the seed=0 player_x/y divergence flagged in
            #   ISAAC64_CALL_ORDER_AUDIT.md.
            #
            #   Phase 3 makerooms (rooms.py::makerooms) + the new
            #   rn2(nroom)/somex/somey/rn2(nroom-1) stair-pick draws in
            #   branches.py::generate_main_branch_l1 consume the bytes in
            #   vendor order, so these pre-draws are dropped.
            #
            #   The n_rooms_vendor / n_monsters_vendor Python locals keep
            #   their initialised defaults from above (8 / 5) so the
            #   downstream calls still receive concrete ints.  In the
            #   vendor path the *true* nroom is whatever generate_rooms
            #   actually placed; downstream consumers re-derive it from
            #   ``active.sum()``.
            state = state.replace(vendor_rng=v_state)

        # Vendor allmain.c order (lines 604-615):
        #   init_objects → role_init → init_dungeons → init_artifacts → u_init → mklev
        # So consume init_dungeons BEFORE create_character (which models u_init).
        # The prior order was REVERSED — every ISAAC64 draw from position 195+
        # was offset, causing the entire downstream dungeon-gen to mis-align.
        # Citation: vendor/nle/src/allmain.c:604-615; vendor/nle/src/dungeon.c:714.
        if use_vendor_rng():
            new_vrng, _dungeon_state = consume_init_dungeons_draws(state.vendor_rng)
            # Variable-count draws: place_level slot picks (dungeon.c:661) and
            # parent_dlevel branch picks (dungeon.c:398).
            new_vrng, _var_state = consume_init_dungeons_variable_draws(
                new_vrng, _dungeon_state
            )
            state = state.replace(vendor_rng=new_vrng)

            # init_artifacts (vendor/nle/src/artifact.c:81-86) sits between
            # init_dungeons and u_init in allmain.c::newgame (vendor
            # vendor/nle/src/allmain.c:613-615).  It is RNG-neutral:
            #   - artifact.c:83  memset(artiexist, 0, ...)   — no RNG
            #   - artifact.c:84  memset(artidisco, 0, ...)   — no RNG
            #   - artifact.c:85  hack_artifacts()            — pure static
            #     config: fixes alignment/role fields on artilist[] entries
            #     (artifact.c:57-77); no rn2/rnd/rne calls.
            # Therefore no ISAAC64 draw is emitted at this site; this comment
            # records the audit so the vendor-order skeleton stays visible.
            # If $WIZKIT-style late artifact wishes are ever ported they must
            # land between this hook and the u_init draws below to preserve
            # byte alignment with allmain.c::newgame.

        # Apply character creation (stats, inventory, AC) — vendor u_init.
        # In NLE_BYTEPARITY mode, thread the ISAAC64 CORE state through
        # create_character so it can consume the u_init rn2(5) BLINDFOLD
        # draw (vendor/nle/src/u_init.c:753-754) in byte-exact call order.
        # The returned dict may include an updated ``vendor_rng`` key; the
        # state.replace(**char_fields) call below threads it forward into
        # EnvState so subsequent dungeon-gen draws stay byte-aligned.
        char_fields = create_character(
            rng_char, role, race, alignment,
            vendor_rng=state.vendor_rng if use_vendor_rng() else None,
        )
        state = state.replace(**char_fields)

        # Initialise role-specific skill caps (vendor/nethack/src/u_init.c Skill_X tables).
        state = state.replace(skills=init_skills(role))

        # Emit the role-specific NLE intro line on row 0 of the message line.
        # Pass race + alignment so the full vendor welcome string is rendered:
        #   "%s %s, welcome to NetHack!  You are a%s %s %s."
        #   Hello()  plname           buf       race.adj  role.name
        # Cite: vendor/nethack/src/allmain.c::welcome lines 679-691;
        #       vendor/nle/nle/env/base.py:306 (plname = "Agent").
        from Nethax.nethax.subsystems.messages import (
            emit_role_intro     as _emit_role_intro,
            emit_moonphase_message as _emit_moonphase_message,
        )
        state = state.replace(
            messages=_emit_role_intro(
                state.messages,
                int(role),
                race=int(race),
                alignment=int(alignment),
            ),
        )

        # Vendor moveloop preamble (allmain.c:53-66):
        #     flags.moonphase = phase_of_the_moon();
        #     if FULL_MOON:  pline(...); change_luck(+1);
        #     else if NEW_MOON: pline(...);
        #     if friday_13th(): pline(...); change_luck(-1);
        # Fires AFTER the welcome banner, BEFORE pickup(1) — so the lunar /
        # Friday-13 line overwrites the welcome in NLE's message obs whenever
        # the wallclock condition triggers.  phase_of_the_moon() is
        # wallclock-driven (hacklib.c:1098-1110), not seed-driven; without
        # this port, Nethax's step-0 obs diverges on full / new moon /
        # Friday-13 days.
        new_msgs, _luck_delta = _emit_moonphase_message(state.messages)
        state = state.replace(messages=new_msgs)
        if _luck_delta != 0:
            state = state.replace(
                player_luck=(state.player_luck.astype(jnp.int8)
                             + jnp.int8(_luck_delta)),
            )

        # Vendor mklev() begins by reseeding BOTH streams (vendor
        # mklev.c:996-997)::
        #     reseed_random(rn2);
        #     reseed_random(rn2_on_display_rng);
        # Under the validator config (``reseed=False`` in env.seed,
        # ``has_strong_rngseed=False``) both calls are no-ops, so the
        # streams pass through unchanged.  Calling the helper explicitly
        # keeps the structural correspondence with vendor C and gives a
        # single hook to extend if a future run flips ``reseed=True``.
        # Cite: vendor/nle/src/mklev.c:996-997
        #       vendor/nle/src/hacklib.c:906-914 ``reseed_random``.
        if use_vendor_rng():
            state = state.replace(
                vendor_rng=_vendor_rng.reseed_random(state.vendor_rng),
                vendor_rng_disp=_vendor_rng.reseed_random(state.vendor_rng_disp),
            )

        # Generate Main branch level 1 and write into the [branch=0, level=0]
        # slot.  This includes the per-room independent feature rolls
        # (fountain / altar / grave / traps) and the 2x2 detached vault —
        # vendor/nethack/src/mklev.c::mklev (line 1577) which calls
        # fill_ordinary_room (line 939) for every OROOM/THEMEROOM and the
        # vault gate at lines 404-410 / 1316-1342.
        # Pass the ISAAC64 state (or None) into dungeon-gen.  Under
        # NLE_BYTEPARITY the per-room y/x/h/w/lit draws consume the same
        # stream as vendor C; in default Threefry mode this is a no-op.
        vendor_rng_for_gen = state.vendor_rng if use_vendor_rng() else None
        # Vendor cite: vendor/nle/src/mklev.c:813-817 — the per-OROOM
        # sleeping-monster spawn is the FIRST draw in each
        # ``fill_ordinary_room`` iteration (before traps/gold/fountain/
        # etc).  Under NLE_BYTEPARITY we now thread ``state`` into
        # :func:`generate_main_branch_l1_with_features` so the monster
        # spawn interleaves with each room's fills inside
        # ``fill_ordinary_rooms`` — see
        # :func:`Nethax.nethax.dungeon.spawning.spawn_oroom_monster_scanbody`
        # which is invoked as step 1 of each per-OROOM iteration inside a
        # SINGLE ``lax.scan`` (no Python unroll), threading monster_ai +
        # next_slot through the scan carry.  The separate
        # ``populate_level_with_monsters`` post-fill call is no longer
        # required.
        if use_vendor_rng():
            (
                terrain,
                _rooms,
                _active,
                up_pos,
                down_pos,
                new_features,
                new_traps,
                vendor_rng_after_gen,
                state,
            ) = generate_main_branch_l1_with_features(
                rng_level,
                self.static,
                state.features,
                state.traps,
                flat_lv=0,
                depth=1,
                player_align=int(alignment),
                n_rooms=n_rooms_vendor,
                vendor_rng=vendor_rng_for_gen,
                state=state,
            )
        else:
            (
                terrain,
                _rooms,
                _active,
                up_pos,
                down_pos,
                new_features,
                new_traps,
                vendor_rng_after_gen,
            ) = generate_main_branch_l1_with_features(
                rng_level,
                self.static,
                state.features,
                state.traps,
                flat_lv=0,
                depth=1,
                player_align=int(alignment),
                n_rooms=n_rooms_vendor,
                vendor_rng=vendor_rng_for_gen,
            )
        state = state.replace(
            terrain=state.terrain.at[0, 0].set(terrain),
            player_pos=up_pos.astype(jnp.int16),
            features=new_features,
            traps=new_traps,
        )
        # When dungeon-gen consumed ISAAC64, commit the threaded state back.
        if use_vendor_rng() and vendor_rng_after_gen is not None:
            state = state.replace(vendor_rng=vendor_rng_after_gen)

        # Threefry-only fallback: when NLE_BYTEPARITY is off, the
        # in-loop per-OROOM monster spawn is skipped (it requires the
        # vendor_rng path).  Drive the legacy host-side per-OROOM
        # populate as a separate pass.  Vendor cite:
        # vendor/nle/src/mklev.c:813-817.
        if not use_vendor_rng():
            state = populate_level_with_monsters(
                state, rng_monsters,
                rooms=_rooms, active=_active,
            )

        # Vendor mklev() ends by reseeding BOTH streams a second time
        # (mklev.c:1034-1035)::
        #     reseed_random(rn2);
        #     reseed_random(rn2_on_display_rng);
        # No-op under validator config (``has_strong_rngseed=False``) — see
        # the matching entry hook above.  Mirroring the C structure here
        # keeps Nethax wire-aligned with vendor for any future
        # ``reseed=True`` parity run.
        # Cite: vendor/nle/src/mklev.c:1034-1035.
        if use_vendor_rng():
            state = state.replace(
                vendor_rng=_vendor_rng.reseed_random(state.vendor_rng),
                vendor_rng_disp=_vendor_rng.reseed_random(state.vendor_rng_disp),
            )

        # Spawn starting pet adjacent to player — vendor/nethack/src/u_init.c::makedog.
        # Host-side (reset is not jit-compiled), so Python loops are fine.
        # Under NLE_BYTEPARITY, thread vendor_rng so the rn2(2) pet-type coin
        # flip (dog.c:66) is consumed for roles with petnum == NON_PM.
        if use_vendor_rng():
            state, new_vrng = _spawn_starting_pet(state, role, vendor_rng=state.vendor_rng)
            state = state.replace(vendor_rng=new_vrng)
            # Vendor makedog (NLE 3.x dog.c::makedog → makemon → ... → moveloop)
            # continues past the rn2(2) pet-type coin flip (dog.c:66) with the
            # following ISAAC64 stream consumption.  Both kitten (PM 32) and
            # little dog (PM 16) have ``LVL(2, ...)`` in vendor/nle/src/monst.c,
            # but ``newmonhp`` calls ``adj_lev(ptr)`` first (makemon.c:989) and
            # adj_lev (makemon.c:1757) reduces level by 1 when ``mlevel >
            # level_difficulty()`` — for a Dlvl=1 fresh game ``level_difficulty
            # = 1`` so the pet's effective ``m_lev = 1``.  Then ``d(1, 8)``
            # silently consumes ONE uint64 via inline RND() (rnd.c:208-224
            # bypasses the rn2/rnd trace hook):
            #
            #   makemon body:
            #     enexto_core (teleport.c:215)            rn2(N)        TRACED  (consumed inside _spawn_starting_pet)
            #     newmonhp d(m_lev=1, 8) (makemon.c:1011) SILENT × 1    [hit-dice roll]
            #     mtmp->female = rn2(2) (makemon.c:1226)  rn2(2)        TRACED
            #   m_initinv (makemon.c:794-797):
            #     rnd_defensive_item gate                 rn2(50)       TRACED
            #     rnd_misc_item gate                      rn2(100)      TRACED
            #   makemon trailer (makemon.c:1386):
            #     is_domestic saddle gate                 rn2(100)      TRACED
            #   moveloop entry (allmain.c:67):
            #     context.rndencode                       rnd(9000)     TRACED
            #
            # Previous attempt consumed the silent uint64 AFTER the female
            # rn2(2), which made the female draw read newmonhp's slot instead
            # of the female slot — seed=1's draw 2448 + seed=4's draw 2512 both
            # diverged.  Moving the silent BEFORE female aligns every uint64.
            # Cite vendor/nle/src/makemon.c:989 (adj_lev), :1011 (d call),
            # :1757 (adj_lev impl), rnd.c:208-224 (d inline RND).
            from Nethax.nethax import vendor_rng as _vrng_mod
            v = state.vendor_rng
            # NOTE: the ``enexto_core`` rn2(num_good) draw is consumed INSIDE
            # :func:`_spawn_starting_pet` (NLE_BYTEPARITY path) so it can both
            # advance the ISAAC64 stream AND drive the pet's actual placement
            # cell (matching vendor teleport.c:215).  Do NOT consume another
            # rn2 here — that would advance the stream twice for a single
            # vendor draw and misalign every subsequent uint64.
            #
            # newmonhp d(1, 8) → 1 silent uint64 BEFORE the female rn2:
            v, _ = _vrng_mod.next_uint64_jax(v)       # newmonhp d(1,8)
            v, _ = _vrng_mod.rn2_jax(v, 2)            # mtmp->female = rn2(2)
            v, _ = _vrng_mod.rn2_jax(v, 50)           # m_initinv rnd_defensive_item gate
            v, _ = _vrng_mod.rn2_jax(v, 100)          # m_initinv rnd_misc_item gate
            v, _ = _vrng_mod.rn2_jax(v, 100)          # is_domestic saddle gate
            v, _ = _vrng_mod.rnd_jax(v, 9000)         # moveloop: context.rndencode
            state = state.replace(vendor_rng=v)
        else:
            state = _spawn_starting_pet(state, role)

        # Seed the explored mask via FOV so the player can see their starting
        # room on the very first frame.  Without this the initial obs is all
        # NO_GLYPH and the UI shows an empty screen.
        #
        # ``build_glyphs`` renders an explored-but-not-visible cell from
        # ``last_seen_terrain`` (vendor/nethack/src/display.c::lastseentyp
        # ~line 850); a never-visible cell whose last_seen is the -1 sentinel
        # falls back to S_stone (cmap_to_glyph(S_stone)=2359).  Seeding only
        # ``explored`` therefore left the entire starting room rendering as
        # stone, because at reset ``visible`` was all-False and
        # ``last_seen_terrain`` was all -1.  Vendor's ``vision_recalc``
        # (vendor/nle/src/vision.c::vision_recalc) runs at level entry and
        # both marks the FOV tiles IN_SIGHT *and* stamps ``levl[x][y].glyph``
        # so the starting room shows its floor (S_room, cmap 19 -> glyph 2378)
        # and bounding walls (S_vwall/S_hwall/corners, cmap 1-6 -> 2360-2365).
        # Mirror that here by also seeding ``visible`` and stamping
        # ``last_seen_terrain`` for the FOV tiles — exactly as the per-step
        # ``subsystems/action_dispatch.py::_apply_fov`` does after a move.
        from Nethax.nethax.fov import compute_fov, lit_room_flood
        terrain_l0_full = state.terrain[0, 0, :, :]
        # Lit-room flood (vendor vision.c:320-335): when the hero stands in a
        # LIT room, vision_recalc makes the whole room region visible at once —
        # the interior PLUS the one-cell bounding wall ring — not just the
        # Bresenham LOS subset.  Compute it FIRST so it can both (a) serve as the
        # per-cell ``lit_mask`` that gates the raycast (dark distal corridor
        # cells reached by a ray through the doorway are dropped — vendor only
        # sets IN_SIGHT for unlit cells within the light radius, not for every
        # line-of-sight cell) and (b) be OR'd back in so the far walls/corners
        # of the starting lit room — which the rays cannot reach — still render
        # on turn 0.  Dark rooms / corridors then reveal only via the LOS rays
        # within the hero's light radius.
        _h, _w = terrain_l0_full.shape
        # Raw Bresenham LOS (no dark-cell gate yet) — used twice below:
        #   1. As the `los_mask` for ``lit_room_flood``'s door gate (vendor
        #      vision.c:744-785 only sets IN_SIGHT on a viz_clear door when the
        #      shadow-caster reaches it; Bresenham approximates that — a wall
        #      corner adjacent to the door stops the ray on the wall, so the
        #      door cell isn't reached).
        #   2. As the input to the dark-cell gate (which is applied locally
        #      below rather than via ``compute_fov(lit_mask=...)``, so we
        #      compute the raycast exactly once).
        vis_raw = compute_fov(
            terrain_l0_full,
            state.player_pos.astype(jnp.int32),
            lit_mask=None,
        )                                                  # bool[MAP_H, MAP_W]
        lit_flood = lit_room_flood(
            state.player_pos.astype(jnp.int32),
            _rooms.x1, _rooms.y1, _rooms.x2, _rooms.y2,
            _active, _rooms.is_lit,
            _h, _w,
            terrain=terrain_l0_full,
            los_mask=vis_raw,
        )                                                  # bool[MAP_H, MAP_W]
        # Local dark-cell gate (mirrors fov.compute_fov's lit_mask branch).
        # A Bresenham-reached cell is actually SEEN iff it's lit OR within the
        # hero's own light radius (Chebyshev <= 1).  Vendor: vision.c:320-335.
        _pr = state.player_pos[0].astype(jnp.int32)
        _pc = state.player_pos[1].astype(jnp.int32)
        _rr = jnp.arange(_h, dtype=jnp.int32)[:, None]
        _cc = jnp.arange(_w, dtype=jnp.int32)[None, :]
        _cheb = jnp.maximum(jnp.abs(_rr - _pr), jnp.abs(_cc - _pc))
        _within_light = _cheb <= jnp.int32(1)
        vis = (vis_raw & (lit_flood | _within_light)) | lit_flood
        new_explored = state.explored.at[0, 0].set(
            state.explored[0, 0] | vis
        )
        # Stamp last_seen_terrain for the visible tiles (display.c lastseentyp).
        terrain_l0 = state.terrain[0, 0]
        old_lst = state.last_seen_terrain[0, 0]
        new_lst = jnp.where(vis, terrain_l0.astype(jnp.int8), old_lst)
        state = state.replace(
            explored=new_explored,
            visible=vis,
            last_seen_terrain=state.last_seen_terrain.at[0, 0].set(new_lst),
        )

        # Drain DISP for the per-obs vendor draws (visible-monster glyph
        # selection + inventory slot ``obj_to_glyph``).  Mirrors
        # vendor/nle/src/display.c:486-498 +
        # vendor/nle/win/rl/winrl.cc:458 every observation cycle.  Under
        # default ParityMode.NLE this is a no-op (use_vendor_rng() is False).
        if use_vendor_rng():
            from Nethax.nethax.obs.nle_obs import consume_disp_for_obs as _consume_disp
            state = state.replace(vendor_rng_disp=_consume_disp(state))

        obs = build_nle_observation(state)
        return state, obs

    def step(
        self,
        state: EnvState,
        action: jax.Array,
        rng: jax.Array,
    ) -> Tuple[EnvState, Dict[str, jax.Array], jax.Array, jax.Array, Dict[str, Any]]:
        """Apply ``action``, return (state', obs, reward, done, info).

        JIT-compiled internally; first call per session pays the compile
        cost (~30-60s for the full dispatch/monster-AI/status pipeline),
        subsequent calls are microseconds.
        """
        new_state, obs, reward, done = self._step_jit(state, action, rng)
        info: Dict[str, Any] = {}
        return new_state, obs, reward, done, info

    # ----------------------------------------------------------------------
    # Batched (vmap-parallel) API — Wave 7 RL-scale training entry points.
    # ----------------------------------------------------------------------
    # The single-env API (reset/step) operates on one PRNGKey at a time.
    # ``reset_batched`` / ``step_batched`` wrap the per-env body in
    # ``jax.vmap`` so 256+ envs can run in lock-step on a single device.
    #
    # Status:
    #   * step_batched     — fully vmap-safe; wraps the JIT-compiled
    #                        ``_step_impl`` which has no host-side branches.
    #   * reset_batched    — vmap-safe in the default ``ParityMode.NLE``
    #                        path (Threefry RNG, pure-JAX dungeon-gen,
    #                        pure-JAX ``_init_attr_vendor`` loop,
    #                        pure-JAX ``_spawn_starting_pet``).
    #                        The opt-in ``ParityMode.NLE_BYTEPARITY``
    #                        branch (ISAAC64) is host-only by design —
    #                        ``_vendor_rng.init`` operates on Python lists
    #                        and cannot vmap.  See ``VMAP_GAPS.md``.
    # ----------------------------------------------------------------------

    def step_batched(
        self,
        states: EnvState,
        actions: jax.Array,
        rngs: jax.Array,
    ) -> Tuple[EnvState, Dict[str, jax.Array], jax.Array, jax.Array]:
        """Vectorised step over a leading batch dim B.

        Parameters
        ----------
        states  : pytree where every leaf has a leading ``[B, ...]`` axis.
        actions : int array ``[B]``.
        rngs    : PRNGKey array ``[B, 2]``.

        Returns
        -------
        (new_states, obs, reward, done) — all with a leading B-axis.
        info is omitted (per-env Python dicts don't vmap; callers can
        attach metadata after the call).
        """
        return jax.vmap(self._step_jit, in_axes=(0, 0, 0))(states, actions, rngs)

    def reset_batched(
        self,
        rngs: jax.Array,
        role: "Role | None" = None,
        race: "Race | None" = None,
        alignment: int = 0,
    ) -> Tuple[EnvState, Dict[str, jax.Array]]:
        """Batched reset over PRNGKeys ``[B, 2]``.

        Uses ``jax.vmap`` over the rng axis in the default ``ParityMode.NLE``
        path — every env runs in parallel on the device.  ``role``/``race``/
        ``alignment`` are static hyperparameters (broadcast to every env).

        Under ``ParityMode.NLE_BYTEPARITY`` the ISAAC64 init in
        :meth:`reset` is host-only (Python ISAAC64 ref impl), so this
        falls back to a Python loop in that mode.
        """
        if use_vendor_rng():
            n = int(rngs.shape[0])
            per_states = []
            per_obs = []
            for i in range(n):
                s_i, o_i = self.reset(rngs[i], role=role, race=race, alignment=alignment)
                per_states.append(s_i)
                per_obs.append(o_i)
            batched_state = jax.tree_util.tree_map(
                lambda *xs: jnp.stack(xs, axis=0), *per_states
            )
            batched_obs = jax.tree_util.tree_map(
                lambda *xs: jnp.stack(xs, axis=0), *per_obs
            )
            return batched_state, batched_obs
        return jax.vmap(
            lambda r: self.reset(r, role=role, race=race, alignment=alignment),
        )(rngs)


def _vendor_draw_prngkey(v_state):
    """Draw a fresh ``jax.random.PRNGKey`` from the ISAAC64 stream.

    Pulls one uint64 via :func:`vendor_rng.next_uint64` and repacks it as the
    uint32[2] layout JAX's Threefry keys use.  Returns ``(new_v_state, key)``.

    This is the shim glue that lets the existing Threefry-based dungeon-gen
    consume an ISAAC64-derived seed without rewriting every ``randint``
    call site.  The downstream PRNG sequence is still Threefry, but its
    starting point is now a function of the ISAAC64 stream — so the same
    ISAAC64 seed always produces the same level layout / monster placement,
    independent of how many Threefry splits happen above the call site.

    Cite: vendor/nethack/src/rnd.c (USE_ISAAC64 path) — every vendor RNG
    draw bottoms out in ``isaac64_next_uint64()``; we consume one such
    word per sub-key.
    """
    v_state, val = _vendor_rng.next_uint64(v_state)
    # Repack uint64 → uint32[2] via pure-JAX bit ops (no host int() shifts).
    # ``next_uint64`` already returns a host-side Python int today, so we
    # box it as a uint64 scalar before slicing.  When the vendor RNG path
    # is itself made traceable, ``val`` will already be a uint64 jnp scalar
    # and this code keeps working unchanged.
    val_u64 = jnp.asarray(val, dtype=jnp.uint64)
    hi = jnp.right_shift(val_u64, jnp.uint64(32)).astype(jnp.uint32)
    lo = jnp.bitwise_and(val_u64, jnp.uint64(0xFFFFFFFF)).astype(jnp.uint32)
    key = jnp.stack([hi, lo]).astype(jnp.uint32)
    return v_state, key


_PET_NEIGHBOUR_OFFSETS = jnp.array(
    [
        [-1, -1], [-1, 0], [-1, 1],
        [ 0, -1],          [ 0, 1],
        [ 1, -1], [ 1, 0], [ 1, 1],
    ],
    dtype=jnp.int32,
)  # [8, 2] — vendor u_init.c::makedog scans the 8 adjacent tiles.


# Vendor enexto_core (teleport.c:126-219) candidate iteration order, encoded
# as (dr, dc) offsets from the centre cell, per ring radius.  Vendor walks
# concentric square rings around (xx, yy) starting at range=1, in two passes
# per ring:
#   Row pass: for x = xmin..xmax: emit (x, ymin) then (x, ymax)
#   Col pass: for y = ymin..ymax-1: emit (xmin, y) then (xmax, y)
# At ring R the unclamped iteration emits 8R+2 slots (2 NW/NE corner DUPs).
# Vendor expands rings until any ring produces a goodpos hit, then picks
# ``good[rn2(count)]``.  MAX_GOOD=15 caps the per-ring candidate list
# (teleport.c:172).  Vendor cite: vendor/nle/src/teleport.c:126-219.
#
# We precompute a [_PET_ENEXTO_MAX_RANGE, _PET_ENEXTO_RING_SIZE, 2] offset
# table once at import time; unused slots are padded with a sentinel that
# falls out-of-bounds.  The pet-spawn body evaluates ALL rings vectorized
# in one pass (820 cells max), then ``jnp.argmax`` selects the first ring
# with goodpos hits — this avoids ``lax.while_loop`` / ``lax.fori_loop``
# which produced non-cacheable HLO (commit 1328967 → reverted; cold compile
# 30+ min, zero cache writes).
_PET_ENEXTO_MAX_RANGE = 10          # vendor practical cap
_PET_ENEXTO_MAX_GOOD = 15           # vendor MAX_GOOD (teleport.c:132)
_PET_ENEXTO_RING_SIZE = 8 * _PET_ENEXTO_MAX_RANGE + 2  # 82 — max slots/ring


def _build_pet_enexto_rings():
    """Precompute the per-ring offset table for vendor enexto_core iteration.

    Returns ``(rings, ring_counts)`` where:
      * ``rings`` is ``int32[_PET_ENEXTO_MAX_RANGE, _PET_ENEXTO_RING_SIZE, 2]``
        with ``rings[R-1, k] = (dr, dc)`` for the k-th candidate of ring R, in
        vendor row-then-col iteration order.  Unused slots are padded with
        a large sentinel that always lands out-of-bounds.
      * ``ring_counts`` is ``int32[_PET_ENEXTO_MAX_RANGE]`` with the actual
        slot count per ring (8R+2).
    """
    import numpy as _np
    _PAD = 32767
    _rings = _np.full(
        (_PET_ENEXTO_MAX_RANGE, _PET_ENEXTO_RING_SIZE, 2), _PAD, dtype=_np.int32
    )
    _counts = _np.zeros((_PET_ENEXTO_MAX_RANGE,), dtype=_np.int32)
    for _R in range(1, _PET_ENEXTO_MAX_RANGE + 1):
        _offs = []
        # Row pass: for x_off in [-R..R]: emit (x_off, -R) then (x_off, R).
        for _x_off in range(-_R, _R + 1):
            _offs.append((-_R, _x_off))   # (dr=-R, dc=x_off)
            _offs.append(( _R, _x_off))   # (dr= R, dc=x_off)
        # Col pass: for y_off in [-R..R-1]: emit (-R, y_off) then (R, y_off).
        for _y_off in range(-_R, _R):
            _offs.append((_y_off, -_R))   # (dr=y_off, dc=-R)
            _offs.append((_y_off,  _R))   # (dr=y_off, dc= R)
        assert len(_offs) == 8 * _R + 2
        for _k, (_dr, _dc) in enumerate(_offs):
            _rings[_R - 1, _k, 0] = _dr
            _rings[_R - 1, _k, 1] = _dc
        _counts[_R - 1] = len(_offs)
    return jnp.asarray(_rings, dtype=jnp.int32), jnp.asarray(_counts, dtype=jnp.int32)


_PET_ENEXTO_RINGS, _PET_ENEXTO_RING_COUNTS = _build_pet_enexto_rings()


def _spawn_starting_pet(state, role: Role, vendor_rng=None):
    """Spawn the role's starting pet adjacent to the player.

    Vendor: vendor/nethack/src/u_init.c::makedog (called from u_init()).
    Pure-JAX so this routine is ``jax.vmap``-safe — the 8-neighbour scan is
    a static-shape ``jnp.where`` selection over the precomputed offsets.
    Pet is placed in slot 5 (after the 5 wild monsters in slots 0-4).

    When ``vendor_rng`` is provided (NLE_BYTEPARITY mode), roles whose
    ``petnum == NON_PM`` consume one ``rn2(2)`` draw from the ISAAC64 stream
    to select kitten (0) vs. little dog (1), matching vendor dog.c:66.
    Returns ``(state, vendor_rng)`` when vendor_rng is not None, else ``state``.

    Citation: vendor/nle/src/dog.c:57-67 (pet_type)
    """
    from Nethax.nethax.constants.monsters import MONSTERS
    from Nethax.nethax.dungeon.spawning import (
        _BASE_AC, _ATK_DICE_N, _ATK_DICE_S, _IS_LARGE, _roll_hp,
    )
    import jax.random as jr

    # Roles with petnum == NON_PM in vendor/nle/src/role.c roles[].
    # dog.c:59-66: if urole.petnum != NON_PM → return petnum; else rn2(2).
    _NON_PM_PET_ROLES = {
        Role.ARCHEOLOGIST, Role.BARBARIAN, Role.HEALER, Role.MONK,
        Role.PRIEST, Role.ROGUE, Role.TOURIST, Role.VALKYRIE,
    }

    # Resolve pet monster name → MONSTERS index.  ``role`` is a Python
    # hyperparameter (not a traced value), so this Python lookup is fine
    # under ``jax.vmap`` over the rng axis.
    #
    # VMAP NOTE: under the NON_PM-pet-role branch the rn2(2) coin flip is
    # routed through ``_vendor_rng.rn2_jax`` (NOT the host-side ``rn2``)
    # so ``flip`` stays a JAX tracer rather than being materialised as a
    # Python int via ``int(flip)``.  ``pet_pm`` then becomes a traced int32
    # selected via ``jnp.where``, and the downstream ``_BASE_AC[pet_pm]`` /
    # ``_ATK_DICE_N[pet_pm]`` / ``_ATK_DICE_S[pet_pm]`` / ``_IS_LARGE[pet_pm]``
    # indexings are all JAX-array gathers that accept traced indices.
    # Kitten and little dog both have ``level=2`` (constants/monster_entries
    # /chunk1.py:323, 599), so ``pet_level`` is a static Python int regardless
    # of the coin flip — no traced-level branch needed.
    _KITTEN_PM = 32      # constants/monster_entries/chunk1.py:595
    _LITTLE_DOG_PM = 16  # constants/monster_entries/chunk1.py:319
    if vendor_rng is not None and role in _NON_PM_PET_ROLES:
        # Consume the rn2(2) coin flip from the vendor ISAAC64 stream.
        # Vendor dog.c:66: return rn2(2) ? PM_KITTEN : PM_LITTLE_DOG;
        # (non-zero → kitten, zero → little dog — same as C truthiness)
        # Cite: vendor/nle/src/dog.c:66
        vendor_rng, flip = _vendor_rng.rn2_jax(vendor_rng, 2)
        pet_pm = jnp.where(
            flip != jnp.int32(0),
            jnp.int32(_KITTEN_PM),
            jnp.int32(_LITTLE_DOG_PM),
        )
        # Both pet candidates share ``level=2`` so the static Python int is
        # bit-identical to the traced selection — keep it static so callers
        # of ``_roll_hp`` outside this branch keep their static-int contract.
        pet_level = 2
    else:
        pet_name = get_starting_pet(role)
        pet_pm = next(
            (i for i, m in enumerate(MONSTERS) if m.name == pet_name),
            32,  # fallback: kitten (index 32)
        )
        pet_level = max(1, int(MONSTERS[pet_pm].level))  # static int.

    # Find an adjacent walkable tile.  Two paths:
    #
    # (A) NLE_BYTEPARITY (vendor_rng is not None): mirror vendor's
    #     enexto_core(range=1) candidate-list construction (10 cells in vendor
    #     iteration order with 2 corner duplicates), then draw ``rn2(num_good)``
    #     from ``vendor_rng`` and index into the dense list of good cells.
    #     Cite: vendor/nle/src/teleport.c:126-216 enexto_core; called from
    #     makemon (makemon.c:1132-1136 ``byyou && !in_mklev``).  ``in_mklev``
    #     is FALSE here because vendor's allmain.c:636 calls ``makedog()`` after
    #     ``mklev()`` has returned.
    #
    # (B) Threefry path (vendor_rng is None): retain the legacy 8-offset
    #     argmax pick so non-parity rollouts keep their existing behaviour.
    player_pos_i32 = state.player_pos.astype(jnp.int32)              # [2]
    terrain = state.terrain[0, 0]                                     # [H, W]
    H, W = terrain.shape

    if vendor_rng is not None:
        # Vendor ``enexto_core`` (teleport.c:126-219): expand concentric square
        # rings around (xx, yy) starting at radius 1; at each ring enumerate
        # the border in row-then-col order, apply ``goodpos``, and stop at the
        # first ring with ≥1 hit — then pick ``good[rn2(count)]``.
        #
        # We materialise ALL rings up to ``_PET_ENEXTO_MAX_RANGE`` in one
        # vectorised pass over the precomputed ``[10, 82, 2]`` offset table.
        # ``jnp.argmax(has_hits)`` then selects the first ring with hits.
        # This is fully flat — no ``lax.while_loop`` / ``lax.fori_loop`` —
        # so the compiled HLO caches cleanly (see commit-after-revert
        # discussion: the loop variant produced non-cacheable HLO).
        #
        # Vendor goodpos (teleport.c:25-106) here reduces to:
        #   isok(x, y) AND not the player cell AND no monster at (x, y) AND
        #   ``accessible(typ) := typ >= DOOR``  (rm.h:92 ACCESSIBLE macro).
        # Vendor coordinate convention: ``x = col``, ``y = row``; we use
        # Nethax ``[row, col]`` throughout, with offsets stored as ``[dr, dc]``.
        # ``xmin = max(1, xx-R)`` excludes col 0 (vendor reserves col 0 for
        # status line); Nethax already drops obs col 0 (commit 307afcb).
        player_r = player_pos_i32[0]
        player_c = player_pos_i32[1]

        # Vectorise candidate positions for ALL rings: [10, 82, 2].
        all_offs = _PET_ENEXTO_RINGS                                   # [R, S, 2]
        all_rr = player_r + all_offs[..., 0]                           # [R, S]
        all_cc = player_c + all_offs[..., 1]                           # [R, S]

        # Mask out pad slots beyond each ring's actual count (8R+2).
        slot_idx = jnp.arange(_PET_ENEXTO_RING_SIZE)[None, :]          # [1, S]
        valid_slot = slot_idx < _PET_ENEXTO_RING_COUNTS[:, None]       # [R, S]

        # isok + vendor xmin = max(1, ...) excludes col 0 (status line).
        in_bounds = (
            (all_rr >= 0) & (all_rr < H) & (all_cc >= 1) & (all_cc < W)
            & valid_slot
        )
        rr_safe = jnp.clip(all_rr, 0, H - 1)
        cc_safe = jnp.clip(all_cc, 0, W - 1)

        tiles = terrain[rr_safe, cc_safe]                              # [R, S]
        # Vendor ``accessible(typ) := typ >= DOOR``.  Nethax TileType codes
        # that map to vendor's accessible enum:
        walkable = (
            (tiles == jnp.int8(TileType.FLOOR))
            | (tiles == jnp.int8(TileType.CORRIDOR))
            | (tiles == jnp.int8(TileType.CLOSED_DOOR))
            | (tiles == jnp.int8(TileType.OPEN_DOOR))
            | (tiles == jnp.int8(TileType.DOORWAY))
            | (tiles == jnp.int8(TileType.STAIRCASE_UP))
            | (tiles == jnp.int8(TileType.STAIRCASE_DOWN))
            | (tiles == jnp.int8(TileType.ALTAR))
            | (tiles == jnp.int8(TileType.FOUNTAIN))
            | (tiles == jnp.int8(TileType.THRONE))
            | (tiles == jnp.int8(TileType.GRAVE))
            | (tiles == jnp.int8(TileType.SINK))
            | (tiles == jnp.int8(TileType.ICE_FLOOR))
        )

        # Vendor goodpos: monster occupancy check (m_at, teleport.c:48).
        # makedog runs AFTER mklev populates wild monsters, so this matters
        # whenever the hero spawn neighbourhood already has a wild monster.
        mai = state.monster_ai
        mon_alive = mai.alive
        mon_pos = mai.pos.astype(jnp.int32)
        mon_r = jnp.clip(mon_pos[:, 0], 0, H - 1)
        mon_c = jnp.clip(mon_pos[:, 1], 0, W - 1)
        occupied = jnp.zeros((H, W), dtype=jnp.bool_)
        occupied = occupied.at[mon_r, mon_c].set(mon_alive, mode="drop")
        no_mon = ~occupied[rr_safe, cc_safe]                           # [R, S]

        not_player = ~((all_rr == player_r) & (all_cc == player_c))    # [R, S]

        good = in_bounds & walkable & no_mon & not_player              # [R, S]

        # Cap each ring to MAX_GOOD in vendor order via per-ring cumsum.
        cum = jnp.cumsum(good.astype(jnp.int32), axis=1)               # [R, S]
        kept = good & (cum <= jnp.int32(_PET_ENEXTO_MAX_GOOD))         # [R, S]
        ring_count = jnp.sum(kept.astype(jnp.int32), axis=1)           # [R]
        has_hits = ring_count > jnp.int32(0)                           # [R]
        any_hits = jnp.any(has_hits)

        # First ring with hits.  ``argmax`` on a bool array returns the
        # smallest index of True; if all False, returns 0 (the fallback path
        # below handles that via ``any_hits``).
        first_ring = jnp.argmax(has_hits.astype(jnp.int32)).astype(jnp.int32)

        kept_ring = kept[first_ring]                                   # [S]
        cands_rr_ring = all_rr[first_ring]                             # [S]
        cands_cc_ring = all_cc[first_ring]                             # [S]
        count = ring_count[first_ring]                                 # int32

        # Vendor: if any hits, draw ``rn2(count)`` ONCE; else fall back to
        # player cell with NO draw (teleport.c:200-218).  Single ``lax.cond``
        # keeps the stream-altering draw inside the success branch.
        def _draw_branch(v):
            v2, idx = _vendor_rng.rn2_jax(v, jnp.maximum(count, jnp.int32(1)))
            cum_r = jnp.cumsum(kept_ring.astype(jnp.int32))            # [S]
            match = (cum_r - jnp.int32(1) == idx) & kept_ring          # [S]
            pick = jnp.argmax(match.astype(jnp.int32)).astype(jnp.int32)
            return v2, jnp.stack([cands_rr_ring[pick], cands_cc_ring[pick]])

        def _no_draw(v):
            return v, player_pos_i32

        vendor_rng, pet_pos_i32_picked = jax.lax.cond(
            any_hits, _draw_branch, _no_draw, vendor_rng,
        )
        pet_pos = pet_pos_i32_picked.astype(jnp.int16)                 # [2]
    else:
        # Threefry path: static-shape 8 candidate offsets, pure-JAX argmax.
        cands = player_pos_i32[None, :] + _PET_NEIGHBOUR_OFFSETS      # [8, 2]
        rr = cands[:, 0]
        cc = cands[:, 1]
        in_bounds = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        rr_safe = jnp.clip(rr, 0, H - 1)
        cc_safe = jnp.clip(cc, 0, W - 1)
        tiles = terrain[rr_safe, cc_safe]
        walkable = (
            (tiles == jnp.int8(TileType.FLOOR))
            | (tiles == jnp.int8(TileType.CORRIDOR))
        )
        ok = in_bounds & walkable                                     # [8]
        any_ok = jnp.any(ok)
        pick = jnp.argmax(ok.astype(jnp.int32))
        pet_pos = jnp.where(
            any_ok,
            cands[pick],
            player_pos_i32,
        ).astype(jnp.int16)                                           # [2]

    # Roll HP using the same formula as wild monsters (makemon.c::newmonhp).
    # ``_roll_hp`` is pure-JAX; store the traced scalar directly (no int()
    # cast — that would force concretisation and break vmap).
    dummy_rng = jr.PRNGKey(0)
    hp_val = _roll_hp(dummy_rng, jnp.int32(pet_level))

    # Pet slot = first free slot AFTER all live wild monsters, so it lands
    # on top of the vendor fmon LIFO stack.  Vendor `makedog` runs after
    # `mklev` (allmain.c:813-820), and each `makemon` prepends to fmon
    # (NLE 3.x: vendor/nle/src/makemon.c — `mtmp->nmon = fmon; fmon = mtmp`),
    # so the pet ends up at the head of the list.  We mirror that with
    # `pet_slot = n_wild` and build `fmon_order` below.  Cite Agent F's
    # fmon-order audit.
    n_wild = jnp.sum(state.monster_ai.alive.astype(jnp.int32))
    PET_SLOT = n_wild
    pm_i16 = jnp.int16(pet_pm)
    mai = state.monster_ai.replace(
        alive=state.monster_ai.alive.at[PET_SLOT].set(True),
        tame=state.monster_ai.tame.at[PET_SLOT].set(True),
        peaceful=state.monster_ai.peaceful.at[PET_SLOT].set(True),
        mtame=state.monster_ai.mtame.at[PET_SLOT].set(jnp.int8(10)),
        entry_idx=state.monster_ai.entry_idx.at[PET_SLOT].set(pm_i16),
        pos=state.monster_ai.pos.at[PET_SLOT].set(pet_pos),
        hp=state.monster_ai.hp.at[PET_SLOT].set(hp_val.astype(jnp.int32)),
        hp_max=state.monster_ai.hp_max.at[PET_SLOT].set(hp_val.astype(jnp.int32)),
        ac=state.monster_ai.ac.at[PET_SLOT].set(_BASE_AC[pet_pm]),
        is_large=state.monster_ai.is_large.at[PET_SLOT].set(_IS_LARGE[pet_pm]),
        attack_dice_n=state.monster_ai.attack_dice_n.at[PET_SLOT].set(
            _ATK_DICE_N[pet_pm].astype(jnp.int8)
        ),
        attack_dice_sides=state.monster_ai.attack_dice_sides.at[PET_SLOT].set(
            _ATK_DICE_S[pet_pm].astype(jnp.int8)
        ),
    )
    # Build the vendor fmon LIFO iteration order:
    #   fmon_order[0]      = pet_slot                (newest, prepended last)
    #   fmon_order[1..n_wild] = n_wild-1, n_wild-2, ..., 0  (wilds newest→oldest)
    #   fmon_order[n_wild+1..]  = -1                  (empty)
    # This is the order vendor `for (mtmp=fmon; mtmp; mtmp=mtmp->nmon)` walks.
    from Nethax.nethax.subsystems.monster_ai import MAX_MONSTERS_PER_LEVEL as _MAX
    k_arr = jnp.arange(_MAX, dtype=jnp.int32)
    fmon_order = jnp.where(
        k_arr == jnp.int32(0),
        PET_SLOT.astype(jnp.int32),
        jnp.where(
            (k_arr >= jnp.int32(1)) & (k_arr <= n_wild),
            (n_wild - k_arr).astype(jnp.int32),
            jnp.int32(-1),
        ),
    )
    mai = mai.replace(fmon_order=fmon_order)
    new_state = state.replace(monster_ai=mai)
    if vendor_rng is not None:
        return new_state, vendor_rng
    return new_state


def _tick_stinking_cloud(state):
    """Per-turn stinking-cloud effect on the hero.

    Cite: vendor/nethack/src/region.c::inside_gas_cloud (called from
    run_regions when the player tile is inside the gas-cloud rectangle).
    On each turn while ``cloud_turns > 0``:
      * if hero is within Chebyshev radius of cloud_pos, apply 1 HP and
        bump VOMITING timer by rnd(1, 3) (vendor inside_gas_cloud's
        ``losehp(1, ...)`` + ``set_property(VOMITING, ...)``)
      * decrement cloud_turns

    Scroll of stinking cloud (items_scrolls SCR_STINKING_CLOUD) writes
    the scalar fields directly; this is the matching tick that vendor's
    region machinery would otherwise drive.
    """
    from Nethax.nethax.subsystems.status_effects import TimedStatus as _TS

    turns = state.cloud_turns.astype(jnp.int32)
    active = turns > jnp.int32(0)
    pr = state.player_pos[0].astype(jnp.int32)
    pc = state.player_pos[1].astype(jnp.int32)
    cr = state.cloud_pos[0].astype(jnp.int32)
    cc = state.cloud_pos[1].astype(jnp.int32)
    dr = jnp.abs(pr - cr)
    dc = jnp.abs(pc - cc)
    cheby = jnp.maximum(dr, dc)
    inside = active & (cheby <= state.cloud_radius.astype(jnp.int32))

    new_hp = jnp.where(
        inside,
        jnp.maximum(state.player_hp - jnp.int32(1), jnp.int32(0)),
        state.player_hp,
    ).astype(state.player_hp.dtype)
    new_done = state.done | (new_hp <= jnp.int32(0))

    ts = state.status.timed_statuses
    cur_vom = ts[int(_TS.VOMITING)].astype(jnp.int32)
    new_vom = jnp.where(inside, cur_vom + jnp.int32(2), cur_vom)
    new_ts = ts.at[int(_TS.VOMITING)].set(new_vom.astype(ts.dtype))
    new_status = state.status.replace(timed_statuses=new_ts)

    new_turns = jnp.where(
        active,
        (turns - jnp.int32(1)).astype(state.cloud_turns.dtype),
        state.cloud_turns,
    )

    return state.replace(
        player_hp=new_hp,
        done=new_done,
        status=new_status,
        cloud_turns=new_turns,
    )


def _step_impl(state, action, rng):
    """JIT-compatible inner body of NethaxEnv.step.

    Vendor order (vendor/nethack/src/allmain.c::moveloop, lines ~200-360):
      1. Player action / docmd                       (line 203: svc.context.move)
      2. Monster turn / movemon                      (line 212)
      3. Turn counter / svm.moves++                  (line 244)
      4. Status timers / nh_timeout                  (line 273)
         (also covers HP regen line 294 / Pw regen line 305)
      5. Were-creature transformation / lycanthropy  (lines 322-339)
      6. age_spells — spell memory decay             (line 355)
      7. Shopkeeper tick (pay-at-exit + pursuit)     (shk.c via moveloop)
      8. Endgame / ascension check                   (allmain.c done() paths)

    Notes:
      * lit_radius_until_turn is an absolute-deadline timestamp (set to
        timestep+100 by SPELL_LIGHT — see magic.py::_effect_light), so the
        "decrement per turn" semantics fall out automatically as
        timestep increases; no separate decrement call is needed here.
        Cite: vendor/nethack/src/light.c::do_light_sources.
    """
    rng_act, rng_monsters, rng_status, rng_poly, rng_shop, rng_swallow, rng_explvl, rng_regions, rng_astral = jax.random.split(rng, 9)
    # NLE-action-index → ASCII-ord remap.  An NLE-trained policy emits
    # ``action`` as an index into ``env.actions`` (86-entry USEFUL_ACTIONS);
    # Nethax's dispatch table expects the ASCII ord.  ``_maybe_remap_action``
    # auto-detects: ``action < 86`` → gather from USEFUL_ACTIONS;
    # ``action >= 86`` → pass through (caller already supplied an ord).
    # JIT-pure via jnp.where, so it survives jax.jit / vmap.
    # Cite: vendor/nle/nle/env/base.py:359 — same remap NLE does numpy-side.
    action = _maybe_remap_action(action)
    already_done = state.done

    # Pre-step snapshot: was the Wizard of Yendor alive?  Used to fire
    # intervene() once on Wizard kill (vendor wizard.c::intervene 784-810).
    _PM_WIZARD_ENTRY = jnp.int32(281)
    prev_wizard_alive = jnp.any(
        state.monster_ai.alive
        & (state.monster_ai.entry_idx.astype(jnp.int32) == _PM_WIZARD_ENTRY)
    )

    # Pre-step (branch, level) snapshot — used by the Astral-Plane mplayer
    # seeder (vendor mplayer.c::create_mplayers).  Edge-triggered: spawn 3
    # mplayers when the player transitions onto (Branch.ENDGAME, level 5).
    prev_branch = state.dungeon.current_branch.astype(jnp.int32)
    prev_level  = state.dungeon.current_level.astype(jnp.int32)

    def _do_step(_):
        # Pre-1. Clear message buffer — vendor topl.c / winrl.cc:353 zeros
        # obs->message when ttyDisplay->toplin==0 (no new pline this turn).
        # Without this the welcome line persists across every step.
        # Cite: vendor/nle/win/rl/winrl.cc line 353 — std::memset(obs->message,
        #       0, NLE_MESSAGE_SIZE) when toplin is false at obs-fill time.
        # Also reset the per-step "did this action consume a game turn" flag to
        # True; handlers (e.g. a blocked wall-bump in action_dispatch
        # ``_move_branch``) clear it to False when the action took zero game
        # time, which gates the game_moves (BL_TIME) increment below.
        ns0 = state.replace(
            messages=_clear_message(state.messages),
            action_consumed_turn=jnp.bool_(True),
        )

        # 1. Player action — allmain.c line 203 (svc.context.move).
        ns = dispatch_action(ns0, action, rng_act)

        # 1a. Astral-Plane mplayer trigger — vendor mplayer.c::create_mplayers
        #     (lines 327-355) called from astral.lua MAP section on level
        #     entry.  Edge-triggered on (prev != Astral) → (curr == Astral).
        ns = _maybe_seed_astral_mplayers(ns, rng_astral, prev_branch, prev_level)

        # 1b. Digging tick — advance multi-turn pickaxe dig (dig.c::dodig).
        ns = _dig_tick(ns, rng_act)

        # 2. Monster turn — allmain.c line 212 (movemon).
        # Vendor moveloop only enters the per-turn block (movemon + mcalcmove
        # + maintenance) when "actual time passed" — i.e. the player's action
        # consumed a turn.  A wall-bump returns early from domove without
        # decrementing ``youmonst.movement``, so the do-while at allmain.c:103
        # short-circuits and monsters do not act.  Gate the call on
        # ``action_consumed_turn`` so wall-bump turns leave monsters AND the
        # vendor_rng stream untouched.  Without this gate Nethax monsters
        # drift relative to vendor on every non-time-consuming step.
        # Cite: vendor/nle/src/allmain.c lines 88-112 (do-while turn loop).
        ns = jax.lax.cond(
            ns.action_consumed_turn,
            lambda s: _monster_ai_step(s, rng_monsters),
            lambda s: s,
            ns,
        )

        # 2b. Per-turn region tick — vendor/nethack/src/region.c::run_regions
        #     (line 414).  Ages every active region by 1 and applies
        #     gas-cloud damage to the player when they stand inside one.
        ns = _run_regions(ns, rng_regions)

        # 2c. Stinking-cloud tick — vendor/nethack/src/region.c::inside_gas_cloud
        #     (called from run_regions when player tile is inside).  Scroll of
        #     stinking cloud writes scalar cloud_pos/radius/turns directly to
        #     EnvState (items_scrolls SCR_STINKING_CLOUD); decrement turns and
        #     when the player is within Chebyshev radius, apply 1 HP damage +
        #     extend VOMITING timer per vendor inside_gas_cloud (region.c:1100+).
        ns = _tick_stinking_cloud(ns)

        # 3. Increment turn counter — allmain.c line 244 (svm.moves++).
        #    ``timestep`` is the monotonic env-step clock (timer deadlines,
        #    hallucination hash, monster-AI cadence) and advances every env
        #    step regardless of whether the action consumed game time.
        ns = ns.replace(timestep=ns.timestep + jnp.int32(1))
        #    ``game_moves`` is the vendor ``moves`` turn counter (BL_TIME
        #    source).  Vendor only reaches ``svm.moves++`` in moveloop for a
        #    time-consuming turn; a blocked move (wall-bump) returns early from
        #    domove without ticking it.  Gate on the per-step flag the action
        #    handlers set, so a wall-bump leaves game_moves unchanged.
        ns = ns.replace(
            game_moves=ns.game_moves
            + jnp.where(ns.action_consumed_turn, jnp.int32(1), jnp.int32(0))
        )

        # 4. Status-effect tick — allmain.c line 273 (nh_timeout),
        #    inclusive of regen_hp (line 294) and regen_pw (line 305).
        #    Pre-tick: emit HALLUCINATION expiry message (vendor timeout.c
        #    HALLU case lines 778-783 — make_hallucinated(0L, TRUE, 0L) →
        #    "Everything looks SO boring now.").
        ns = _tick_hallu_expiry(ns)
        # Stoning is cancelled silently on the "turning into slime" tick
        # (slime_dialogue i==1 branch, timeout.c lines 436-440).  Must run
        # BEFORE the timer decrement in _status_step so we read the
        # pre-decrement SLIMED value.
        ns = _tick_slime_cancels_stoning(ns)
        new_status, new_hp, new_pw, new_done = _status_step(
            ns.status,
            rng_status,
            ns.player_hp,
            ns.player_hp_max,
            ns.player_pw,
            ns.player_pw_max,
            ns.player_xl,
            ns.player_role,
            ns.done,
        )
        ns = ns.replace(
            status=new_status,
            player_hp=new_hp,
            player_pw=new_pw,
            done=new_done,
        )

        # 4a-riding.  Per-turn riding ticks: decrement u.ugallop (gallop
        # counter) and apply 1/100-chance saddle wear.  Both gated internally
        # on player_steed_mid != 0.
        # Cite: vendor/nethack/src/timeout.c lines 664-667 (ugallop--);
        #       vendor/nethack/src/steed.c saddle wear (implicit).
        ns = _tick_gallop(ns)
        ns = _tick_saddle(ns)

        # 4a0. Luck drift toward baseluck=0 every 300/600 moves (300 if the
        #    hero is carrying the Amulet of Yendor or god_anger>0, else 600).
        #    Cite: vendor/nethack/src/timeout.c lines 606-620 — nh_timeout
        #    luck-decay block.  No-op when |Luck|==baseluck or off-cadence.
        ns = _tick_luck_drift(ns)

        # 4a1. Generic timer queue drain (vendor timeout.c::run_timers
        #    lines 2222-2245).  Fires any timer whose fire_turn <= current
        #    timestep, then clears the slot.  Wave 47f scaffolding —
        #    callbacks are currently no-ops awaiting per-consumer wiring.
        ns = _tick_timer_queue(ns)

        # 4a2. Multi-turn occupation tick (vendor ga.afternmv invocation
        #    when gm.multi reaches 0).  Decrements occupation_remaining;
        #    fires the matching callback (e.g. STEAL_ARM) when it hits
        #    zero.  Wave 47g scaffolding.
        ns = _tick_occupation(ns)

        # 4a3. Ball-as-trap-escape (vendor ball.c::drop_ball lines 882-961).
        # When the hero is punished AND in a trap, vendor's drop_ball
        # mechanic lets the iron ball "drop" forcefully and break the
        # trap with a 1/4 chance per turn.  Per vendor:927-953 the ball
        # can break a bear trap, dispel a web, or fill in a pit when it
        # lands on the player's tile.
        from Nethax.nethax.subsystems.status_effects import TimedStatus as _TS_ball
        rng_ball_esc, _rng_after = jax.random.split(jax.random.fold_in(rng_status, jnp.int32(0xBA77)), 2)
        _ball_roll = jax.random.randint(rng_ball_esc, (), 0, 4, dtype=jnp.int32)
        _can_drop_escape = ns.is_punished & ns.player_in_trap & (_ball_roll == jnp.int32(0))
        # On successful drop-escape: clear trap; ball lands at player.
        ns = ns.replace(
            player_in_trap=jnp.where(_can_drop_escape, jnp.bool_(False), ns.player_in_trap),
            player_trap_timer=jnp.where(_can_drop_escape, jnp.int16(0), ns.player_trap_timer),
            ball_pos=jnp.where(_can_drop_escape, ns.player_pos, ns.ball_pos),
            ball_thrown_pos=jnp.where(_can_drop_escape, ns.player_pos, ns.ball_thrown_pos),
            ball_thrown_turns=jnp.where(_can_drop_escape, jnp.int8(1), ns.ball_thrown_turns),
        )
        # Decay the ball_thrown_turns counter for landed projectiles.
        ns = ns.replace(
            ball_thrown_turns=jnp.maximum(
                ns.ball_thrown_turns - jnp.int8(1), jnp.int8(0)
            ).astype(jnp.int8),
        )

        # 4a. Experience-level check — vendor exper.c::newexplevel called from
        #    allmain.c (after nh_timeout / before the next turn).  Promotes
        #    ulevel when uexp crosses the next newuexp(ulevel) threshold.
        ns = _newexplevel(ns, rng_explvl)

        # 4b. Swallow/engulf digestion tick — vendor/nethack/src/mhitu.c:1418.
        ns = _digest_tick(ns, rng_swallow)

        # 5. Were-creature / polymorph timer tick — allmain.c lines 322-339
        #    (mvl_change handling).  polymorph.step decrements both
        #    poly_timer and lycanthropy_timer.
        ns = _polymorph_step(ns, rng_poly)

        # 6. age_spells — vendor/nethack/src/spell.c::age_spells (called
        #    from allmain.c line 355).  Decrement every spell_memory > 0
        #    by 1, or by 2 when the hero is Confused (vendor doubles the
        #    decay under confusion — Wave 47i parity).
        from Nethax.nethax.subsystems.status_effects import TimedStatus as _TS_spell
        magic = ns.magic
        is_confused = ns.status.timed_statuses[int(_TS_spell.CONFUSION)] > jnp.int32(0)
        decrement = jnp.where(is_confused, jnp.int32(2), jnp.int32(1))
        new_mem = jnp.maximum(magic.spell_memory - decrement, jnp.int32(0))
        ns = ns.replace(magic=magic.replace(spell_memory=new_mem))

        # 6b. Calendar tick — cycle moonphase every 250 turns (Wave 47i
        # approximation of vendor calendar.c).  Mirrors flags.moonphase
        # cycle (new→waxing→full→waning).  Used by luck-drift baseluck
        # and were_change rate.
        moon_advance = (ns.timestep % jnp.int32(250)) == jnp.int32(0)
        new_moon = jnp.where(
            moon_advance,
            (ns.calendar_moonphase.astype(jnp.int32) + jnp.int32(1)) % jnp.int32(4),
            ns.calendar_moonphase.astype(jnp.int32),
        ).astype(jnp.int8)
        ns = ns.replace(calendar_moonphase=new_moon)

        # 7. Shop tick — Wave 6 #47 (pay-at-exit + angry shopkeeper pursuit).
        #    Vendor: shk.c invoked from moveloop's per-turn block.
        ns = _shop_step(ns, rng_shop)

        # 8. Ascension / endgame check — vendor allmain.c done() paths.
        ns = maybe_ascend(ns)

        # 8a. intervene() — post-Wizard-kill harassment.
        # Vendor wizard.c::intervene lines 784-810 picks rn2(6):
        #   0,1 → "You feel vaguely nervous."  (no gameplay effect)
        #   2   → rndcurse — random curse on player items
        #   3   → aggravate — wake all sleeping monsters
        #   4   → nasty — summon a high-difficulty monster (placeholder)
        #   5   → resurrect — re-spawn the Wizard (placeholder)
        # Fires once when the Wizard transitions alive → dead this turn.
        wiz_now = jnp.any(
            ns.monster_ai.alive
            & (ns.monster_ai.entry_idx.astype(jnp.int32) == _PM_WIZARD_ENTRY)
        )
        wiz_just_died = prev_wizard_alive & ~wiz_now

        rng_iv, rng_iv2 = jax.random.split(rng_status, 2)
        which = jax.random.randint(rng_iv, (), 0, 6, dtype=jnp.int32)

        # Effect 2: rndcurse — flip BUC on one random inventory slot to cursed(1).
        is_curse = wiz_just_died & (which == jnp.int32(2))
        items = ns.inventory.items
        n_slots = items.category.shape[0]
        cur_slot = jax.random.randint(rng_iv2, (), 0, n_slots, dtype=jnp.int32)
        occupied = items.category[cur_slot] != jnp.int8(0)
        new_buc = jnp.where(is_curse & occupied, jnp.int8(1), items.buc_status[cur_slot])
        new_items = items.replace(buc_status=items.buc_status.at[cur_slot].set(new_buc))

        # Effect 3: aggravate — wake all sleeping monsters.
        is_aggr = wiz_just_died & (which == jnp.int32(3))
        mai = ns.monster_ai
        new_asleep = jnp.where(is_aggr, jnp.zeros_like(mai.asleep), mai.asleep)
        new_sleep_t = jnp.where(
            is_aggr, jnp.zeros_like(mai.sleep_timer), mai.sleep_timer
        )

        # Effect 4: nasty — summon a high-difficulty monster at first
        # dead monster slot.  Vendor nasty() picks from a 44-species
        # nasties[] pool; this expanded set covers a representative
        # 10-monster slice across demons/devils/giant beasts:
        #   297 water demon     299 horned devil    300 erinys
        #   301 barbed devil    302 marilith        307 nalfeshnee
        #   150 red dragon       49 master mind flayer
        #   182 jabberwock      234 vampire lord
        is_nasty = wiz_just_died & (which == jnp.int32(4))
        rng_iv3, rng_iv4 = jax.random.split(rng_iv2, 2)
        _NASTY_POOL = jnp.array(
            [297, 299, 300, 301, 302, 307, 150, 49, 182, 234], dtype=jnp.int16
        )
        nasty_pick_roll = jax.random.randint(
            rng_iv3, (), 0, _NASTY_POOL.shape[0], dtype=jnp.int32
        )
        nasty_entry = _NASTY_POOL[nasty_pick_roll]
        dead_slots = ~new_asleep & ~mai.alive   # logic-only: use alive only
        dead_mask = ~mai.alive
        any_dead = jnp.any(dead_mask)
        dead_idx = jnp.argmax(dead_mask.astype(jnp.int32)).astype(jnp.int32)
        do_nasty = is_nasty & any_dead

        spawn_pos = state.player_pos  # Nasty spawns near hero (vendor enexto)
        nasty_alive = jnp.where(do_nasty, jnp.bool_(True),  mai.alive[dead_idx])
        nasty_entry_v = jnp.where(do_nasty, nasty_entry, mai.entry_idx[dead_idx])
        nasty_hp     = jnp.where(do_nasty, jnp.int32(20), mai.hp[dead_idx])
        nasty_hpmax  = jnp.where(do_nasty, jnp.int32(20), mai.hp_max[dead_idx])

        # Effect 5: resurrect — re-spawn the Wizard at first dead slot.
        is_resurr = wiz_just_died & (which == jnp.int32(5))
        do_resurr = is_resurr & any_dead
        resurr_alive = jnp.where(do_resurr, jnp.bool_(True), nasty_alive)
        resurr_entry = jnp.where(do_resurr, jnp.int16(281), nasty_entry_v)
        resurr_hp    = jnp.where(do_resurr, jnp.int32(50), nasty_hp)
        resurr_hpmax = jnp.where(do_resurr, jnp.int32(50), nasty_hpmax)

        new_alive_arr  = mai.alive.at[dead_idx].set(resurr_alive)
        new_entry_arr  = mai.entry_idx.at[dead_idx].set(resurr_entry)
        new_hp_arr     = mai.hp.at[dead_idx].set(resurr_hp)
        new_hpmax_arr  = mai.hp_max.at[dead_idx].set(resurr_hpmax)

        ns = ns.replace(
            inventory=ns.inventory.replace(items=new_items),
            monster_ai=mai.replace(
                asleep=new_asleep,
                sleep_timer=new_sleep_t,
                alive=new_alive_arr,
                entry_idx=new_entry_arr,
                hp=new_hp_arr,
                hp_max=new_hpmax_arr,
            ),
        )
        # Effects 0/1 remain no-ops (vendor's "You feel nervous." flavor).
        return ns

    new_state = jax.lax.cond(already_done, lambda _: state, _do_step, operand=None)

    # Drain DISP for the per-step obs draws (display.c:486-498 glyph
    # selection + winrl.cc:458 inv glyph emit) — see env.reset() for
    # rationale.  Host-side ``use_vendor_rng()`` flag selects the branch
    # at JIT trace time, so this is a compile-time no-op under default
    # ParityMode.NLE and a fixed-shape scan under NLE_BYTEPARITY.
    if use_vendor_rng():
        from Nethax.nethax.obs.nle_obs import consume_disp_for_obs as _consume_disp
        new_state = new_state.replace(vendor_rng_disp=_consume_disp(new_state))
    obs = build_nle_observation(new_state)
    # Reward = score delta (NLE convention: vendor topten.c::u.urexp running
    # accumulator, surfaced as bl_score in blstats).  Already-done steps
    # contribute 0 since new_state == state.
    reward = jnp.float32(new_state.scoring.score - state.scoring.score)
    return new_state, obs, reward, new_state.done
