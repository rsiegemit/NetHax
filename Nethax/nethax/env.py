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
            v_state = _vendor_rng.init(int(seed_arr))

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
            disp_seed_val = int(seed_arr) if disp_seed is None else int(disp_seed)
            v_state_disp = _vendor_rng.init(disp_seed_val)
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

            v_state, rng_level = _vendor_draw_prngkey(v_state)
            v_state, rng_char = _vendor_draw_prngkey(v_state)
            v_state, rng_monsters = _vendor_draw_prngkey(v_state)

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

        # Apply character creation (stats, inventory, AC).
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
        # Mirrors vendor/nethack/src/allmain.c::welcome lines 920-922 ::
        #     pline("%s %s, welcome to NetHack!  You are a%s.",
        #           Hello(), svp.plname, buf);
        # The greeting prefix (Hello / Salutations / Konnichi wa / Aloha /
        # Velkommen) is selected by role.c::Hello.  Without this the
        # validator sees ``tty_chars row 0`` as blank while NLE shows the
        # per-role welcome line.
        from Nethax.nethax.subsystems.messages import emit_role_intro as _emit_role_intro
        state = state.replace(
            messages=_emit_role_intro(state.messages, int(role)),
        )

        # Consume the fixed ~18 ISAAC64 draws of vendor init_dungeons.
        # Vendor sequence (allmain.c:610): init_dungeons fires AFTER u_init
        # and BEFORE mklev.  The 18 fixed draws are:
        #   4 × dungeon depth rn1 (dungeon.c:796-798)
        #   1 × Fort Ludios chance gate rn2(100) (dungeon.c:775-776)
        #   8 × RNDLEVEL chance gates rn2(100) (dungeon.c:548)
        #   5 × tune rn2(7) (dungeon.c:917-918)
        # Citation: vendor/nle/src/dungeon.c:714 init_dungeons.
        if use_vendor_rng():
            new_vrng, _dungeon_state = consume_init_dungeons_draws(state.vendor_rng)
            # Consume the variable-count draws that follow the 18 fixed draws:
            # place_level slot picks (dungeon.c:661) and parent_dlevel branch
            # picks (dungeon.c:398), interleaved per dungeon in parse order.
            # Citation: vendor/nle/src/dungeon.c:398, 502, 661, 772-913.
            new_vrng, _var_state = consume_init_dungeons_variable_draws(
                new_vrng, _dungeon_state
            )
            state = state.replace(vendor_rng=new_vrng)

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

        # Populate level 1 with monsters after dungeon gen.  Under
        # NLE_BYTEPARITY, thread the Isaac64State so per-monster HP rolls
        # (newmonhp d(mlvl,8)) consume from the ISAAC64 stream and the
        # updated state is written back into ``state.vendor_rng``.
        if use_vendor_rng():
            state = populate_level_with_monsters(
                state, rng_monsters, n_monsters=n_monsters_vendor,
                vendor_rng=state.vendor_rng,
            )
        else:
            state = populate_level_with_monsters(
                state, rng_monsters, n_monsters=n_monsters_vendor,
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
        state = _spawn_starting_pet(state, role)

        # Seed the explored mask via FOV so the player can see their starting
        # room on the very first frame.  Without this the initial obs is all
        # NO_GLYPH and the UI shows an empty screen.
        from Nethax.nethax.fov import compute_fov
        vis = compute_fov(
            state.terrain[0, 0, :, :],
            state.player_pos.astype(jnp.int32),
        )                                                  # bool[MAP_H, MAP_W]
        new_explored = state.explored.at[0, 0].set(
            state.explored[0, 0] | vis
        )
        state = state.replace(explored=new_explored)

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


def _spawn_starting_pet(state, role: Role):
    """Spawn the role's starting pet adjacent to the player.

    Vendor: vendor/nethack/src/u_init.c::makedog (called from u_init()).
    Pure-JAX so this routine is ``jax.vmap``-safe — the 8-neighbour scan is
    a static-shape ``jnp.where`` selection over the precomputed offsets.
    Pet is placed in slot 5 (after the 5 wild monsters in slots 0-4).
    """
    from Nethax.nethax.constants.monsters import MONSTERS
    from Nethax.nethax.dungeon.spawning import (
        _BASE_AC, _ATK_DICE_N, _ATK_DICE_S, _IS_LARGE, _roll_hp,
    )
    import jax.random as jr

    # Resolve pet monster name → MONSTERS index.  ``role`` is a Python
    # hyperparameter (not a traced value), so this Python lookup is fine
    # under ``jax.vmap`` over the rng axis.
    pet_name = get_starting_pet(role)
    pet_pm = next(
        (i for i, m in enumerate(MONSTERS) if m.name == pet_name),
        32,  # fallback: kitten (index 32)
    )
    pet_level = max(1, int(MONSTERS[pet_pm].level))  # also a static int.

    # Find an adjacent FLOOR or CORRIDOR tile (Chebyshev distance == 1).
    # Static-shape: 8 candidate offsets, pure-JAX gather + argmax.
    player_pos_i32 = state.player_pos.astype(jnp.int32)              # [2]
    cands = player_pos_i32[None, :] + _PET_NEIGHBOUR_OFFSETS          # [8, 2]
    terrain = state.terrain[0, 0]                                     # [H, W]
    H, W = terrain.shape
    rr = cands[:, 0]
    cc = cands[:, 1]
    in_bounds = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
    # Clamp before indexing so out-of-bounds rows don't error (their mask
    # bit is False anyway and they're discarded).
    rr_safe = jnp.clip(rr, 0, H - 1)
    cc_safe = jnp.clip(cc, 0, W - 1)
    tiles = terrain[rr_safe, cc_safe]
    walkable = (
        (tiles == jnp.int8(TileType.FLOOR))
        | (tiles == jnp.int8(TileType.CORRIDOR))
    )
    ok = in_bounds & walkable                                         # [8]
    # First-True argmax — when no neighbour is walkable, fall back to the
    # player's own tile (vendor fallback in makedog when no slot is free).
    any_ok = jnp.any(ok)
    pick = jnp.argmax(ok.astype(jnp.int32))
    pet_pos = jnp.where(
        any_ok,
        cands[pick],
        player_pos_i32,
    ).astype(jnp.int16)                                               # [2]

    # Roll HP using the same formula as wild monsters (makemon.c::newmonhp).
    # ``_roll_hp`` is pure-JAX; store the traced scalar directly (no int()
    # cast — that would force concretisation and break vmap).
    dummy_rng = jr.PRNGKey(0)
    hp_val = _roll_hp(dummy_rng, jnp.int32(pet_level))

    # Write pet into slot 5 (first slot after the 5 wild monsters).
    PET_SLOT = 5
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
            _ATK_DICE_N[pet_pm]
        ),
        attack_dice_sides=state.monster_ai.attack_dice_sides.at[PET_SLOT].set(
            _ATK_DICE_S[pet_pm]
        ),
    )
    return state.replace(monster_ai=mai)


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
        # 1. Player action — allmain.c line 203 (svc.context.move).
        ns = dispatch_action(state, action, rng_act)

        # 1a. Astral-Plane mplayer trigger — vendor mplayer.c::create_mplayers
        #     (lines 327-355) called from astral.lua MAP section on level
        #     entry.  Edge-triggered on (prev != Astral) → (curr == Astral).
        ns = _maybe_seed_astral_mplayers(ns, rng_astral, prev_branch, prev_level)

        # 1b. Digging tick — advance multi-turn pickaxe dig (dig.c::dodig).
        ns = _dig_tick(ns, rng_act)

        # 2. Monster turn — allmain.c line 212 (movemon).
        ns = _monster_ai_step(ns, rng_monsters)

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
        ns = ns.replace(timestep=ns.timestep + jnp.int32(1))

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
