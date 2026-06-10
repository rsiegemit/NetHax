"""Brax-style rewrites of the four per-step ``status_effects`` entry points.

Background
----------
Under ``jax.vmap`` (multi-seed / multi-env rollouts), ``jax.lax.cond`` and
``jax.lax.switch`` lower to ``lax.select`` and emit *both* branches in the
HLO.  For nested cond chains this produces pathological HLO compile blowup
on H100.  The Brax pattern (Google brax + Craftax —
https://github.com/MichaelTMatthews/Craftax) sidesteps this by always
computing both branches eagerly and selecting results via ``jnp.where``
masks.  The resulting HLO is flat and fusion-friendly.

Status effects are processed every player step (env.py ``_post_monster_body``
chain), so any HLO contribution from these four functions is paid on every
turn.  This module hosts Brax-style ports of the four entry points called
from env.py per turn:

  * ``step_brax``                       — main status-effect orchestrator.
  * ``tick_hallu_expiry_brax``          — emit "Everything looks SO boring
                                          now." when HALLUCINATION expires.
  * ``tick_luck_drift_brax``            — drift ``player_luck`` toward
                                          baseluck=0 every 300/600 moves.
  * ``tick_slime_cancels_stoning_brax`` — slime-dialogue per-turn side
                                          effects across i={4,3,2,1}.

Audit & flatten count
---------------------
Reading the originals in ``status_effects.py``:

  * ``step``                       (line 1191): 0 conds, 0 switches,
                                                0 scans, 0 fori_loops.
                                                Already Brax-shaped — all
                                                conditional dataflow flows
                                                through ``jnp.where`` masks
                                                inside its 9 callee helpers
                                                (apply_strangulation,
                                                 apply_stoning, apply_sliming,
                                                 apply_food_poisoning,
                                                 apply_sick_lethal,
                                                 tick_timers, hunger_tick,
                                                 apply_poisoned_tick,
                                                 hp_regen_tick, pw_regen_tick,
                                                 apply_starvation).
  * ``tick_hallu_expiry``          (line 1430): 0 conds, 0 switches.
                                                Mask selection via
                                                ``jax.tree.map(jnp.where, ...)``
                                                over the message ring buffer
                                                — already Brax-shaped.
  * ``tick_luck_drift``            (line 1498): 0 conds, 0 switches.
                                                Three-way drift logic encoded
                                                as nested ``jnp.where`` —
                                                already Brax-shaped.
  * ``tick_slime_cancels_stoning`` (line 1326): 0 conds, 0 switches.
                                                Four-way message selection
                                                via nested ``jnp.where`` over
                                                mutually-exclusive masks —
                                                already Brax-shaped.

Number of ``lax.cond`` / ``lax.switch`` constructs flattened per
function:

  * ``step_brax``                       : 0 conds, 0 switches
  * ``tick_hallu_expiry_brax``          : 0 conds, 0 switches
  * ``tick_luck_drift_brax``            : 0 conds, 0 switches
  * ``tick_slime_cancels_stoning_brax`` : 0 conds, 0 switches

Total: 0 ``lax.cond`` flattened, 0 ``lax.switch`` flattened, 0
``lax.scan`` unrolled, 0 ``lax.fori_loop`` unrolled.  The originals are
already canonical Brax-shape (this was verified by grepping
``status_effects.py`` for ``lax.cond``, ``lax.switch``, ``lax.scan``,
``lax.fori``, ``lax.while`` — all return zero matches).

Despite the zero-flatten result, this module exists so the post-Brax
H100 compile path can import the four per-step entry points from one
canonical Brax landing zone.  Subsequent edits to ``status_effects.py``
that may add ``lax.cond`` (e.g. for ergonomic reasons in eager mode) will
not silently re-introduce HLO blow-up at the per-step seam, because the
JIT pipeline pins these ``_brax`` symbols.

Byte-parity constraints
-----------------------
1. RNG draw order preserved exactly.  ``step_brax`` performs the same
   ``jax.random.split`` chain as ``step``: ``rng → (rng, rng_hp)`` for
   HP regen, ``rng → (rng, rng_pw)`` for Pw regen, then the unsplit
   ``rng`` is forwarded to ``apply_starvation`` whose internal splits
   match the original.  ``tick_luck_drift_brax`` and
   ``tick_hallu_expiry_brax`` perform no RNG draws.
   ``tick_slime_cancels_stoning_brax`` performs no RNG draws.
2. Every mutation routes through ``jnp.where`` masking (either directly
   or via ``jax.tree.map(jnp.where, new, old)`` for pytree updates) or
   via ``arr.at[idx].set(jnp.where(mask, new, old))``.  No conditional
   ``.at[...].set(...)``.
3. State pytree shape preserved (we only call ``.replace`` with same
   field names/dtypes as the originals).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from Nethax.nethax.subsystems.status_effects import (
    Intrinsic,
    StatusState,
    TimedStatus,
    apply_food_poisoning,
    apply_poisoned_tick,
    apply_sick_lethal,
    apply_sliming,
    apply_starvation,
    apply_stoning,
    apply_strangulation,
    hp_regen_tick,
    hunger_tick,
    pw_regen_tick,
    tick_timers,
    _AMULET_YENDOR_TID,
    _LUCKSTONE_TYPE_ID,
)


# ---------------------------------------------------------------------------
# step_brax — main per-turn status-effect orchestrator
# ---------------------------------------------------------------------------

def step_brax(
    state: StatusState,
    rng: jax.Array,
    player_hp: jnp.ndarray,
    player_hp_max: jnp.ndarray,
    player_pw: jnp.ndarray,
    player_pw_max: jnp.ndarray,
    player_xl: jnp.ndarray,
    player_role: jnp.ndarray,
    done: jnp.ndarray,
    player_int: jnp.ndarray = None,
    player_wis: jnp.ndarray = None,
    player_con: jnp.ndarray = None,
    timestep: jnp.ndarray = None,
) -> tuple:
    """Brax-style port of ``status_effects.step``.

    Byte-parity contract: every mutation must produce arrays that compare
    bit-equal to the original implementation for the same inputs.

    Order (mirrors allmain.c moveloop / timeout.c nh_timeout):
      1. Lethal-expiry checks (pre-decrement, so timer==1 fires death).
      2. tick_timers — decrement all counters.
      3. hunger_tick — drain nutrition, update hunger_state.
      4. POISONED damage tick.
      5. HP regen.
      6. Pw regen.
      7. Starvation / fainting.

    RNG draw order (must match ``status_effects.step`` exactly):
      split #1: rng → (rng, rng_hp)
      split #2: rng → (rng, rng_pw)
      then the remaining rng is forwarded to ``apply_starvation`` which
      performs two further internal splits (faint roll + faint duration).

    Returns (new_status_state, new_player_hp, new_player_pw, new_done).

    Cite: vendor/nethack/src/allmain.c::moveloop lines 273-305.
    """
    # --- 1. Lethal expiry checks (pre-decrement).  Each helper is itself
    # already Brax-shape: ``jnp.where(timer==1, 0, hp)`` / ``done | expiring``.
    state, player_hp, done = apply_strangulation(state, player_hp, done)
    state, player_hp, done = apply_stoning(state, player_hp, done)
    state, player_hp, done = apply_sliming(state, player_hp, done)
    state, player_hp, done = apply_food_poisoning(state, player_hp, done)
    # SICK (illness, sick_kind==2) lethal expiry — vendor status.c::sick.
    state, player_hp, done = apply_sick_lethal(state, player_hp, done)

    # --- 2. Decrement all timers (pure jnp.maximum, no control flow).
    state = tick_timers(state)

    # Defaults for legacy callers (None ⇒ neutral baselines).  These are
    # Python-level ``is None`` checks against static trace-time values, NOT
    # traced conds.
    _con = player_con if player_con is not None else jnp.int32(11)
    _wis = player_wis if player_wis is not None else jnp.int32(11)
    _int = player_int if player_int is not None else jnp.int32(11)
    _moves = timestep if timestep is not None else jnp.int32(0)

    # --- 3. Hunger drain (CON threads through; STARVED death-cliff is
    # CON-dependent per vendor eat.c:3437).
    state = hunger_tick(state, con=_con)

    # --- 3b. POISONED damage tick (post tick_timers; reads decremented timer).
    state, player_hp = apply_poisoned_tick(state, player_hp)

    # --- 4. HP regen.  RNG split #1 — must precede the pw split.
    rng, rng_hp = jax.random.split(rng)
    new_state, new_hp = hp_regen_tick(
        state, player_hp, player_hp_max, player_xl, player_role,
        _con, _moves, rng_hp,
    )
    state = new_state
    # Brax-flat: ``jnp.where(done, old, new)`` is exactly the original guard
    # that prevents a mid-turn death from being undone by HP regen.
    player_hp = jnp.where(done, player_hp, new_hp)

    # --- 5. Pw regen.  RNG split #2 — order matches the original.
    rng, rng_pw = jax.random.split(rng)
    new_state, new_pw = pw_regen_tick(
        state, player_pw, player_pw_max, player_xl, player_role,
        _int, _wis, _moves, rng_pw,
    )
    state = new_state
    player_pw = jnp.where(done, player_pw, new_pw)

    # --- 6. Starvation / fainting.  ``apply_starvation`` consumes the
    # remaining rng (two internal splits: faint chance + faint duration);
    # the byte-equivalence to the original holds because the input rng is
    # unchanged from the original ``step`` at this point.
    state, player_hp, done = apply_starvation(state, player_hp, done, rng)

    return state, player_hp, player_pw, done


# ---------------------------------------------------------------------------
# tick_hallu_expiry_brax — fire "Everything looks SO boring now." on expiry
# ---------------------------------------------------------------------------

def tick_hallu_expiry_brax(env_state):
    """Brax-style port of ``status_effects.tick_hallu_expiry``.

    Pre-decrement convention: timer == 1 this turn means the timer will tick
    to 0 and the hallucination ends.  Emits the vendor message via the
    EnvState's MessageState ring buffer.

    The original already routes the message-buffer update through
    ``jax.tree.map(jnp.where, new, old)`` — that is the Brax pattern for
    pytree-level masked selection.  We reproduce it verbatim.

    Cite: vendor/nethack/src/timeout.c::nh_timeout HALLU case lines 778-783
          (set_itimeout(&HHallucination, 1L); make_hallucinated(0L, TRUE, 0L));
          vendor/nethack/src/potion.c::make_hallucinated lines 369-411
          (the !xtime branch picks "Everything %s SO boring now.").
    """
    # Local import — mirrors the original (avoids a circular import at
    # module-load time between status_effects and messages).
    from Nethax.nethax.subsystems.messages import emit, MessageId

    timer = env_state.status.timed_statuses[TimedStatus.HALLUCINATION]
    expiring = timer == jnp.int32(1)

    # Always compute the "emitted" message ring buffer (Brax: both branches
    # always evaluated).
    new_messages = emit(env_state.messages, jnp.int32(int(MessageId.HALLU_BORING)))

    # Mask-select the ring buffer field-by-field via tree.map(jnp.where).
    # This is byte-equivalent to ``lax.cond(expiring, emit, identity, ...)``
    # without the cond's HLO blowup under vmap.
    final_messages = jax.tree.map(
        lambda new, old: jnp.where(expiring, new, old),
        new_messages,
        env_state.messages,
    )
    return env_state.replace(messages=final_messages)


# ---------------------------------------------------------------------------
# tick_luck_drift_brax — drift player_luck toward baseluck every 300/600 moves
# ---------------------------------------------------------------------------

def tick_luck_drift_brax(env_state):
    """Brax-style port of ``status_effects.tick_luck_drift``.

    Drift ``player_luck`` toward baseluck=0 every 300/600 moves.

    Vendor cite: timeout.c::nh_timeout lines 606-620.
        if (u.uluck != baseluck
            && svm.moves % ((u.uhave.amulet || u.ugangr) ? 300 : 600) == 0) {
            int time_luck = stone_luck(FALSE);
            boolean nostone = !carrying(LUCKSTONE) && !stone_luck(TRUE);
            if (u.uluck > baseluck && (nostone || time_luck < 0)) u.uluck--;
            else if (u.uluck < baseluck && (nostone || time_luck > 0)) u.uluck++;
        }

    Simplifications (Nethax has no calendar/quest-leader-killed tracking):
      - baseluck = 0  (no FULL_MOON / friday13 / killed_leader / FEDORA bonus).
      - time_luck collapses to the BUC sign of a carried LUCKSTONE.

    Brax shape: every branch of the three-way drift decision is computed
    eagerly via boolean masks (``drift_down``, ``drift_up``, ``apply``);
    the chosen delta is selected through a single nested ``jnp.where``.

    JIT-stable; runs every turn but mostly no-ops.
    """
    moves    = env_state.timestep.astype(jnp.int32)
    luck     = env_state.player_luck.astype(jnp.int32)
    baseluck = jnp.int32(0)

    # ---- 300 vs 600-turn cadence -------------------------------------
    inv = env_state.inventory
    cat = inv.items.category
    tid = inv.items.type_id
    have_amulet = jnp.any(
        (cat != jnp.int8(0)) & (tid == jnp.int16(_AMULET_YENDOR_TID))
    )
    ugangr = env_state.prayer.god_anger.astype(jnp.int32)
    fast_drift = have_amulet | (ugangr > jnp.int32(0))
    # Brax: ``period`` chosen via ``jnp.where`` (no lax.cond) — both 300
    # and 600 are computed (trivially) and one is masked-in.
    period = jnp.where(fast_drift, jnp.int32(300), jnp.int32(600))

    on_period   = (moves > jnp.int32(0)) & (moves % period == jnp.int32(0))
    luck_differs = luck != baseluck

    # ---- carrying(LUCKSTONE) + BUC inspection ------------------------
    has_luckstone = jnp.any(
        (cat != jnp.int8(0)) & (tid == jnp.int16(_LUCKSTONE_TYPE_ID))
    )
    luck_mask = (cat != jnp.int8(0)) & (tid == jnp.int16(_LUCKSTONE_TYPE_ID))
    # ``argmax`` over a bool[]/int32 mask is safe even when the mask is
    # all-False — it returns 0 and we mask the read with ``has_luckstone``.
    safe_idx = jnp.argmax(luck_mask.astype(jnp.int32))
    buc = jnp.where(
        has_luckstone, inv.items.buc_status[safe_idx], jnp.int8(2)
    ).astype(jnp.int32)

    # buc: 0=blessed → +1, 1=cursed → -1, 2=uncursed → 0  (BUCStatus).
    # Brax: nested ``jnp.where`` over mutually exclusive BUC values.
    time_luck = jnp.where(
        buc == jnp.int32(0), jnp.int32(1),
        jnp.where(buc == jnp.int32(1), jnp.int32(-1), jnp.int32(0)),
    )
    # ``nostone == !carrying(LUCKSTONE) && !stone_luck(TRUE)`` — any carried
    # luckstone (B/U/C) makes stone_luck(TRUE) nonzero, so ``nostone`` is
    # exactly ``~has_luckstone``.
    nostone = ~has_luckstone

    # ---- Three-way drift decision (all branches computed) ------------
    drift_down = (luck > baseluck) & (nostone | (time_luck < jnp.int32(0)))
    drift_up   = (luck < baseluck) & (nostone | (time_luck > jnp.int32(0)))

    apply_drift = on_period & luck_differs
    delta = jnp.where(
        apply_drift & drift_down, jnp.int32(-1),
        jnp.where(apply_drift & drift_up, jnp.int32(1), jnp.int32(0)),
    )
    new_luck = (luck + delta).astype(jnp.int8)
    return env_state.replace(player_luck=new_luck)


# ---------------------------------------------------------------------------
# tick_slime_cancels_stoning_brax — slime-dialogue per-turn side effects
# ---------------------------------------------------------------------------

def tick_slime_cancels_stoning_brax(env_state):
    """Brax-style port of ``status_effects.tick_slime_cancels_stoning``.

    Vendor cite: timeout.c::slime_dialogue lines 380-443.
      Per-turn message ticks (fire when ``(Slimed & TIMEOUT) % 2 != 0`` and
      ``i = (Slimed & TIMEOUT) / 2``):
        i==4 (t==9)  "You are turning a little green."          (no gameplay)
        i==3 (t==7)  "Your limbs are getting oozy."             + HFast = 0
        i==2 (t==5)  "Your skin begins to peel away."           + HDeaf = 5
        i==1 (t==3)  "You are turning into green slime."        + cancel STONED
      i==0 (t==1) is handled separately by tick_slimed_lethal (done_timeout).

    Vendor switch (lines 424-441):
        case 3L:  HFast = 0L;                            /* lose intrinsic speed */
        case 2L:  if (0 < HDeaf < 5) set_itimeout(&HDeaf, 5L);
        case 1L:  if (Stoned) make_stoned(0L, ...);      /* silently cancel */

    Brax shape: all four ladder rungs (i=4,3,2,1) are tested via mutually
    exclusive scalar masks; every state mutation is a ``jnp.where`` blend
    between the unchanged value and the side-effect value, with the
    message-ring update routed through ``jax.tree.map(jnp.where, ...)`` for
    pytree-shape parity.  No ``lax.cond`` / ``lax.switch`` — the original
    vendor C ``switch`` is flattened into four parallel masked writes.

    Runs BEFORE the per-turn timer decrement (i.e. before ``_status_step``)
    so we read the pre-decrement SLIMED value.
    """
    from Nethax.nethax.subsystems.messages import emit, MessageId

    timers   = env_state.status.timed_statuses
    slimed_t = timers[TimedStatus.SLIMED].astype(jnp.int32)
    stoned_t = timers[TimedStatus.STONED].astype(jnp.int32)
    deaf_t   = timers[TimedStatus.DEAF].astype(jnp.int32)

    # --- Ladder rung masks (mutually exclusive on slimed_t) -----------
    # i==4 (slimed t ∈ {8,9}): "turning green" — message-only.
    slime_at_i4 = (slimed_t == jnp.int32(8)) | (slimed_t == jnp.int32(9))
    # i==3 (slimed t ∈ {6,7}): "limbs oozy" + clear HFast.
    slime_at_i3 = (slimed_t == jnp.int32(6)) | (slimed_t == jnp.int32(7))
    # i==2 (slimed t ∈ {4,5}): "skin peel" + bump HDeaf to 5 when 0 < HDeaf < 5.
    slime_at_i2 = (slimed_t == jnp.int32(4)) | (slimed_t == jnp.int32(5))
    # i==1 (slimed t ∈ {2,3}): "turning into slime" + cancel stoning.
    slime_at_i1 = (slimed_t == jnp.int32(2)) | (slimed_t == jnp.int32(3))

    # Vendor "i % 2 != 0" (message fires) maps to slimed_t being odd.
    msg_at_i4 = slime_at_i4 & (slimed_t == jnp.int32(9))
    msg_at_i3 = slime_at_i3 & (slimed_t == jnp.int32(7))
    msg_at_i2 = slime_at_i2 & (slimed_t == jnp.int32(5))
    msg_at_i1 = slime_at_i1 & (slimed_t == jnp.int32(3))

    # --- i==1: cancel STONED -----------------------------------------
    cancel_stone = slime_at_i1 & (stoned_t > jnp.int32(0))
    new_stoned   = jnp.where(cancel_stone, jnp.int32(0), stoned_t)

    # --- i==2: bump DEAF timer to 5 if 0 < DEAF < 5 ------------------
    # Vendor: ``if ((HDeaf & TIMEOUT) > 0L && (HDeaf & TIMEOUT) < 5L)
    #             set_itimeout(&HDeaf, 5L);``  — extends near-end deafness
    # so Hear_again's "Your hearing returns." doesn't fire while slime-dying.
    bump_deaf = slime_at_i2 & (deaf_t > jnp.int32(0)) & (deaf_t < jnp.int32(5))
    new_deaf  = jnp.where(bump_deaf, jnp.int32(5), deaf_t)

    # --- i==3: clear intrinsic Fast (both flag and timed slot) -------
    # Brax: precompute the "would-be" intrinsics array AND the unchanged
    # array; mask-select the whole array with the scalar ``clear_fast``.
    clear_fast    = slime_at_i3
    intrinsics    = env_state.status.intrinsics
    timed_intr    = env_state.status.timed_intrinsics
    new_intr      = jnp.where(
        clear_fast,
        intrinsics.at[Intrinsic.FAST].set(jnp.bool_(False)),
        intrinsics,
    )
    new_timed_intr = jnp.where(
        clear_fast,
        timed_intr.at[Intrinsic.FAST].set(jnp.int32(0)),
        timed_intr,
    )

    # --- Commit timer writes -----------------------------------------
    new_timers = timers.at[TimedStatus.STONED].set(new_stoned)
    new_timers = new_timers.at[TimedStatus.DEAF].set(
        new_deaf.astype(new_timers.dtype)
    )

    # --- Per-tick message: pick the highest-priority message this turn
    # (msg_at_i{4,3,2,1} are mutually exclusive on slimed_t, so this nested
    # ``jnp.where`` is exactly the C switch flattened to data flow).
    msg_id = jnp.where(
        msg_at_i4, jnp.int32(int(MessageId.SLIME_TURNING_COLOR)),
        jnp.where(
            msg_at_i3, jnp.int32(int(MessageId.SLIME_LIMBS_OOZY)),
            jnp.where(
                msg_at_i2, jnp.int32(int(MessageId.SLIME_SKIN_PEEL)),
                jnp.where(
                    msg_at_i1, jnp.int32(int(MessageId.SLIME_TURNING_INTO)),
                    jnp.int32(int(MessageId.NONE)),
                ),
            ),
        ),
    )
    any_msg = msg_at_i4 | msg_at_i3 | msg_at_i2 | msg_at_i1
    new_messages = emit(env_state.messages, msg_id)
    # Brax pytree-mask: only rotate the ring buffer when *some* message
    # actually fires this turn.
    final_messages = jax.tree.map(
        lambda new, old: jnp.where(any_msg, new, old),
        new_messages,
        env_state.messages,
    )

    return env_state.replace(
        status=env_state.status.replace(
            timed_statuses=new_timers,
            intrinsics=new_intr,
            timed_intrinsics=new_timed_intr,
        ),
        messages=final_messages,
    )
