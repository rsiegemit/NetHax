"""Monster passive-attack effects on the player.

When the player melees a monster, any AT_NONE attack in the monster's attack
list fires as a passive contact effect — regardless of whether the player hit
or missed.

Canonical source:
  vendor/nethack/src/uhitm.c::passive() (lines 5864–6119) — hero melee
    triggers monster passive; includes AD_PLYS, AD_COLD, AD_STUN, AD_FIRE,
    AD_ACID, AD_STON, AD_RUST, AD_ENCH.
  vendor/nethack/src/mhitu.c::hitmu() line 1060 — AD_DRIN brain drain fires
    on every tentacle hit.

Design:
  - _PASSIVE_TYPE[N_MONSTERS] int8 maps each monster's entry_idx to a passive
    type constant (0 = none).
  - apply_passive_to_player(state, attacker_slot, rng) -> EnvState dispatches
    via jax.lax.switch over 11 cases.
  - JIT-pure: no Python branches on tracers.
"""
import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Passive type constants
# ---------------------------------------------------------------------------
_P_NONE        = 0
_P_EYE_PARALYZE = 1   # floating eye  — AD_PLYS  uhitm.c:6022
_P_COLD        = 2    # brown mold    — AD_COLD   uhitm.c:6066
_P_YLW_PARALYZE = 3   # yellow mold   — AD_STUN   uhitm.c:6085
_P_ACID        = 4    # green mold    — AD_ACID   uhitm.c:5906
_P_FIRE        = 5    # red mold      — AD_FIRE   uhitm.c:5895
_P_COLD_SLEEP  = 6    # blue jelly    — AD_COLD + sleep chance  uhitm.c:6066
_P_STONING     = 7    # cockatrice    — AD_STON   uhitm.c:5934
_P_BRAIN_DRAIN = 8    # mind flayer   — AD_DRIN   mhitu.c:1060
_P_DISENCHANT  = 9    # disenchanter  — AD_ENCH   uhitm.c:5992
_P_RUST        = 10   # rust monster  — AD_RUST   uhitm.c:5958

_N_PASSIVE_TYPES = 11  # indices 0..10


# ---------------------------------------------------------------------------
# Build _PASSIVE_TYPE table once at module load (JIT-safe constant).
# Monster indices verified against Nethax/nethax/constants/monsters.py.
#
#   28  floating eye    → _P_EYE_PARALYZE
#   55  blue jelly      → _P_COLD_SLEEP
#   156 brown mold      → _P_COLD
#   157 yellow mold     → _P_YLW_PARALYZE
#   158 green mold      → _P_ACID
#   159 red mold        → _P_FIRE
#   10  cockatrice      → _P_STONING
#   47  mind flayer     → _P_BRAIN_DRAIN
#   48  master mind flayer → _P_BRAIN_DRAIN
#   209 disenchanter    → _P_DISENCHANT
#   208 rust monster    → _P_RUST
# ---------------------------------------------------------------------------
def _build_passive_type_table() -> jnp.ndarray:
    from Nethax.nethax.constants.monsters import MONSTERS
    n = len(MONSTERS)
    tbl = [_P_NONE] * n
    _MAP = {
        28: _P_EYE_PARALYZE,
        55: _P_COLD_SLEEP,
        156: _P_COLD,
        157: _P_YLW_PARALYZE,
        158: _P_ACID,
        159: _P_FIRE,
        10: _P_STONING,
        47: _P_BRAIN_DRAIN,
        48: _P_BRAIN_DRAIN,
        209: _P_DISENCHANT,
        208: _P_RUST,
    }
    for idx, ptype in _MAP.items():
        if idx < n:
            tbl[idx] = ptype
    return jnp.array(tbl, dtype=jnp.int8)


_PASSIVE_TYPE: jnp.ndarray = _build_passive_type_table()


# ---------------------------------------------------------------------------
# Individual passive handlers — each takes (state, rng) and returns EnvState.
# All are JIT-pure (no Python conditionals on traced values).
# ---------------------------------------------------------------------------

def _passive_none(state, _rng):
    return state


def _passive_eye_paralyze(state, rng):
    """Floating eye passive: player frozen by gaze unless blind or reflecting.

    Cite: vendor/nethack/src/uhitm.c::passive() line 6022-6064.
    Gate: mon.mcansee (always true in Nethax) AND !Blind AND !reflects.
    Duration: rnd(120) turns added to FROZEN timer (Nethax uses FROZEN for
    paralysis; vendor uses nomul(-tmp)).
    """
    from Nethax.nethax.subsystems.status_effects import TimedStatus, Intrinsic
    from Nethax.nethax.rng import rnd

    blind = state.status.timed_statuses[int(TimedStatus.BLIND)].astype(jnp.int32) > 0
    reflecting = state.status.intrinsics[int(Intrinsic.REFLECTING)]
    free_action = state.status.intrinsics[int(Intrinsic.FREE_ACTION)]

    duration = rnd(rng, 120).astype(jnp.int32)
    should_paralyze = (~blind) & (~reflecting) & (~free_action)

    current_frozen = state.status.timed_statuses[int(TimedStatus.FROZEN)].astype(jnp.int32)
    new_frozen = jnp.where(should_paralyze, current_frozen + duration, current_frozen)
    new_statuses = state.status.timed_statuses.at[int(TimedStatus.FROZEN)].set(new_frozen)
    new_status = state.status.replace(timed_statuses=new_statuses)
    return state.replace(status=new_status)


def _passive_cold(state, rng):
    """Brown mold passive: rnd(6) cold damage, gated by cold resistance.

    Cite: vendor/nethack/src/uhitm.c::passive() line 6066-6083.
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic
    from Nethax.nethax.rng import rnd

    cold_res = (
        state.status.intrinsics[int(Intrinsic.RESIST_COLD)]
        | (state.status.timed_intrinsics[int(Intrinsic.RESIST_COLD)] > 0)
    )
    dmg = jnp.where(cold_res, jnp.int32(0), rnd(rng, 6).astype(jnp.int32))
    new_hp = jnp.maximum(state.player_hp - dmg, jnp.int32(0))
    new_done = state.done | (new_hp <= jnp.int32(0))
    return state.replace(player_hp=new_hp, done=new_done)


def _passive_yellow_paralyze(state, rng):
    """Yellow mold passive: d(mlev+1, 4) stun, gated by magic resistance.

    Cite: vendor/nethack/src/uhitm.c::passive() lines 5885-5890 + 6085-6088
    (AD_STUN). tmp computed at 5887-5888 as ``d(mon->m_lev+1, damd)`` when
    damn==0 (yellow mold has ATTK(AT_NONE, AD_STUN, 0, 4) in monsters.h:1634,
    so damn==0, damd==4, m_lev==1 ⇒ tmp = d(2, 4)). make_stunned(tmp) at
    6087 applies the stun duration. Nethax maps the contact stun onto the
    FROZEN slot (paralytic contact); the duration formula is byte-equal.
    """
    from Nethax.nethax.subsystems.status_effects import TimedStatus, Intrinsic
    from Nethax.nethax.rng import dice_roll

    magic_res = state.status.intrinsics[int(Intrinsic.MAGIC_RESIST)]

    # Vendor uhitm.c:5887-5888 — tmp = d((int) mon->m_lev + 1, damd).
    # Yellow mold: m_lev=1, damd=4 ⇒ d(2, 4).
    duration = dice_roll(rng, 2, 4).astype(jnp.int32)
    current_frozen = state.status.timed_statuses[int(TimedStatus.FROZEN)].astype(jnp.int32)
    new_frozen = jnp.where(magic_res, current_frozen, current_frozen + duration)
    new_statuses = state.status.timed_statuses.at[int(TimedStatus.FROZEN)].set(new_frozen)
    new_status = state.status.replace(timed_statuses=new_statuses)
    return state.replace(status=new_status)


def _passive_acid(state, rng):
    """Green mold passive: rnd(4) acid + weapon corrosion, gated by acid resistance.

    Cite: vendor/nethack/src/uhitm.c::passive() lines 5906-5933.
    Vendor semantics:
      - splash damage fires only if rn2(2) (50% chance) — line 5907.
      - body armor corrosion fires only if !rn2(30) (1/30 chance) — line 5920.
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic
    from Nethax.nethax.rng import rnd
    from Nethax.nethax.rng import rn2 as _rn2

    rng_splash, rng_dmg, rng_corrode = jax.random.split(rng, 3)

    acid_res = (
        state.status.intrinsics[int(Intrinsic.RESIST_ACID)]
        | (state.status.timed_intrinsics[int(Intrinsic.RESIST_ACID)] > 0)
    )
    # rn2(2) gate — 50% chance to splash. Cite: uhitm.c:5907.
    splash = _rn2(rng_splash, 2) != jnp.int32(0)
    raw_dmg = rnd(rng_dmg, 4).astype(jnp.int32)
    dmg = jnp.where(splash & ~acid_res, raw_dmg, jnp.int32(0))
    new_hp = jnp.maximum(state.player_hp - dmg, jnp.int32(0))
    new_done = state.done | (new_hp <= jnp.int32(0))
    state = state.replace(player_hp=new_hp, done=new_done)

    # Corrode wielded weapon (vendor passive_obj AD_ACID → erode_obj ERODE_CORRODE).
    # Cite: vendor/nethack/src/uhitm.c::passive() line 5920 — !rn2(30) gate.
    from Nethax.nethax.subsystems.items import erode_obj_slot, ERODE_CORRODE

    wielded = state.inventory.wielded.astype(jnp.int32)
    has_weapon = wielded >= jnp.int32(0)
    # 1/30 chance: !rn2(30) is true when rn2(30) == 0.
    corrode_roll = _rn2(rng_corrode, 30) == jnp.int32(0)
    should_corrode = splash & has_weapon & (~acid_res) & corrode_roll

    def _do_corrode(items_in):
        safe_w = jnp.clip(wielded, 0, items_in.oeroded.shape[0] - 1)
        new_items, _ = erode_obj_slot(items_in, safe_w, ERODE_CORRODE, True)
        return new_items

    new_items = jax.lax.cond(
        should_corrode, _do_corrode, lambda x: x, state.inventory.items
    )
    new_inv = state.inventory.replace(items=new_items)
    return state.replace(inventory=new_inv)


def _passive_fire(state, rng):
    """Red mold passive: rnd(4) fire damage, gated by fire resistance.

    Cite: vendor/nethack/src/uhitm.c::passive() lines 5895-5905 (AD_FIRE
    outer case ➜ passive_obj on the attacking weapon, NOT body armor) and
    lines 6089-6101 (AD_FIRE inner case ➜ fire damage gated by Fire_resistance).
    passive_obj AD_FIRE: vendor/nethack/src/uhitm.c:6157-6162 calls
    erode_obj on the wielded weapon with a 1/6 chance (rn2(6)==0).
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic
    from Nethax.nethax.subsystems.items import erode_obj_slot, ERODE_BURN
    from Nethax.nethax.rng import rnd
    from Nethax.nethax.rng import rn2 as _rn2

    rng_dmg, rng_erode = jax.random.split(rng)
    fire_res = (
        state.status.intrinsics[int(Intrinsic.RESIST_FIRE)]
        | (state.status.timed_intrinsics[int(Intrinsic.RESIST_FIRE)] > 0)
    )
    dmg = jnp.where(fire_res, jnp.int32(0), rnd(rng_dmg, 4).astype(jnp.int32))
    new_hp = jnp.maximum(state.player_hp - dmg, jnp.int32(0))
    new_done = state.done | (new_hp <= jnp.int32(0))
    state = state.replace(player_hp=new_hp, done=new_done)

    # Burn the wielded weapon via passive_obj AD_FIRE (rn2(6)==0 gate).
    # Cite: vendor/nethack/src/uhitm.c::passive_obj AD_FIRE lines 6157-6162.
    wielded = state.inventory.wielded.astype(jnp.int32)
    has_weapon = wielded >= jnp.int32(0)
    burn_chance = _rn2(rng_erode, 6) == jnp.int32(0)
    should_burn = has_weapon & burn_chance

    def _do_burn(items_in):
        safe_w = jnp.clip(wielded, 0, items_in.oeroded.shape[0] - 1)
        new_items, _ = erode_obj_slot(items_in, safe_w, ERODE_BURN, True)
        return new_items

    new_items = jax.lax.cond(
        should_burn, _do_burn, lambda x: x, state.inventory.items
    )
    new_inv = state.inventory.replace(items=new_items)
    return state.replace(inventory=new_inv)


def _passive_cold_sleep(state, rng):
    """Blue jelly passive: rnd(6) cold damage + 1/3 chance sleep rnd(25).

    Cite: vendor/nethack/src/uhitm.c::passive() line 6066 (AD_COLD for blue jelly).
    Blue jelly is the only AD_COLD monster with a sleep secondary effect;
    brown mold uses _P_COLD.
    """
    from Nethax.nethax.subsystems.status_effects import TimedStatus, Intrinsic
    from Nethax.nethax.rng import rnd
    from Nethax.nethax.rng import rn2 as _rn2

    cold_res = (
        state.status.intrinsics[int(Intrinsic.RESIST_COLD)]
        | (state.status.timed_intrinsics[int(Intrinsic.RESIST_COLD)] > 0)
    )
    sleep_res = (
        state.status.intrinsics[int(Intrinsic.RESIST_SLEEP)]
        | (state.status.timed_intrinsics[int(Intrinsic.RESIST_SLEEP)] > 0)
    )

    rng_cold, rng_chance, rng_sleep = jax.random.split(rng, 3)
    dmg = jnp.where(cold_res, jnp.int32(0), rnd(rng_cold, 6).astype(jnp.int32))
    new_hp = jnp.maximum(state.player_hp - dmg, jnp.int32(0))
    new_done = state.done | (new_hp <= jnp.int32(0))
    state = state.replace(player_hp=new_hp, done=new_done)

    # 1/3 chance of sleep (vendor: rn2(3) == 0).
    sleep_chance = _rn2(rng_chance, 3) == jnp.int32(0)
    sleep_dur = rnd(rng_sleep, 25).astype(jnp.int32)
    apply_sleep = sleep_chance & (~cold_res) & (~sleep_res)
    cur_sleep = state.status.timed_statuses[int(TimedStatus.SLEEP)].astype(jnp.int32)
    new_sleep = jnp.where(apply_sleep, cur_sleep + sleep_dur, cur_sleep)
    new_statuses = state.status.timed_statuses.at[int(TimedStatus.SLEEP)].set(new_sleep)
    new_status = state.status.replace(timed_statuses=new_statuses)
    return state.replace(status=new_status)


def _passive_stoning(state, rng):
    """Cockatrice passive: touching without gloves petrifies instantly.

    Cite: vendor/nethack/src/uhitm.c::passive() lines 5934-5957 (AD_STON).
    Vendor effect: ``done_in_by(mon, STONING)`` at uhitm.c:5952 — IMMEDIATE
    death, NOT a delayed begin_stoning() timer. Gate at uhitm.c:5949-5951:
    not Stone_resistance and not (poly_when_stoned && polymon(STONE_GOLEM)).
    Glove gate at uhitm.c:5943-5948: protector mask must indicate no body-part
    is protected (cockatrice AT_TUCH ⇒ W_ARMG ⇒ no gloves ⇒ stoned).
    """
    from Nethax.nethax.subsystems.status_effects import TimedStatus, Intrinsic
    from Nethax.nethax.subsystems.inventory import ArmorSlot
    from Nethax.nethax.subsystems.scoring import DeathCause

    stone_res = (
        state.status.intrinsics[int(Intrinsic.RESIST_STONE)]
        | (state.status.timed_intrinsics[int(Intrinsic.RESIST_STONE)] > 0)
    )
    gloves_slot = state.inventory.worn_armor[int(ArmorSlot.GLOVES)].astype(jnp.int32)
    has_gloves = gloves_slot >= jnp.int32(0)

    # uhitm.c:5949-5953 — immediate done_in_by(mon, STONING).
    should_stone = (~stone_res) & (~has_gloves)
    new_hp = jnp.where(should_stone, jnp.int32(0), state.player_hp)
    new_done = state.done | should_stone
    new_cause = jnp.where(
        should_stone,
        jnp.int8(int(DeathCause.STONING)),
        state.scoring.death_cause,
    )
    new_scoring = state.scoring.replace(death_cause=new_cause)
    # Also flag STONED status so on-tile observers see petrified state.
    cur_stoned = state.status.timed_statuses[int(TimedStatus.STONED)].astype(jnp.int32)
    new_stoned = jnp.where(should_stone, jnp.int32(1), cur_stoned)
    new_statuses = state.status.timed_statuses.at[int(TimedStatus.STONED)].set(new_stoned)
    new_status = state.status.replace(timed_statuses=new_statuses)
    return state.replace(
        player_hp=new_hp,
        done=new_done,
        scoring=new_scoring,
        status=new_status,
    )


def _passive_brain_drain(state, rng):
    """Mind flayer passive: tentacle hit drains rnd(2) Int permanently.

    Cite: vendor/nethack/src/mhitu.c::hitmu() line 1060 (AD_DRIN tentacle
    hit drains intelligence).  Lose XL if Int reaches 1.
    """
    from Nethax.nethax.rng import rnd

    drain = rnd(rng, 2).astype(jnp.int8)
    new_int = jnp.maximum(state.player_int - drain, jnp.int8(1))
    state = state.replace(player_int=new_int)

    # Lose one XL when Int reaches 1 (vendor: losexp()).
    xl_loss = (new_int <= jnp.int8(1)) & (state.player_xl > jnp.int32(1))
    new_xl = jnp.where(xl_loss, state.player_xl - jnp.int32(1), state.player_xl)
    return state.replace(player_xl=new_xl)


def _passive_disenchant(state, rng):
    """Disenchanter passive: reduces wielded weapon enchantment by 1.

    Cite: vendor/nethack/src/uhitm.c::passive() lines 5992-6011 (AD_ENCH).
    passive_obj → drain_item reduces spe by 1 (minimum clamp varies; we
    allow negative enchantment as vendor does).
    """
    wielded = state.inventory.wielded.astype(jnp.int32)
    has_weapon = wielded >= jnp.int32(0)
    safe_w = jnp.clip(wielded, 0, state.inventory.items.enchantment.shape[0] - 1)
    cur_ench = state.inventory.items.enchantment[safe_w].astype(jnp.int32)
    new_ench = jnp.where(has_weapon, (cur_ench - jnp.int32(1)).astype(jnp.int8),
                         state.inventory.items.enchantment[safe_w])
    new_enchantments = state.inventory.items.enchantment.at[safe_w].set(new_ench)
    new_items = state.inventory.items.replace(enchantment=new_enchantments)
    new_inv = state.inventory.replace(items=new_items)
    return state.replace(inventory=new_inv)


def _passive_rust(state, rng):
    """Rust monster passive: increments oeroded on wielded metal weapon/armor.

    Cite: vendor/nethack/src/uhitm.c::passive() lines 5958-5967 (AD_RUST) →
    vendor/nethack/src/trap.c::erode_obj kind=ERODE_RUST: rust-proof items
    are immune via the central erode_obj path.
    """
    from Nethax.nethax.subsystems.items import erode_obj_slot, ERODE_RUST

    wielded = state.inventory.wielded.astype(jnp.int32)
    has_weapon = wielded >= jnp.int32(0)

    def _do_rust(items_in):
        safe_w = jnp.clip(wielded, 0, items_in.oeroded.shape[0] - 1)
        new_items, _ = erode_obj_slot(items_in, safe_w, ERODE_RUST, True)
        return new_items

    new_items = jax.lax.cond(
        has_weapon, _do_rust, lambda x: x, state.inventory.items
    )
    new_inv = state.inventory.replace(items=new_items)
    return state.replace(inventory=new_inv)


# ---------------------------------------------------------------------------
# Dispatch table — must match _P_* constants above (indices 0..10).
# ---------------------------------------------------------------------------
_PASSIVE_HANDLERS = [
    _passive_none,           # 0  _P_NONE
    _passive_eye_paralyze,   # 1  _P_EYE_PARALYZE
    _passive_cold,           # 2  _P_COLD
    _passive_yellow_paralyze, # 3  _P_YLW_PARALYZE
    _passive_acid,           # 4  _P_ACID
    _passive_fire,           # 5  _P_FIRE
    _passive_cold_sleep,     # 6  _P_COLD_SLEEP
    _passive_stoning,        # 7  _P_STONING
    _passive_brain_drain,    # 8  _P_BRAIN_DRAIN
    _passive_disenchant,     # 9  _P_DISENCHANT
    _passive_rust,           # 10 _P_RUST
]


def apply_passive_to_player(state, attacker_slot: jnp.ndarray, rng: jax.Array):
    """Fire the passive contact effect of the monster at ``attacker_slot``.

    Fires regardless of whether the player hit or missed (vendor uhitm.c
    passive() is called after both hit and miss branches).

    Parameters
    ----------
    state        : EnvState
    attacker_slot : int32 — index into state.monster_ai
    rng          : JAX PRNGKey

    Returns
    -------
    EnvState with player stats/status updated.
    """
    idx = attacker_slot.astype(jnp.int32)
    mai = state.monster_ai
    n_passive = _PASSIVE_TYPE.shape[0]
    # Clip ``idx`` BEFORE indexing into entry_idx — vendor passes a valid
    # monster slot, but JAX silently wraps negative scalar indices, so a
    # sentinel of -1 (or any slot bumped past mai length) would read a
    # bogus entry.  Defensive clip preserves byte-equal behavior for valid
    # slots and produces a safe (slot-0 entry) for invalid sentinels.
    safe_idx = jnp.clip(idx, 0, mai.entry_idx.shape[0] - 1)
    entry = jnp.clip(mai.entry_idx[safe_idx].astype(jnp.int32), 0, n_passive - 1)
    ptype = _PASSIVE_TYPE[entry].astype(jnp.int32)
    ptype_safe = jnp.clip(ptype, 0, _N_PASSIVE_TYPES - 1)

    return jax.lax.switch(
        ptype_safe,
        _PASSIVE_HANDLERS,
        state,
        rng,
    )
