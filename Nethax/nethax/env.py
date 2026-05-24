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
from Nethax.nethax.subsystems.ascension import maybe_ascend
from Nethax.nethax.subsystems.polymorph import step as _polymorph_step
from Nethax.nethax.subsystems.shop import shop_step as _shop_step
from Nethax.nethax.dungeon.branches import (
    generate_main_branch_l1,
    generate_main_branch_l1_with_features,
)
from Nethax.nethax.dungeon.spawning import populate_level_with_monsters
from Nethax.nethax.constants.roles import Role
from Nethax.nethax.constants.races import Race
from Nethax.nethax.constants import TileType
from Nethax.nethax.subsystems.character import create_character, get_starting_pet
from Nethax.nethax.subsystems.skills import init_skills
from Nethax.nethax.subsystems.digging import dig_tick as _dig_tick
from Nethax.nethax.subsystems.swallow import digest_tick as _digest_tick
from Nethax.nethax.subsystems.experience import newexplevel as _newexplevel
from Nethax.nethax.subsystems.regions import run_regions as _run_regions


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
    ) -> Tuple[EnvState, Dict[str, jax.Array]]:
        """Return (initial_state, initial_observation).

        Parameters
        ----------
        rng       : JAX PRNG key
        role      : Role enum value; defaults to VALKYRIE if None.
        race      : Race enum value; defaults to HUMAN if None.
        alignment : 0=lawful, 1=neutral, 2=chaotic; default 0.
        """
        if role is None:
            role = Role.VALKYRIE
        if race is None:
            race = Race.HUMAN

        rng_state, rng_level, rng_char, rng_monsters = jax.random.split(rng, 4)
        state = EnvState.default(rng=rng_state, static=self.static)

        # Apply character creation (stats, inventory, AC)
        char_fields = create_character(rng_char, role, race, alignment)
        state = state.replace(**char_fields)

        # Initialise role-specific skill caps (vendor/nethack/src/u_init.c Skill_X tables).
        state = state.replace(skills=init_skills(role))

        # Generate Main branch level 1 and write into the [branch=0, level=0]
        # slot.  This includes the per-room independent feature rolls
        # (fountain / altar / grave / traps) and the 2x2 detached vault —
        # vendor/nethack/src/mklev.c::mklev (line 1577) which calls
        # fill_ordinary_room (line 939) for every OROOM/THEMEROOM and the
        # vault gate at lines 404-410 / 1316-1342.
        (
            terrain,
            _rooms,
            _active,
            up_pos,
            down_pos,
            new_features,
            new_traps,
        ) = generate_main_branch_l1_with_features(
            rng_level,
            self.static,
            state.features,
            state.traps,
            flat_lv=0,
            depth=1,
            player_align=int(alignment),
        )
        state = state.replace(
            terrain=state.terrain.at[0, 0].set(terrain),
            player_pos=up_pos.astype(jnp.int16),
            features=new_features,
            traps=new_traps,
        )

        # Populate level 1 with monsters after dungeon gen.
        state = populate_level_with_monsters(state, rng_monsters, n_monsters=5)

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


def _spawn_starting_pet(state, role: Role):
    """Spawn the role's starting pet adjacent to the player.

    Vendor: vendor/nethack/src/u_init.c::makedog (called from u_init()).
    Host-side only — reset() is not jit-compiled.
    Pet is placed in slot 5 (after the 5 wild monsters in slots 0-4).
    """
    from Nethax.nethax.constants.monsters import MONSTERS
    from Nethax.nethax.dungeon.spawning import (
        _BASE_AC, _ATK_DICE_N, _ATK_DICE_S, _IS_LARGE, _roll_hp,
    )
    import numpy as np
    import jax.random as jr

    # Resolve pet monster name → MONSTERS index (host-side name lookup).
    pet_name = get_starting_pet(role)
    pet_pm = next(
        (i for i, m in enumerate(MONSTERS) if m.name == pet_name),
        32,  # fallback: kitten (index 32)
    )

    # Find an adjacent FLOOR or CORRIDOR tile (Chebyshev distance == 1).
    terrain = np.array(state.terrain[0, 0])   # host numpy copy
    pr = int(state.player_pos[0])
    pc = int(state.player_pos[1])
    H, W = terrain.shape
    pet_pos = (pr, pc)  # fallback: same tile as player (vendor fallback)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = pr + dr, pc + dc
            if 0 <= rr < H and 0 <= cc < W:
                t = int(terrain[rr, cc])
                if t in (int(TileType.FLOOR), int(TileType.CORRIDOR)):
                    pet_pos = (rr, cc)
                    break
        else:
            continue
        break

    # Roll HP using the same formula as wild monsters (makemon.c::newmonhp).
    dummy_rng = jr.PRNGKey(0)
    hp_val = int(_roll_hp(dummy_rng, jnp.int32(max(1, int(MONSTERS[pet_pm].level)))))

    # Write pet into slot 5 (first slot after the 5 wild monsters).
    PET_SLOT = 5
    pm_i16 = jnp.int16(pet_pm)
    mai = state.monster_ai.replace(
        alive=state.monster_ai.alive.at[PET_SLOT].set(True),
        tame=state.monster_ai.tame.at[PET_SLOT].set(True),
        peaceful=state.monster_ai.peaceful.at[PET_SLOT].set(True),
        mtame=state.monster_ai.mtame.at[PET_SLOT].set(jnp.int8(10)),
        entry_idx=state.monster_ai.entry_idx.at[PET_SLOT].set(pm_i16),
        pos=state.monster_ai.pos.at[PET_SLOT].set(
            jnp.array(pet_pos, dtype=jnp.int16)
        ),
        hp=state.monster_ai.hp.at[PET_SLOT].set(jnp.int32(hp_val)),
        hp_max=state.monster_ai.hp_max.at[PET_SLOT].set(jnp.int32(hp_val)),
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
    rng_act, rng_monsters, rng_status, rng_poly, rng_shop, rng_swallow, rng_explvl, rng_regions = jax.random.split(rng, 8)
    already_done = state.done

    def _do_step(_):
        # 1. Player action — allmain.c line 203 (svc.context.move).
        ns = dispatch_action(state, action, rng_act)

        # 1b. Digging tick — advance multi-turn pickaxe dig (dig.c::dodig).
        ns = _dig_tick(ns, rng_act)

        # 2. Monster turn — allmain.c line 212 (movemon).
        ns = _monster_ai_step(ns, rng_monsters)

        # 2b. Per-turn region tick — vendor/nethack/src/region.c::run_regions
        #     (line 414).  Ages every active region by 1 and applies
        #     gas-cloud damage to the player when they stand inside one.
        ns = _run_regions(ns, rng_regions)

        # 3. Increment turn counter — allmain.c line 244 (svm.moves++).
        ns = ns.replace(timestep=ns.timestep + jnp.int32(1))

        # 4. Status-effect tick — allmain.c line 273 (nh_timeout),
        #    inclusive of regen_hp (line 294) and regen_pw (line 305).
        #    Pre-tick: emit HALLUCINATION expiry message (vendor timeout.c
        #    HALLU case lines 778-783 — make_hallucinated(0L, TRUE, 0L) →
        #    "Everything looks SO boring now.").
        ns = _tick_hallu_expiry(ns)
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
        #    by 1.
        magic = ns.magic
        new_mem = jnp.maximum(magic.spell_memory - jnp.int32(1), jnp.int32(0))
        ns = ns.replace(magic=magic.replace(spell_memory=new_mem))

        # 7. Shop tick — Wave 6 #47 (pay-at-exit + angry shopkeeper pursuit).
        #    Vendor: shk.c invoked from moveloop's per-turn block.
        ns = _shop_step(ns, rng_shop)

        # 8. Ascension / endgame check — vendor allmain.c done() paths.
        ns = maybe_ascend(ns)
        return ns

    new_state = jax.lax.cond(already_done, lambda _: state, _do_step, operand=None)
    obs = build_nle_observation(new_state)
    # Reward = score delta (NLE convention: vendor topten.c::u.urexp running
    # accumulator, surfaced as bl_score in blstats).  Already-done steps
    # contribute 0 since new_state == state.
    reward = jnp.float32(new_state.scoring.score - state.scoring.score)
    return new_state, obs, reward, new_state.done
