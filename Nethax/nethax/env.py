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
from Nethax.nethax.subsystems.ascension import maybe_ascend
from Nethax.nethax.subsystems.polymorph import step as _polymorph_step
from Nethax.nethax.subsystems.shop import shop_step as _shop_step
from Nethax.nethax.dungeon.branches import generate_main_branch_l1
from Nethax.nethax.dungeon.spawning import populate_level_with_monsters
from Nethax.nethax.constants.roles import Role
from Nethax.nethax.constants.races import Race
from Nethax.nethax.subsystems.character import create_character


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

        # Generate Main branch level 1 and write into the [branch=0, level=0] slot.
        terrain, _rooms, _active, up_pos, down_pos = generate_main_branch_l1(
            rng_level, self.static
        )
        state = state.replace(
            terrain=state.terrain.at[0, 0].set(terrain),
            player_pos=up_pos.astype(jnp.int16),
        )

        # Populate level 1 with monsters after dungeon gen.
        state = populate_level_with_monsters(state, rng_monsters, n_monsters=5)

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
    rng_act, rng_monsters, rng_status, rng_poly, rng_shop = jax.random.split(rng, 5)
    already_done = state.done

    def _do_step(_):
        # 1. Player action — allmain.c line 203 (svc.context.move).
        ns = dispatch_action(state, action, rng_act)

        # 2. Monster turn — allmain.c line 212 (movemon).
        ns = _monster_ai_step(ns, rng_monsters)

        # 3. Increment turn counter — allmain.c line 244 (svm.moves++).
        ns = ns.replace(timestep=ns.timestep + jnp.int32(1))

        # 4. Status-effect tick — allmain.c line 273 (nh_timeout),
        #    inclusive of regen_hp (line 294) and regen_pw (line 305).
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
    reward = jnp.float32(0.0)
    return new_state, obs, reward, new_state.done
