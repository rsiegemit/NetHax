"""Master ``EnvState`` for the JAX NetHack reimplementation.

Composes every subsystem's state slice into a single Flax pytree, so the entire
game state can be passed through ``jax.jit`` and ``jax.lax.scan`` as one value.

Wave 1 status:
    Slices for all stubbed subsystems are wired in.  ``EnvState.default()``
    returns a fresh zero-initialised game (no map, no monsters, no items)
    suitable for proving the API surface and running pytest.

Canonical source: vendor/nethack/src/decl.c (global state declarations),
                  vendor/nethack/include/you.h (player struct),
                  vendor/nethack/include/flag.h (game flags).
"""
from __future__ import annotations
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from Nethax.nethax.subsystems.combat import CombatState
from Nethax.nethax.subsystems.magic import MagicState
from Nethax.nethax.subsystems.monster_ai import MonsterAIState, make_monster_ai_state
from Nethax.nethax.subsystems.polymorph import PolymorphState, make_polymorph_state
from Nethax.nethax.subsystems.inventory import (
    InventoryState,
    _empty_ground_items_array,
    BASE_AC,
)
from Nethax.nethax.subsystems.identification import IdentificationState
from Nethax.nethax.subsystems.traps import TrapState
from Nethax.nethax.subsystems.features import FeaturesState
from Nethax.nethax.subsystems.timer_queue import TimerState
from Nethax.nethax.subsystems.prayer import PrayerState
from Nethax.nethax.subsystems.priest import PriestState
from Nethax.nethax.subsystems.conduct import ConductState
from Nethax.nethax.subsystems.shop import ShopState
from Nethax.nethax.subsystems.quest import QuestState
from Nethax.nethax.subsystems.status_effects import StatusState
from Nethax.nethax.subsystems.scoring import ScoringState
from Nethax.nethax.subsystems.messages import MessageState
from Nethax.nethax.subsystems.containers import ContainerState
from Nethax.nethax.subsystems.engrave import EngraveState
from Nethax.nethax.subsystems.skills import SkillState
from Nethax.nethax.subsystems.digging import DigState
from Nethax.nethax.subsystems.swallow import SwallowState
from Nethax.nethax.subsystems.lighting import LightingState
from Nethax.nethax.subsystems.regions import RegionState, make_region_state
from Nethax.nethax.subsystems.worm import WormState, make_worm_state

from Nethax.nethax.dungeon.branches import (
    DungeonState,
    N_BRANCHES,
    MAX_LEVELS_PER_BRANCH,
    MAP_H,
    MAP_W,
)
from Nethax.nethax.dungeon.level_memory import LevelMemoryState, make_empty_level_memory


@struct.dataclass
class StaticParams:
    """Compile-time game-shape parameters.

    These determine pytree shapes and therefore JIT trace identity.  Changing
    one of these invalidates compiled functions.

    Defaults match NLE conventions:
        - map: 21 rows x 80 cols (NetHack's ROWNO=21, COLNO=80)
        - dungeon: 7 branches x 32 max levels per branch
    """
    map_h: int = MAP_H
    map_w: int = MAP_W
    n_branches: int = N_BRANCHES
    max_levels_per_branch: int = MAX_LEVELS_PER_BRANCH


@struct.dataclass
class EnvState:
    """Master game state — every subsystem slice plus core player fields.

    Composition principle: each subsystem owns a Flax ``struct.dataclass``
    defined in its own module.  ``EnvState`` simply aggregates them, so adding
    a new subsystem only requires (a) defining its slice and (b) adding a
    field here.

    Per-tile state lives in two places:
        * ``terrain`` / ``glyphs`` / ``explored`` arrays here  (dungeon map)
        * ``traps`` and ``features`` slices  (overlay layers)
    """
    # ---- Subsystem slices ----
    combat: CombatState
    magic: MagicState
    monster_ai: MonsterAIState
    polymorph: PolymorphState
    inventory: InventoryState
    identification: IdentificationState
    traps: TrapState
    features: FeaturesState
    prayer: PrayerState
    priest: PriestState
    conduct: ConductState
    shop: ShopState
    quest: QuestState
    status: StatusState
    scoring: ScoringState
    messages: MessageState
    dungeon: DungeonState
    level_memory: LevelMemoryState
    containers: ContainerState
    engrave: EngraveState
    skills: SkillState
    dig: DigState
    swallow: SwallowState
    lighting: LightingState
    region_state: RegionState
    worm_state: WormState
    timers: TimerState   # Wave 47f: generic per-turn timer queue
                         # (timeout.c::run_timers infrastructure;
                         # consumer callbacks in timer_queue.TIMER_CALLBACKS)

    # ---- Wave 47g: multi-turn occupation (vendor ga.afternmv) ----------
    # When occupation_kind > 0, the hero is busy with a multi-turn task
    # (e.g. doffing armor while a nymph steals).  Action dispatch checks
    # ``occupation.is_occupied(state)`` and may no-op.  When
    # ``occupation_remaining`` ticks to 0, the matching callback fires.
    # Cite: vendor/nethack/src/steal.c::stealarm + unstolenarm.
    occupation_kind: jax.Array       # int8   OccupationKind enum
    occupation_target: jax.Array     # int32  monster slot / item idx
    occupation_remaining: jax.Array  # int8   turns left

    # ---- Wave 47x: lock-picking occupation scratch ----------------------
    # Per-turn chance used by ``picklock`` (vendor lock.c::picklock line
    # 98 — rn2(100) >= gx.xlock.chance).  Captured once at occupation
    # start (lock.c:650) so role/dex changes mid-attempt don't perturb
    # the running attempt — matches vendor's gx.xlock.chance semantics.
    # Also bumped by +20 on the turn the player notices a trap on a
    # magic-keyed box (lock.c:107).
    pick_lock_chance: jax.Array      # int8   per-turn chance 0..99
    pick_lock_magic_key: jax.Array   # bool   is_magic_key(uwep) at start

    # ---- Wave 47i: calendar (vendor calendar.c / flags.moonphase) -----
    # Approximate vendor's astronomical lookup with a 250-turn cycle:
    # ``calendar_moonphase`` cycles 0..3 (new→waxing→full→waning); ``
    # calendar_friday13`` is False in default play.  Used by luck-drift
    # baseluck and were_change probability.
    # Cite: vendor/nethack/include/flag.h flags.moonphase / friday13.
    calendar_moonphase: jax.Array    # int8 in [0, 3]
    calendar_friday13: jax.Array     # bool

    # ---- Wave 47h: ball-as-projectile state (vendor ball.c::drop_ball) -
    # Tracks when the hero throws the ball at a trap to escape (vendor
    # drop_ball lines 882-961).  Once thrown, the ball is in flight for
    # ``ball_thrown_turns`` turns; on landing it can free the hero from
    # a bear trap / web / pit.
    # Cite: vendor/nethack/src/ball.c::drop_ball lines 882-961.
    ball_thrown_pos: jax.Array       # int16[2] (row, col) target tile
    ball_thrown_turns: jax.Array     # int8     turns left in flight

    # ---- NLE multi-key follow-up state machine ------------------------
    # NLE-trained policies emit certain actions as two-step sequences:
    # ``WEAR → letter('b')`` selects which inventory slot to wear.  Vendor
    # NetHack implements this via getobj() / getdir() prompts inside the
    # command handlers (vendor/nethack/src/cmd.c).  In Nethax, we expose
    # the prompts as ``PendingActionKind`` states that the dispatcher
    # checks BEFORE the normal action lookup.  When pending != NONE, the
    # incoming action argument is interpreted as the answer to the prompt
    # (a letter for inventory, a direction for getdir) and the deferred
    # handler is invoked with that arg.
    #
    # Cite: vendor/nethack/src/cmd.c::doapply, dowield, dowear, dodrink,
    #       doread, dozap, dothrow, docast (each calls getobj / getdir).
    pending_action_kind: jax.Array   # int8  PendingActionKind enum (0=NONE)
    pending_action_root: jax.Array   # int8  original action enum value
    pending_action_slot: jax.Array   # int8  inv-slot remembered for 2-step
    # Two-step ZAP/THROW: direction (dy, dx) decoded from the agent's
    # follow-up direction key.  Default (0, 0) means "no direction set"
    # — handlers fall back to the legacy east-default.  Cite:
    # vendor/nethack/src/cmd.c::getdir return value into u.dx/u.dy.
    pending_action_dir: jax.Array    # int8[2]  (dy, dx) direction follow-up

    # ---- Player core (kept here for fast access; not part of any subsystem) ----
    player_pos: jax.Array       # int16[2]  (row, col)
    player_hp: jax.Array        # int32
    player_hp_max: jax.Array    # int32
    player_pw: jax.Array        # int32
    player_pw_max: jax.Array    # int32
    player_xp: jax.Array        # int32  experience points (u.uexp)
    player_xl: jax.Array        # int32  experience level (u.ulevel)
    # u.urexp — vendor 64-bit running score accumulator (you.h:399).  Bumped
    # by ``more_experienced`` (exper.c:168-203) with 4*xp + extra rexp; used as
    # the final-score base in really_done (end.c:1325-1352) / topten.c:675.
    player_urexp: jax.Array     # int64  u.urexp
    # u.uhpinc[MAXULEV] / u.ueninc[MAXULEV] — per-level HP/Pw increments saved
    # by newhp()/newpw() so losexp() can subtract the same amount when a level
    # is drained.  Cite: vendor/nethack/include/you.h:480-481;
    # exper.c::losexp (lines 251, 269).
    # Spec uses [31] (MAXULEV+1) so index by ulevel directly without -1.
    player_uhpinc: jax.Array    # int16[31]
    player_ueninc: jax.Array    # int16[31]
    player_role: jax.Array      # int8   Role enum value
    player_race: jax.Array      # int8   Race enum value
    player_align: jax.Array     # int8   Alignment enum value
    player_str: jax.Array       # int16  raw strength (0..125)
    player_dex: jax.Array       # int8
    player_con: jax.Array       # int8
    player_int: jax.Array       # int8
    player_wis: jax.Array       # int8
    player_cha: jax.Array       # int8
    player_gold: jax.Array      # int32
    player_ac: jax.Array        # int32  armor class (10 = unarmored, lower = better)

    # ---- Player core (Wave 6 closing-audit additions; vendor u.* parity) ----
    # Citations refer to vendor/nethack/include/you.h::struct you (lines 360-510).
    player_in_trap:   jax.Array  # bool   u.utrap (vendor/nethack/src/uhitm.c:410); True while standing on a trap tile
    # ---- Per-trap utrap counter system (Audit M item #62; hack.c:1555-1690) --
    # vendor u.utrap is a 16-bit counter decremented per move; u.utraptype
    # records the trap kind so escape rules differ per trap.  Bear trap uses
    # rn1(4,4) = 4..7 initial turns; pit/web use their own initial values.
    # Cite: vendor/nethack/src/hack.c lines 1565-1675 (per-utraptype switch).
    player_trap_timer: jax.Array # int16  u.utrap counter (turns remaining)
    player_trap_type:  jax.Array # int8   u.utraptype (TrapType enum value)
    player_luck:      jax.Array  # int8   u.uluck   (you.h line 460); range [-10,10]
    player_moreluck:  jax.Array  # int8   u.moreluck (you.h line 460); luckstone bonus
    player_in_water:  jax.Array  # bool   u.uinwater (you.h line 431)
    turns_underwater: jax.Array  # int16  consecutive turns in water; reset on leaving (trap.c::drown line 5059)
    player_buried:    jax.Array  # bool   u.uburied  (you.h line 436)
    player_steed_mid: jax.Array    # uint32 u.usteed_mid (you.h line 494); 0 = not riding
    player_extra_speed: jax.Array  # int8  extra move speed while riding steed (steed.c:447 ugallop)
    saddle_condition: jax.Array    # int8  0=broken, 100=new; degrades 1/100 turns riding (steed.c)
    # u.ugallop accumulator — set by kick_steed (steed.c:448 ugallop += rn1(20,30))
    # decrements 1/turn (timeout.c:664-666); pet gallops while > 0.  Cleared on
    # dismount (steed.c:659).  int32 to mirror vendor long.
    # Cite: vendor/nethack/src/steed.c lines 401-449 + timeout.c lines 664-667.
    gallop_counter:   jax.Array    # int32  u.ugallop
    player_killer_mid: jax.Array   # uint32 last-attacker monster id (you.h: svk.killer)
    player_mortality: jax.Array  # int32  u.umortality (you.h line 497); deaths so far
    player_uhitinc:   jax.Array  # int8   u.uhitinc (you.h); ring of increase accuracy
    player_udaminc:   jax.Array  # int8   u.udaminc (you.h); ring of increase damage

    # ---- Per-stat race+role attribute maxima (vendor u.urace.attrmax[]) ----
    # int8[6]: per-stat racial/role max for STR/INT/WIS/DEX/CON/CHA (attrib.h order).
    # STR is capped at 18 in this field (the 18/** percentile range is NOT encoded
    # here — restore_ability clamps to this value).
    # Cite: vendor/nethack/src/u_init.c lines 250-580 (init_attr race cap loop);
    #       vendor/nethack/src/potion.c::peffect_restore_ability (full_restore).
    player_amax:      jax.Array  # int8[6]
    # TODO (post-Wave-6): mirror u.uintrinsic[] timed-intrinsic array and
    # u.uprops[LAST_PROP+1] property timers — currently fielded indirectly
    # through StatusState (status_effects.py); revisit when an end-to-end
    # property-timer simulation is needed.

    # ---- Artifact invoke cooldown (artifact.c::arti_invoke) ----
    # int16[30]: cooldown turns remaining per artifact slot; 0 = ready.
    # 30 slots = 22 wish-table entries + 8 headroom (synthetic idx like Magicbane=29).
    # Cite: vendor/nethack/src/artifact.c::arti_invoke artiintrinsics_taught[].
    invoke_cooldown: jax.Array  # int16[30]

    # ---- Tin-opening mechanic (eat.c:1370) ----
    tin_opening_turns_left: jax.Array  # int8  turns remaining; 0 = no tin
    tin_opening_type_id: jax.Array     # int16 type_id of the tin being opened

    # ---- Punishment / genocide state (read.c) ----
    is_punished: jax.Array        # bool  — iron ball attached
    ball_pos:    jax.Array        # int16[2]  (row, col) of iron ball
    # u.bglyph / u.cglyph — blind-hero glyph cache for the ball and chain.
    # Vendor stores the glyph that "should appear" under the ball/chain when
    # the hero is Blinded so that move_bc() can restore it after the b&c
    # leave a tile.  We store the underlying terrain TileType byte (int8)
    # which mirrors the data dependency without modelling glyph IDs.
    # Cite: vendor/nethack/src/ball.c::set_bc / move_bc lines 379-556.
    ball_under_glyph:  jax.Array  # int8  — terrain under ball when blind
    chain_under_glyph: jax.Array  # int8  — terrain under chain when blind
    genocided_species: jax.Array  # bool[381] — True = genocided

    # ---- Stinking cloud state (read.c::do_stinking_cloud) ----
    # vendor/nethack/src/read.c::do_stinking_cloud (~3082):
    #   create_gas_cloud(cc.x, cc.y, radius, turns)
    # TODO: wire per-turn tick in env.py step (decrement cloud_turns, 1HP+VOMITING
    #       to anything within cloud_radius of cloud_pos while cloud_turns > 0).
    cloud_pos:    jax.Array       # int16[2]  tile cloud is centered on
    cloud_radius: jax.Array       # int8      Chebyshev radius (vendor: 15+10*bcsign)
    cloud_turns:  jax.Array       # int8      turns remaining; 0 = inactive

    # ---- Food detection cache (read.c::seffect_food_detection) ----
    # Count of FOOD_CLASS ground items on current level at time of detection.
    last_food_count: jax.Array    # int8

    # ---- Sokoban pit counter (sokoban.c::sokoban_prize) ----
    # Number of boulders pushed into pits while in the Sokoban branch.
    # When this reaches SOKOBAN_PITS_TO_FILL (4) the prize spawns.
    # Citation: vendor/nethack/src/sokoban.c::sokoban_in_play / sokoban_prize.
    sokoban_boulders_pitted: jax.Array  # int8

    # ---- Probe result cache (apply.c::use_stethoscope / zap.c::probe_monster) ----
    # Last monster probed by stethoscope or wand of probing.
    # Cite: vendor/nethack/src/apply.c::use_stethoscope (line 318);
    #       vendor/nethack/src/zap.c::probe_monster (~line 4700).
    probed_hp:  jax.Array  # int32  HP of the last probed monster; 0 = no probe
    probed_idx: jax.Array  # int32  MonsterAIState slot index; -1 = no probe

    # ---- Unblind telepathy range (worn.c::recalc_telepat_range) ----
    # Mirrors vendor ``u.unblind_telepat_range`` (you.h:406).  Recomputed on
    # every wear/take-off as ``(BOLT_LIM * BOLT_LIM) * nobjs`` where ``nobjs``
    # is the count of currently-worn items granting TELEPAT.  ``-1`` is the
    # vendor sentinel for "no worn ESP source".
    # Cite: vendor/nethack/src/worn.c::recalc_telepat_range lines 50-69;
    #       vendor/nethack/src/u_init.c line 1021 (init -1).
    unblind_telepat_range: jax.Array  # int32

    # ---- Terrain layers (kept at top level; subsystems read but rarely write) ----
    terrain: jax.Array          # int8[N_BRANCHES, MAX_LEVELS_PER_BRANCH, MAP_H, MAP_W]  tile type
    explored: jax.Array         # bool[N_BRANCHES, MAX_LEVELS_PER_BRANCH, MAP_H, MAP_W]
    visible: jax.Array          # bool[MAP_H, MAP_W]  FOV for current level only
    # Last-seen terrain: mirrors vendor/nethack/src/display.c::lastseentyp[x][y]
    # (~line 850). Stores the terrain type last observed at each cell; -1 = never
    # seen. Off-FOV explored tiles render from this layer so that terrain changes
    # (monster opens door, digging) are not immediately visible to the player.
    last_seen_terrain: jax.Array  # int8[N_BRANCHES, MAX_LEVELS_PER_BRANCH, MAP_H, MAP_W]

    # ---- Ground items (item stack per tile) ----
    # Item[N_BRANCHES, MAX_LEVELS, MAP_H, MAP_W, MAX_GROUND_STACK]
    # Each field is an array of that shape; category==0 means empty stack entry.
    ground_items: Any  # Item pytree (flax struct)

    # ---- Game-loop bookkeeping ----
    rng: Any                    # jax.random.PRNGKey
    timestep: jax.Array         # int32
    done: jax.Array             # bool

    @classmethod
    def default(
        cls,
        rng: Any,
        static: StaticParams = StaticParams(),
    ) -> "EnvState":
        """Return a freshly initialised game state.

        Wave 1: all arrays zero-initialised, player at (0,0), no map content.
        Later waves wire in role/race choice and dungeon generation.
        """
        b, l, h, w = static.n_branches, static.max_levels_per_branch, static.map_h, static.map_w
        return cls(
            # subsystem slices
            combat=CombatState.default(),
            magic=MagicState.default(),
            monster_ai=make_monster_ai_state(),
            polymorph=make_polymorph_state(),
            inventory=InventoryState.empty(),
            identification=IdentificationState.unshuffled(),
            traps=TrapState.default(num_levels=b * l, map_h=h, map_w=w),
            features=FeaturesState.default(num_levels=b * l, map_h=h, map_w=w),
            prayer=PrayerState.default(),
            priest=PriestState.default(),
            conduct=ConductState.default(),
            shop=ShopState.default(),
            quest=QuestState.default(),
            status=StatusState.default(),
            scoring=ScoringState.default(),
            messages=MessageState.default(),
            dungeon=_default_dungeon_state(b, l),
            level_memory=make_empty_level_memory(),
            containers=ContainerState.empty(),
            engrave=EngraveState.default(map_h=h, map_w=w),
            skills=SkillState.default(),
            dig=DigState.default(),
            swallow=SwallowState.default(),
            lighting=LightingState.default(),
            region_state=make_region_state(),
            worm_state=make_worm_state(),
            timers=TimerState.default(),
            occupation_kind=jnp.int8(0),
            occupation_target=jnp.int32(-1),
            occupation_remaining=jnp.int8(0),
            pick_lock_chance=jnp.int8(0),
            pick_lock_magic_key=jnp.bool_(False),
            ball_thrown_pos=jnp.array([-1, -1], dtype=jnp.int16),
            ball_thrown_turns=jnp.int8(0),
            pending_action_kind=jnp.int8(0),
            pending_action_root=jnp.int8(0),
            pending_action_slot=jnp.int8(-1),
            pending_action_dir=jnp.zeros((2,), dtype=jnp.int8),
            calendar_moonphase=jnp.int8(0),
            calendar_friday13=jnp.bool_(False),
            # player core
            player_pos=jnp.zeros((2,), dtype=jnp.int16),
            player_hp=jnp.int32(10),
            player_hp_max=jnp.int32(10),
            player_pw=jnp.int32(0),
            player_pw_max=jnp.int32(0),
            player_xp=jnp.int32(0),
            player_xl=jnp.int32(1),
            player_urexp=jnp.int64(0),
            player_uhpinc=jnp.zeros((31,), dtype=jnp.int16),
            player_ueninc=jnp.zeros((31,), dtype=jnp.int16),
            player_role=jnp.int8(0),
            player_race=jnp.int8(0),
            player_align=jnp.int8(0),
            player_str=jnp.int16(18),
            player_dex=jnp.int8(10),
            player_con=jnp.int8(10),
            player_int=jnp.int8(10),
            player_wis=jnp.int8(10),
            player_cha=jnp.int8(10),
            player_gold=jnp.int32(0),
            player_ac=jnp.int32(BASE_AC),
            # Wave 6 closing-audit additions (vendor u.* parity).
            player_in_trap=jnp.bool_(False),
            player_trap_timer=jnp.int16(0),
            player_trap_type=jnp.int8(0),
            player_luck=jnp.int8(0),
            player_moreluck=jnp.int8(0),
            player_uhitinc=jnp.int8(0),
            player_udaminc=jnp.int8(0),
            player_in_water=jnp.bool_(False),
            turns_underwater=jnp.int16(0),
            player_buried=jnp.bool_(False),
            player_steed_mid=jnp.uint32(0),
            player_extra_speed=jnp.int8(0),
            saddle_condition=jnp.int8(100),
            gallop_counter=jnp.int32(0),
            player_killer_mid=jnp.uint32(0),
            player_mortality=jnp.int32(0),
            # per-stat attribute maxima (vendor u.urace.attrmax[]; init_attr race cap)
            # Default: 18 for all stats (human unconstrained cap).
            # Cite: vendor/nethack/src/u_init.c init_attr; potion.c peffect_restore_ability.
            player_amax=jnp.full((6,), 18, dtype=jnp.int8),
            # artifact invoke cooldown
            invoke_cooldown=jnp.zeros((30,), dtype=jnp.int16),
            # tin-opening mechanic
            tin_opening_turns_left=jnp.int8(0),
            tin_opening_type_id=jnp.int16(0),
            # punishment / genocide
            is_punished=jnp.bool_(False),
            ball_pos=jnp.zeros((2,), dtype=jnp.int16),
            ball_under_glyph=jnp.int8(0),
            chain_under_glyph=jnp.int8(0),
            genocided_species=jnp.zeros((381,), dtype=jnp.bool_),
            # stinking cloud (read.c::do_stinking_cloud)
            cloud_pos=jnp.zeros((2,), dtype=jnp.int16),
            cloud_radius=jnp.int8(0),
            cloud_turns=jnp.int8(0),
            # food detection cache (read.c::seffect_food_detection)
            last_food_count=jnp.int8(0),
            # sokoban pit counter (sokoban.c::sokoban_prize)
            sokoban_boulders_pitted=jnp.int8(0),
            # probe result cache (apply.c::use_stethoscope / zap.c::probe_monster)
            probed_hp=jnp.int32(0),
            probed_idx=jnp.int32(-1),
            # unblind telepathy range — vendor sentinel -1 = none
            # (vendor/nethack/src/u_init.c:1021)
            unblind_telepat_range=jnp.int32(-1),
            # terrain layers
            terrain=jnp.zeros((b, l, h, w), dtype=jnp.int8),
            explored=jnp.zeros((b, l, h, w), dtype=jnp.bool_),
            visible=jnp.zeros((h, w), dtype=jnp.bool_),
            last_seen_terrain=jnp.full((b, l, h, w), -1, dtype=jnp.int8),
            # ground items
            ground_items=_empty_ground_items_array(b, l, h, w),
            # game loop
            rng=rng,
            timestep=jnp.int32(0),
            done=jnp.bool_(False),
        )


def _default_dungeon_state(n_branches: int, max_levels: int) -> DungeonState:
    """Build a fresh DungeonState with empty branch graph."""
    return DungeonState(
        branch_levels=jnp.zeros((n_branches,), dtype=jnp.int8),
        current_branch=jnp.int8(0),
        current_level=jnp.int8(1),
        stair_links=jnp.full((n_branches, max_levels, 2, 2), -1, dtype=jnp.int8),
        level_rng_seeds=jnp.zeros((n_branches, max_levels), dtype=jnp.uint32),
        vibrating_square_revealed=jnp.bool_(False),
        vibrating_square_pos=jnp.full((2,), -1, dtype=jnp.int16),
        lit_radius_until_turn=jnp.int32(-1),
        portal_destination=jnp.full((n_branches, max_levels, 2), -1, dtype=jnp.int8),
    )
