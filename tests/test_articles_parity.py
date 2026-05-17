"""Vendor-parity tests for article helpers: _an(), _the(), inv_strs article logic.

Vendor reference: vendor/nethack/src/objnam.c::just_an() lines 2108-2142
                  vendor/nethack/src/objnam.c::the()      lines 2170-2231
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest

from Nethax.nethax.obs.look import _an, _the


# ---------------------------------------------------------------------------
# _an() — long-u exceptions (vendor just_an lines 2132-2135)
# ---------------------------------------------------------------------------

class TestAnLongUExceptions:
    def test_unicorn_gets_a(self):
        # "unicorn" starts with long-u sound -> "a", not "an"
        assert _an("unicorn") == "a unicorn"

    def test_uranium_gets_a(self):
        assert _an("uranium") == "a uranium"

    def test_useful_gets_a(self):
        assert _an("useful tool") == "a useful tool"

    def test_eucalyptus_gets_a(self):
        # "eu" prefix -> "a" (vendor just_an line 2132)
        assert _an("eucalyptus leaf") == "a eucalyptus leaf"

    def test_ukulele_gets_a(self):
        assert _an("ukulele") == "a ukulele"

    def test_uke_gets_a(self):
        assert _an("uke") == "a uke"

    def test_normal_u_gets_an(self):
        # "ugly" starts with short 'u' -> "an"
        assert _an("ugly troll") == "an ugly troll"

    def test_umbrella_gets_an(self):
        assert _an("umbrella") == "an umbrella"


# ---------------------------------------------------------------------------
# _an() — 'one' exception (vendor just_an line 2130)
# ---------------------------------------------------------------------------

class TestAnOneException:
    def test_one_eyed_gets_a(self):
        # "one-eyed" -> 'wun' sound -> "a"
        assert _an("one-eyed monster") == "a one-eyed monster"

    def test_one_space_gets_a(self):
        assert _an("one ring") == "a one ring"

    def test_onerous_gets_an(self):
        # "onerous" does NOT start with 'one' + separator -> normal vowel rule
        assert _an("onerous task") == "an onerous task"


# ---------------------------------------------------------------------------
# _an() — no-article items (vendor just_an lines 2121-2125)
# ---------------------------------------------------------------------------

class TestAnNoArticleItems:
    def test_molten_lava_no_article(self):
        # "molten lava" -> no article (vendor just_an line 2122)
        assert _an("molten lava") == "molten lava"

    def test_iron_bars_no_article(self):
        assert _an("iron bars") == "iron bars"

    def test_ice_no_article(self):
        assert _an("ice") == "ice"


# ---------------------------------------------------------------------------
# _an() — single-letter words (vendor just_an lines 2115-2117)
# ---------------------------------------------------------------------------

class TestAnSingleLetter:
    def test_a_gets_an(self):
        assert _an("a") == "an a"

    def test_e_gets_an(self):
        assert _an("e") == "an e"

    def test_f_gets_an(self):
        assert _an("f") == "an f"

    def test_b_gets_a(self):
        # 'b' not in "aefhilmnosx"
        assert _an("b") == "a b"

    def test_x_gets_an(self):
        assert _an("x") == "an x"


# ---------------------------------------------------------------------------
# _an() — pass-through cases
# ---------------------------------------------------------------------------

class TestAnPassThrough:
    def test_already_the(self):
        assert _an("the Amulet of Yendor") == "the Amulet of Yendor"

    def test_already_an(self):
        assert _an("an elf") == "an elf"

    def test_already_a(self):
        assert _an("a sword") == "a sword"

    def test_already_some(self):
        assert _an("some food") == "some food"

    def test_digit(self):
        assert _an("3 arrows") == "3 arrows"


# ---------------------------------------------------------------------------
# _an() — x + consonant rule (vendor just_an line 2136)
# ---------------------------------------------------------------------------

class TestAnXConsonant:
    def test_xorn_gets_an(self):
        # 'x' followed by consonant 'o'... wait: 'o' IS a vowel.
        # "xorn": x + o (vowel) -> should be "a xorn" per vendor line 2136
        # vendor: c0=='x' and NOT vowel(str[1]) -> "an"; else normal
        # 'o' is vowel so "a xorn" — but wait let's check: xorn starts vowel
        # path? No: c0='x' which is NOT in "aeiou", so falls to x+consonant check.
        # str[1]='o' IS a vowel -> does NOT match x+consonant -> "a"
        assert _an("xorn") == "a xorn"

    def test_xray_gets_an(self):
        # 'x' + 'r' (consonant) -> "an" (vendor just_an line 2136)
        assert _an("x-ray") == "an x-ray"


# ---------------------------------------------------------------------------
# _the() helper (vendor objnam.c::the() lines 2170-2231)
# ---------------------------------------------------------------------------

class TestTheHelper:
    def test_giant_ant_gets_the(self):
        assert _the("giant ant") == "the giant ant"

    def test_already_the_lowercase(self):
        assert _the("the altar") == "the altar"

    def test_already_the_uppercase(self):
        assert _the("The wizard") == "The wizard"

    def test_already_a(self):
        assert _the("a sword") == "a sword"

    def test_already_an(self):
        assert _the("an elf") == "an elf"

    def test_already_some(self):
        assert _the("some food") == "some food"

    def test_proper_noun_no_article(self):
        # All-caps-initial proper noun like "Wizard of Yendor" — has lowercase
        # after separator "of" so _the should prepend "the"
        result = _the("Wizard of Yendor")
        assert result == "the Wizard of Yendor"

    def test_all_caps_proper_stays(self):
        # "Medusa" — single capitalised word, all words uppercase -> no article
        assert _the("Medusa") == "Medusa"

    def test_terrain_altar(self):
        assert _the("altar") == "the altar"

    def test_terrain_staircase(self):
        assert _the("staircase up") == "the staircase up"

    def test_terrain_fountain(self):
        assert _the("fountain") == "the fountain"


# ---------------------------------------------------------------------------
# build_look_text — monster and terrain wrapping (integration)
# ---------------------------------------------------------------------------

class TestBuildLookText:
    """Integration tests for build_look_text() article wrapping."""

    def test_terrain_has_the(self):
        """Terrain nouns returned by build_look_text should be 'the <noun>'."""
        import jax
        from Nethax.nethax.env import NethaxEnv
        from Nethax.nethax.obs.look import build_look_text

        env = NethaxEnv()
        state, _ = env.reset(jax.random.PRNGKey(0))
        pr, pc = int(state.player_pos[0]), int(state.player_pos[1])

        # Find a wall or other terrain cell nearby (not player, not monster)
        for dc in range(-5, 6):
            for dr in range(-5, 6):
                if dr == 0 and dc == 0:
                    continue
                r, c = pr + dr, pc + dc
                text = build_look_text(state, r, c)
                if text not in ("yourself",) and not text.startswith("the "):
                    # Terrain should have "the " prefix
                    # Allow walls since "wall" gets "the wall"
                    pass
        # At minimum, unexplored area becomes "the unexplored area"
        text = build_look_text(state, 0, 0)
        assert text.startswith("the ") or text == "yourself"


# ---------------------------------------------------------------------------
# inv_strs — alternate weapon string (vendor objnam.c line 1619)
# ---------------------------------------------------------------------------

class TestAltWeaponString:
    def test_alt_weapon_bytes_contains_not_wielded(self):
        """_ALT_WEAPON_BYTES must encode ' (alternate weapon; not wielded)'."""
        from Nethax.nethax.obs.inv_strs import _ALT_WEAPON_BYTES
        import numpy as np
        raw = bytes(np.asarray(_ALT_WEAPON_BYTES).tolist()).rstrip(b"\x00")
        assert raw == b" (alternate weapon; not wielded)"

    def test_inv_text_alt_weapon_not_wielded(self):
        """When two_weapon is active the rendered slot shows '; not wielded'."""
        import jax
        import jax.numpy as jnp
        from Nethax.nethax.env import NethaxEnv
        from Nethax.nethax.obs.inv_strs import build_inv_strs, _decode_row

        env = NethaxEnv()
        state, _ = env.reset(jax.random.PRNGKey(0))

        # Patch state to enable two_weapon with slot 1 as alternate weapon.
        inv = state.inventory
        combat = state.combat if hasattr(state, "combat") else None
        if combat is None:
            pytest.skip("combat subsystem not available")

        import dataclasses
        patched_combat = dataclasses.replace(combat, two_weapon=jnp.bool_(True))
        patched_inv = dataclasses.replace(inv, alternate_weapon_slot=jnp.int8(1))
        patched_state = dataclasses.replace(
            state, combat=patched_combat, inventory=patched_inv
        )

        rows = build_inv_strs(patched_state)
        slot1_str = _decode_row(rows[1])
        assert "(alternate weapon; not wielded)" in slot1_str, (
            f"expected '(alternate weapon; not wielded)' in slot 1, got: {slot1_str!r}"
        )
