"""Doctor-name SQL patterns must start with the name, not its title."""

from app.services.text_helpers import _arabic_like_pattern, _strip_dr_prefix


def test_doctor_prefix_without_space_after_slash_is_removed():
    assert _strip_dr_prefix("د/محمد نصار") == "محمد نصار"
    assert _arabic_like_pattern("د/محمد نصار", first_word_only=True) == "محمد%"


def test_doctor_prefix_without_space_after_dot_is_removed():
    assert _arabic_like_pattern("د.محمد نصار", first_word_only=True) == "محمد%"


def test_parenthesized_doctor_name_uses_name_as_first_word():
    assert _arabic_like_pattern("د/ ( محمد نصار )", first_word_only=True) == "محمد%"


def test_name_starting_with_dal_is_not_treated_as_title():
    assert _strip_dr_prefix("داليا حسن") == "داليا حسن"
