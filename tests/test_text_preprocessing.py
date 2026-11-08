import pandas as pd
import pytest

from src.text_preprocessing import clean_text


@pytest.mark.parametrize(
    "value",
    [None, pd.NA, float("nan")]
)
def test_missing_values_return_empty_string(value):
    assert clean_text(value) == ""


def test_text_is_converted_to_lowercase():
    assert clean_text("HELLO WORLD") == "hello world"


def test_urls_are_removed():
    result = clean_text(
        "Visit https://example.com for details"
    )

    assert "https" not in result
    assert "example.com" not in result
    assert "visit" in result


def test_html_tags_are_removed():
    result = clean_text("<p>Hello <b>World</b></p>")

    assert result == "hello world"


def test_unicode_languages_are_preserved():
    result = clean_text("Hello 中文 العربية বাংলা")

    assert "中文" in result
    assert "العربية" in result
    assert "বাংলা" in result


def test_unicode_compatibility_normalization():
    result = clean_text("Ｈｅｌｌｏ")

    assert result == "hello"


def test_emoji_is_converted_to_readable_text():
    result = clean_text("👍")

    assert "thumbs up" in result


def test_extra_spaces_are_removed():
    result = clean_text("  hello     world   ")

    assert result == "hello world"


def test_numbers_and_financial_symbols_are_preserved():
    result = clean_text("Profit increased by 12.5% to $500")

    assert "12.5%" in result
    assert "$500" in result