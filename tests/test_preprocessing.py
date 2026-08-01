from src.preprocessing import (
    to_lowercase,
    remove_punctuation_and_numbers,
    tokenize,
    remove_stopwords,
    stem_words,
    preprocess_text,
)


def test_to_lowercase():
    assert to_lowercase("WINNER!!") == "winner!!"


def test_remove_punctuation_and_numbers():
    assert remove_punctuation_and_numbers("winner!! £900 prize!") == "winner  prize"


def test_tokenize():
    result = tokenize("winner prize reward")
    assert result == ["winner", "prize", "reward"]


def test_remove_stopwords():
    result = remove_stopwords(["winner", "you", "have", "a", "prize"])
    assert result == ["winner", "prize"]


def test_stem_words():
    result = stem_words(["selected", "winning", "prizes"])
    assert result == ["select", "win", "prize"]


def test_preprocess_text_full_pipeline():
    result = preprocess_text("WINNER!! You have been selected to receive a prize!")
    assert result == "winner select receiv prize"


def test_preprocess_text_handles_empty_string():
    result = preprocess_text("")
    assert result == ""