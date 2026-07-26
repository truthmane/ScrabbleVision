import gzip

import pytest

from autoscorer.gamelogic.dictionary.lexicon import (
    _DEFAULT_LEXICON_PATH,
    Lexicon,
    load_lexicon,
)


def test_default_lexicon_loads_with_the_documented_word_count():
    lexicon = load_lexicon()
    assert lexicon.word_count == 168551
    assert lexicon.info.is_default is True
    assert lexicon.name == "enable_2to15"


def test_default_lexicon_contains_sanity_words():
    lexicon = load_lexicon()
    for word in ["CAT", "DOG", "SCRABBLE", "AA", "AX", "OX", "EX"]:
        assert word in lexicon, f"expected {word!r} to be a real word"


def test_default_lexicon_rejects_obvious_nonwords():
    lexicon = load_lexicon()
    for nonword in ["ZZQX", "QXZZ", "NOTAWORDATALL"]:
        assert nonword not in lexicon


def test_membership_is_case_insensitive():
    lexicon = load_lexicon()
    assert lexicon.is_valid_word("cat")
    assert lexicon.is_valid_word("Cat")
    assert lexicon.is_valid_word("CAT")


def test_definition_is_always_none():
    lexicon = load_lexicon()
    assert lexicon.definition("CAT") is None


def test_satisfies_the_dictionary_lookup_protocol():
    from autoscorer.gamelogic.dictionary.lookup import DictionaryLookup

    lexicon = load_lexicon()
    assert isinstance(lexicon, DictionaryLookup)


def test_load_lexicon_by_explicit_path(tmp_path):
    custom = tmp_path / "custom.txt"
    custom.write_text("HELLO\nWORLD\n")

    lexicon = load_lexicon(str(custom))

    assert lexicon.word_count == 2
    assert "HELLO" in lexicon
    assert lexicon.info.is_default is False


def test_load_lexicon_by_explicit_gzipped_path(tmp_path):
    custom = tmp_path / "custom.txt.gz"
    with gzip.open(custom, "wt") as f:
        f.write("FOO\nBAR\n")

    lexicon = load_lexicon(str(custom))

    assert lexicon.word_count == 2
    assert "FOO" in lexicon


def test_load_lexicon_from_configs_lexicons_by_name(tmp_path, monkeypatch):
    import autoscorer.gamelogic.dictionary.lexicon as lexicon_module

    lexicons_dir = tmp_path / "lexicons"
    lexicons_dir.mkdir()
    (lexicons_dir / "mylex.txt").write_text("ALPHA\nBETA\n")
    monkeypatch.setattr(lexicon_module, "_LEXICONS_DIR", lexicons_dir)

    lexicon = load_lexicon("mylex")

    assert lexicon.word_count == 2
    assert "ALPHA" in lexicon
    assert lexicon.name == "mylex"


def test_load_lexicon_via_env_var(monkeypatch, tmp_path):
    custom = tmp_path / "envlex.txt"
    custom.write_text("ENVWORD\n")
    monkeypatch.setenv("AUTOSCORER_LEXICON", str(custom))

    lexicon = load_lexicon()

    assert "ENVWORD" in lexicon
    assert lexicon.info.is_default is False


def test_load_lexicon_raises_for_unknown_name():
    with pytest.raises(FileNotFoundError):
        load_lexicon("definitely_does_not_exist_anywhere")


def test_default_lexicon_path_exists_on_disk():
    assert _DEFAULT_LEXICON_PATH.exists()
