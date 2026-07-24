from autoscorer.gamelogic.dictionary.lookup import InMemoryDictionary


def test_valid_word_lookup_is_case_insensitive():
    d = InMemoryDictionary({"cat": "a small domesticated feline"})
    assert d.is_valid_word("CAT")
    assert d.is_valid_word("cat")
    assert d.definition("Cat") == "a small domesticated feline"


def test_unknown_word_is_invalid_with_no_definition():
    d = InMemoryDictionary({"cat": "a small domesticated feline"})
    assert not d.is_valid_word("zzyzx")
    assert d.definition("zzyzx") is None
