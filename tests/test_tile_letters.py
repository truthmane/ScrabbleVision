import pytest

from autoscorer.gamelogic.board import Tile


@pytest.mark.parametrize("letter", ["É", "ß", "Ω", "1", "AB"])
def test_tile_rejects_non_ascii_letters(letter):
    with pytest.raises(ValueError):
        Tile(letter)


def test_tile_uppercases_ascii():
    assert Tile("q").letter == "Q"
