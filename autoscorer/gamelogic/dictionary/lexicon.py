"""Word-list backed lexicon for lexicon-constrained decoding
(`autoscorer.gamelogic.movedetect.lexicon_decoder`).

**Phonies are legal, scoring Scrabble plays** -- a word that isn't in any
dictionary, played and left unchallenged, scores exactly like a real one.
That means a lexicon here can never be a hard filter: it may only ever
re-rank candidate readings, never reject or silently substitute one. See
`lexicon_decoder.py` for where that principle actually gets enforced;
this module is just the word-membership data structure it ranks against.

Data structure: a plain `frozenset[str]`. The only two queries this
module (or the decoder built on it) ever needs are exact membership
("is this string a word?") and, for blank resolution, "which of 26
letters completes this pattern?" -- 26 exact lookups. A DAWG/GADDAG
buys nothing for either of those; that structure earns its complexity
for move *generation* over a rack, which this project never does.
~170k words as a `frozenset` is a few tens of MB and O(1) lookups, with
zero supporting code.
"""
from __future__ import annotations

import gzip
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_DATA_DIR = Path(__file__).resolve().parent / "data"
_DEFAULT_LEXICON_PATH = _DATA_DIR / "enable_2to15.txt.gz"
_LEXICONS_DIR = Path(__file__).resolve().parents[3] / "configs" / "lexicons"

_ENV_VAR = "AUTOSCORER_LEXICON"


@dataclass(frozen=True)
class LexiconInfo:
    """Provenance for whichever lexicon actually got loaded -- travels
    with an eval report's `Provenance` so a number is never compared
    across two runs that silently used different word lists."""
    name: str
    path: str
    word_count: int
    is_default: bool


class Lexicon:
    """A word list wrapped for membership queries. Satisfies the
    existing `autoscorer.gamelogic.dictionary.lookup.DictionaryLookup`
    Protocol (`is_valid_word`/`definition`) so it's a drop-in wherever
    that's expected -- `definition` always returns `None` here; this
    module answers "is it a word," not "what does it mean."
    """

    def __init__(self, words: frozenset, info: LexiconInfo) -> None:
        self._words = words
        self.info = info

    def __contains__(self, word: str) -> bool:
        return word.upper() in self._words

    def is_valid_word(self, word: str) -> bool:
        return word.upper() in self._words

    def definition(self, word: str) -> Optional[str]:
        return None

    @property
    def word_count(self) -> int:
        return len(self._words)

    @property
    def name(self) -> str:
        return self.info.name


def _read_words(path: Path) -> frozenset:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return frozenset(line.strip().upper() for line in f if line.strip())


def load_lexicon(name_or_path: Optional[str] = None) -> Lexicon:
    """Resolves and loads a lexicon, in this order:

    1. `name_or_path`, if given -- either a bare name looked up under
       `configs/lexicons/<name>.txt` or `<name>.txt.gz`, or a direct
       filesystem path if it points at an existing file.
    2. The `AUTOSCORER_LEXICON` environment variable, same resolution.
    3. `configs/lexicons/` is otherwise untouched -- gitignored, so a
       licensed CSW21/NWL2023 drop-in (e.g. `configs/lexicons/csw24.txt`)
       never needs to be committed to use here.
    4. The vendored default (`data/enable_2to15.txt.gz`) -- always
       present, always loadable, no external file required.
    """
    requested = name_or_path or os.environ.get(_ENV_VAR)
    if requested:
        candidate_path = Path(requested)
        if not candidate_path.exists():
            for suffix in (".txt", ".txt.gz"):
                maybe = _LEXICONS_DIR / f"{requested}{suffix}"
                if maybe.exists():
                    candidate_path = maybe
                    break
        if not candidate_path.exists():
            raise FileNotFoundError(
                f"lexicon {requested!r} not found (looked in {_LEXICONS_DIR} and as a direct path)"
            )
        words = _read_words(candidate_path)
        info = LexiconInfo(
            name=candidate_path.stem.removesuffix(".txt"),
            path=str(candidate_path), word_count=len(words), is_default=False,
        )
        return Lexicon(words, info)

    words = _read_words(_DEFAULT_LEXICON_PATH)
    info = LexiconInfo(
        name="enable_2to15", path=str(_DEFAULT_LEXICON_PATH),
        word_count=len(words), is_default=True,
    )
    return Lexicon(words, info)
