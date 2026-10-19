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
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_DATA_DIR = Path(__file__).resolve().parent / "data"
_DEFAULT_LEXICON_PATH = _DATA_DIR / "enable_2to15.txt.gz"
_CSW_TWOS_SUPPLEMENT_PATH = _DATA_DIR / "csw_twos_supplement.txt"
_LEXICONS_DIR = Path(__file__).resolve().parents[3] / "configs" / "lexicons"

_ENV_VAR = "AUTOSCORER_LEXICON"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LexiconInfo:
    """Provenance for whichever lexicon actually got loaded -- travels
    with an eval report's `Provenance` so a number is never compared
    across two runs that silently used different word lists."""
    name: str
    path: str
    word_count: int
    is_default: bool
    requested: Optional[str] = None  # set when a named lexicon was missing and this is the fallback


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


def _looks_like_path(requested: str) -> bool:
    return os.sep in requested or "/" in requested or requested.endswith((".txt", ".gz"))


def load_lexicon(name_or_path: Optional[str] = None) -> Lexicon:
    """Resolves and loads a lexicon, in this order:

    1. `name_or_path`, if given. Either a bare name looked up under
       `configs/lexicons/<name>.txt` or `<name>.txt.gz`, or a direct
       filesystem path.
    2. The `AUTOSCORER_LEXICON` environment variable, same resolution.
    3. The vendored default (`data/enable_2to15.txt.gz`).

    `configs/lexicons/` is gitignored, so a licensed list such as
    `configs/lexicons/csw24.txt` is used when present and never committed.

    A missing direct path raises `FileNotFoundError`. A missing bare name
    logs a warning and falls back to the vendored default, so a venue
    profile can name a licensed list that only some machines have. When
    the missing name starts with `csw`, the fallback also includes the
    CSW 2-letter words ENABLE lacks (`data/csw_twos_supplement.txt`).
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
        if candidate_path.exists():
            words = _read_words(candidate_path)
            info = LexiconInfo(
                name=candidate_path.stem.removesuffix(".txt"),
                path=str(candidate_path), word_count=len(words), is_default=False,
            )
            return Lexicon(words, info)
        if _looks_like_path(requested):
            raise FileNotFoundError(f"lexicon file {requested!r} not found")
        return _load_fallback(requested)

    return _load_default()


def _load_default() -> Lexicon:
    words = _read_words(_DEFAULT_LEXICON_PATH)
    info = LexiconInfo(
        name="enable_2to15", path=str(_DEFAULT_LEXICON_PATH),
        word_count=len(words), is_default=True,
    )
    return Lexicon(words, info)


def _load_fallback(requested: str) -> Lexicon:
    words = _read_words(_DEFAULT_LEXICON_PATH)
    name = "enable_2to15"
    if requested.lower().startswith("csw"):
        words = words | _read_words(_CSW_TWOS_SUPPLEMENT_PATH)
        name = "enable_2to15+csw_twos"
    logger.warning(
        "lexicon %r not found in %s; falling back to %s (%d words). "
        "Real %s words absent from this list will be flagged as invalid. "
        "Drop the licensed list at %s to use it.",
        requested, _LEXICONS_DIR, name, len(words), requested,
        _LEXICONS_DIR / f"{requested}.txt",
    )
    info = LexiconInfo(
        name=name, path=str(_DEFAULT_LEXICON_PATH), word_count=len(words),
        is_default=True, requested=requested,
    )
    return Lexicon(words, info)
