"""Word validity + definitions lookup. Explicitly out of scope for v1
scoring (word legality at a tournament is a human word-judge's call), but
the interface exists now so it can be wired to `WordPlayedEvent`s later
without touching the scoring/move-detection pipeline.

The bundled `InMemoryDictionary` is a placeholder for development/testing --
production deployment should supply a real word list (e.g. TWL or CSW) and
a definitions source via the same interface.
"""
from __future__ import annotations

from typing import Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class DictionaryLookup(Protocol):
    def is_valid_word(self, word: str) -> bool: ...

    def definition(self, word: str) -> Optional[str]: ...


class InMemoryDictionary:
    """A small, explicitly non-exhaustive word list for development and
    unit testing. Do not use this for real tournament word validation.
    """

    def __init__(self, entries: Optional[Dict[str, str]] = None) -> None:
        self._entries: Dict[str, str] = {
            word.upper(): definition for word, definition in (entries or {}).items()
        }

    def is_valid_word(self, word: str) -> bool:
        return word.upper() in self._entries

    def definition(self, word: str) -> Optional[str]:
        return self._entries.get(word.upper())
