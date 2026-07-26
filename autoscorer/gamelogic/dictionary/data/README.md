# `enable_2to15.txt.gz`

The vendored default word list for `autoscorer.gamelogic.dictionary.lexicon`.

- **Source**: ENABLE (Enhanced North American Benchmark Lexicon), fetched from
  https://raw.githubusercontent.com/dolph/dictionary/master/enable1.txt
  (172,823 raw entries).
- **Licence**: public domain. ENABLE was compiled by Alan Beale as an
  explicitly public-domain word list for word-game use (building on the
  public-domain Moby word lists); it is widely redistributed unmodified for
  exactly this purpose (e.g. by Words with Friends-style implementations).
- **Filter applied**: kept only entries with 2-15 alphabetic ASCII characters
  (matching the board's 15x15 max word length and Scrabble's 2-letter-word
  floor), uppercased, deduplicated, sorted.
- **Resulting count**: 168,551 words.
- **SHA256 of the filtered, uncompressed word list**:
  `0d6d47327b05142362f52844113497d0439bd0a7fc8d0bc33ee40ecff0216ac1`

This is deliberately a generic English word list, not a real tournament
lexicon (CSW/NWL) -- measured coverage against the 7 real WESPA Word Wars
games in `tests/fixtures/`: 35% of the words those real games actually
formed (98 of 282) are absent from a generic list, since Scrabble words
include many valid short words and CSW-specific entries a general dictionary
doesn't carry. Concretely, this exact list is even missing some short words
that ARE valid tournament Scrabble plays (e.g. `QI`, `ZA`) -- found while
writing this module's own sanity-word test. **A licensed lexicon should be
used for real accuracy work** (see `lexicon.load_lexicon`'s resolution
order) -- this vendored list exists so the loader, the decoder, and the
never-reject-a-reading safety property all have a default that requires no
external file to test and exercise.

A test (`tests/unit/test_lexicon.py`) asserts the word count and a handful
of sanity words, so a corrupted or accidentally-swapped file fails loudly
rather than silently degrading decoding.
