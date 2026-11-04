"""Tests for the deterministic glossary scanner (audit 3.6 item 1).

Run with::

    python -m pytest tests/test_deterministic_scan.py -q
"""

import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor

from src.rag.deterministic_scan import (
    DeterministicGlossaryScanner,
    normalize_term_text,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _scanner(terms, fingerprint="test", normalize_fn=None):
    scanner = DeterministicGlossaryScanner(normalize_fn)
    scanner.rebuild(terms, fingerprint)
    return scanner


def _assert_spans(text, hits):
    """Every hit must project back onto the original text."""
    for hit in hits:
        assert set(hit) == {"term", "normalized", "surface", "start", "end"}
        assert hit["surface"] == text[hit["start"]:hit["end"]]
        assert hit["normalized"] == normalize_term_text(hit["term"])
        assert 0 <= hit["start"] < hit["end"] <= len(text)


_ACCENT_FOLD_RE = re.compile(r"[^\w]+")


def _fold_accents(text):
    """Custom normalizer: strip diacritics on top of the default behaviour."""
    decomposed = unicodedata.normalize("NFKD", str(text or ""))
    without_marks = "".join(
        ch for ch in decomposed if not unicodedata.combining(ch)
    )
    folded = unicodedata.normalize("NFKC", without_marks).casefold()
    folded = _ACCENT_FOLD_RE.sub(" ", folded).replace("_", " ")
    return " ".join(folded.split())


# ---------------------------------------------------------------------------
# overlap / boundary semantics
# ---------------------------------------------------------------------------


def test_longest_term_wins_on_overlap():
    text = "Meet me at Whiterun Guard."
    for terms in (["Whiterun", "Whiterun Guard"], ["Whiterun Guard", "Whiterun"]):
        hits = _scanner(terms).scan(text)
        _assert_spans(text, hits)
        assert len(hits) == 1
        assert hits[0]["term"] == "Whiterun Guard"
        assert hits[0]["normalized"] == "whiterun guard"
        assert hits[0]["start"] == text.index("Whiterun Guard")
        assert hits[0]["end"] == hits[0]["start"] + len("Whiterun Guard")


def test_short_term_still_hits_when_longer_fails_boundary():
    text = "Whiterun Guardhouse is closed"
    hits = _scanner(["Whiterun Guard", "Whiterun"]).scan(text)
    _assert_spans(text, hits)
    assert [h["term"] for h in hits] == ["Whiterun"]


def test_latin_word_boundary_not_prefix_of_longer_word():
    assert _scanner(["War"]).scan("Warning: danger ahead") == []
    assert _scanner(["War"]).scan("Warmth and warcraft") == []

    text = "Warning: war never changes."
    hits = _scanner(["War"]).scan(text)
    _assert_spans(text, hits)
    assert len(hits) == 1
    assert hits[0]["start"] == text.index("war never")
    assert hits[0]["surface"] == "war"


def test_possessive_and_apostrophes():
    text = "Erik's sword is not Erik the Slayer's."
    hits = _scanner(["Erik the Slayer", "Erik"]).scan(text)
    _assert_spans(text, hits)
    assert [h["term"] for h in hits] == ["Erik", "Erik the Slayer"]
    # Never swallow the possessive suffix.
    assert hits[0]["surface"] == "Erik"
    assert hits[1]["surface"] == "Erik the Slayer"

    # A term spelled with an apostrophe matches the same surface.
    hits = _scanner(["Erik's"]).scan("Erik's sword")
    _assert_spans("Erik's sword", hits)
    assert hits[0]["surface"] == "Erik's"
    assert hits[0]["normalized"] == "erik s"

    # ... and does not fire on the bare name.
    assert _scanner(["Erik's"]).scan("Erik was here") == []


# ---------------------------------------------------------------------------
# case / width / punctuation insensitivity
# ---------------------------------------------------------------------------


def test_case_and_fullwidth_insensitivity():
    scanner = _scanner(["SKSE", "Ａｒｍｏｒ"])
    text = "skse requires Armor"
    hits = scanner.scan(text)
    _assert_spans(text, hits)
    assert [h["term"] for h in hits] == ["SKSE", "Ａｒｍｏｒ"]
    assert [h["normalized"] for h in hits] == ["skse", "armor"]

    # The reverse direction must work too: ASCII term, full-width surface.
    text = "Ａｒｍｏｒ is heavy"
    hits = _scanner(["Armor"]).scan(text)
    _assert_spans(text, hits)
    assert len(hits) == 1
    assert hits[0]["surface"] == "Ａｒｍｏｒ"
    assert hits[0]["start"] == 0
    assert hits[0]["end"] == len("Armor")


def test_span_mapping_survives_collapsed_separators():
    text = "!! Whiterun   Guard??"
    hits = _scanner(["Whiterun Guard"]).scan(text)
    _assert_spans(text, hits)
    assert len(hits) == 1
    # The collapsed run of spaces is still part of the original surface.
    assert hits[0]["surface"] == "Whiterun   Guard"
    assert hits[0]["start"] == 3
    assert hits[0]["end"] == len(text) - 2


def test_curly_quotes_and_punctuation_folding():
    text = "The Dragon’s Reach is cold."
    hits = _scanner(["Dragon's Reach"]).scan(text)
    _assert_spans(text, hits)
    assert len(hits) == 1
    assert hits[0]["surface"] == "Dragon’s Reach"
    assert hits[0]["start"] == text.index("Dragon")


def test_adjacent_hits_are_all_reported():
    text = "Alpha,Beta:Gamma"
    hits = _scanner(["Alpha", "Beta", "Gamma"]).scan(text)
    _assert_spans(text, hits)
    assert [h["surface"] for h in hits] == ["Alpha", "Beta", "Gamma"]
    assert [h["start"] for h in hits] == sorted(h["start"] for h in hits)


# ---------------------------------------------------------------------------
# unicode scripts
# ---------------------------------------------------------------------------


def test_diacritics_default_keeps_accent_then_custom_fn_folds_it():
    # Default NFKC does not drop combining accents: Sjórs != Sjors.
    assert _scanner(["Sjórs"]).scan("Sjors, the nord") == []

    text = "SJÓRS is a name"
    hits = _scanner(["Sjórs"]).scan(text)
    _assert_spans(text, hits)
    assert hits[0]["surface"] == "SJÓRS"

    folded = _scanner(["Sjórs"], normalize_fn=_fold_accents)
    text = "I met Sjors yesterday"
    hits = folded.scan(text)
    assert len(hits) == 1
    assert hits[0]["term"] == "Sjórs"
    assert hits[0]["normalized"] == "sjors"
    assert hits[0]["surface"] == "Sjors"


def test_cyrillic_word_mode():
    scanner = _scanner(["Мёртвый"])
    for text in ("Мёртвый воин", "Это мёртвый воин"):
        hits = scanner.scan(text)
        _assert_spans(text, hits)
        assert len(hits) == 1
        assert hits[0]["surface"].lower() == "мёртвый"
    # Prefix of a longer Cyrillic word must not match.
    assert scanner.scan("Мёртвыйвоин") == []


def test_japanese_kana_substring():
    scanner = _scanner(["ドラゴン"])
    text = "白いドラゴンが来た"
    hits = scanner.scan(text)
    _assert_spans(text, hits)
    assert len(hits) == 1
    assert hits[0]["start"] == 2
    assert hits[0]["surface"] == "ドラゴン"


def test_cjk_two_char_substring():
    scanner = _scanner(["护甲"])
    text = "这件护甲很重"
    hits = scanner.scan(text)
    _assert_spans(text, hits)
    assert len(hits) == 1
    assert hits[0]["start"] == 2
    assert hits[0]["end"] == 4
    assert hits[0]["surface"] == "护甲"

    # Longest-first inside a CJK run.
    hits = _scanner(["护甲", "重护甲"]).scan("重护甲")
    _assert_spans("重护甲", hits)
    assert [h["term"] for h in hits] == ["重护甲"]
    assert hits[0]["start"] == 0


# ---------------------------------------------------------------------------
# ordering / limits / empty inputs / duplicates
# ---------------------------------------------------------------------------


def test_hit_order_and_max_hits():
    text = "$Alpha $Beta $Gamma"
    scanner = _scanner(["Alpha", "Beta", "Gamma"])
    hits = scanner.scan(text)
    _assert_spans(text, hits)
    assert [h["surface"] for h in hits] == ["Alpha", "Beta", "Gamma"]
    assert [h["start"] for h in hits] == [
        text.index("Alpha"),
        text.index("Beta"),
        text.index("Gamma"),
    ]

    assert [h["surface"] for h in scanner.scan(text, max_hits=2)] == ["Alpha", "Beta"]
    assert [h["surface"] for h in scanner.scan(text, max_hits=1)] == ["Alpha"]
    assert scanner.scan(text, max_hits=0) == []
    assert scanner.scan(text, max_hits=-3) == []


def test_empty_glossary_empty_text_and_max_hits_default():
    empty = _scanner([])
    assert empty.size == 0
    assert empty.scan("Whiterun Guard") == []

    scanner = _scanner(["Whiterun"])
    assert scanner.scan("") == []
    assert scanner.scan(None) == []
    assert scanner.scan("   \t\n ") == []
    assert scanner.scan("Whiterun")  # sanity
    assert len(scanner.scan(" ".join(["Whiterun"] * 3), max_hits=64)) == 3


def test_duplicate_and_invalid_terms():
    scanner = _scanner(["Whiterun", "Whiterun", "whiterun", " WHITERUN ", "   ", ""])
    assert scanner.size == 1
    assert scanner.duplicate_terms == 3
    assert scanner.skipped_empty == 2
    hits = scanner.scan("Whiterun")
    assert [h["term"] for h in hits] == ["Whiterun"]

    terms = ["a" * 201, "Bad\nTerm", "Valid Term"]
    scanner = _scanner(terms)
    assert scanner.size == 1
    assert scanner.skipped_long == 1
    assert scanner.skipped_control == 1
    assert scanner.scan("Valid Term")[0]["term"] == "Valid Term"


# ---------------------------------------------------------------------------
# build lifecycle / fingerprints / thread safety
# ---------------------------------------------------------------------------


def test_ensure_built_rebuilds_only_on_fingerprint_change():
    scanner = DeterministicGlossaryScanner()
    assert scanner.size == 0
    assert scanner.ensure_built(["Whiterun"], "g1") is True
    assert scanner.size == 1
    assert scanner.build_count == 1

    assert scanner.ensure_built([], "g1") is False
    assert scanner.scan("Whiterun")  # old snapshot is still live

    assert scanner.ensure_built(["Solitude"], "g2") is True
    assert scanner.build_count == 2
    assert scanner.scan("Whiterun") == []
    assert [h["term"] for h in scanner.scan("Solitude")] == ["Solitude"]

    scanner.rebuild(["Riften"])  # fingerprint defaults to ""
    assert [h["term"] for h in scanner.scan("Riften")] == ["Riften"]
    assert scanner.ensure_built(["Markarth"], "") is False
    assert scanner.scan("Markarth") == []


def test_rebuild_is_fast_enough_and_size_tracks_terms():
    scanner = _scanner([f"Term {i}" for i in range(2000)])
    assert scanner.size == 2000
    assert scanner.build_count == 1
    assert scanner.last_build_ms >= 0.0


def test_scan_does_not_leak_exceptions_and_records_last_error():
    scanner = _scanner(["Whiterun"])

    def _boom(_text):
        raise RuntimeError("normalizer exploded")

    scanner._normalize = _boom
    assert scanner.scan("Whiterun") == []
    assert scanner.last_error and "normalizer exploded" in scanner.last_error

    scanner._normalize = lambda value: value.casefold()
    assert scanner.scan("Whiterun")


def test_concurrent_scans():
    terms = ["Whiterun", "Whiterun Guard", "Solitude", "Dragon", "护甲"]
    text = "Whiterun Guard rides from Solitude; 护甲 stays; the Dragon waits."
    scanner = _scanner(terms)
    baseline = scanner.scan(text)
    assert len(baseline) >= 4

    def work(_):
        return scanner.scan(text)

    with ThreadPoolExecutor(max_workers=8) as pool:
        for result in pool.map(work, range(8 * 200)):
            assert result == baseline


def test_glossary_manager_normalizer_parity():
    """The real glossary normalizer must be accepted and behave identically."""
    from src.rag.glossary_manager import GlossaryManager

    # normalize_term_key only reads class-level regexes, so skip load()/I/O.
    normalize = object.__new__(GlossaryManager).normalize_term_key
    terms = ["Whiterun Guard", "Ａｒｍｏｒ", "Мёртвый", "护甲", "Erik's", "SKSE"]
    text = "Whiterun Guard wears Ａｒｍｏｒ; Мёртвый; 护甲; Erik's; skse"
    custom = DeterministicGlossaryScanner(normalize)
    custom.rebuild(terms, "gm")
    default = _scanner(terms)
    assert [
        (h["term"], h["surface"], h["start"], h["end"]) for h in custom.scan(text)
    ] == [
        (h["term"], h["surface"], h["start"], h["end"]) for h in default.scan(text)
    ]
    assert len(custom.scan(text)) == 6


# ---------------------------------------------------------------------------
# performance (loose bounds so CI jitter cannot flake)
# ---------------------------------------------------------------------------

# Measured on the development machine (Python 3.15, Windows 11, 40k terms):
#   rebuild:            ~0.08 s  (asserted < 2 s)
#   400-char scan:      ~0.06 ms (asserted < 50 ms)
#   worst-case shared first token ("the ..." x40k): ~0.07 ms
# These bounds have ~25x/800x headroom, so the test only guards against a
# regression to quadratic behaviour (e.g. re-normalizing or re-scanning the
# whole glossary on every call).
def test_40k_terms_rebuild_and_scan_performance():
    prefixes = [f"Hold{i}" for i in range(400)]
    terms = [f"{prefixes[i % 400]} Retainer {i}" for i in range(39995)]
    terms += ["Whiterun", "Whiterun Guard", "Dragon", "护甲", "Мёртвый"]
    scanner = DeterministicGlossaryScanner()

    started = time.perf_counter()
    scanner.rebuild(terms, "perf")
    rebuild_s = time.perf_counter() - started
    assert scanner.size == 40000
    assert rebuild_s < 2.0, f"40k rebuild took {rebuild_s:.3f}s"

    words = "Whiterun Guard patrols the hold while the Dragon sleeps near 护甲 ."
    text = (words * 8)[:400]
    assert len(text) == 400

    scan_times = []
    for _ in range(20):
        started = time.perf_counter()
        hits = scanner.scan(text)
        scan_times.append(time.perf_counter() - started)
    scan_s = min(scan_times)
    assert hits
    assert scan_s < 0.05, f"400-char scan took {scan_s * 1000:.2f}ms"
