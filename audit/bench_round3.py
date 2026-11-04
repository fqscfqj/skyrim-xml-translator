"""Round-3 acceptance micro-benchmarks.

Usage:  python audit/bench_round3.py

Prints a compact table of the metrics the third-round fixes target.  Each
benchmark is defensive: if a module/API is missing the row is skipped rather
than failing the whole run.
"""

from __future__ import annotations

import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ROWS: list[tuple[str, str, str]] = []


def record(name: str, value: str, note: str = "") -> None:
    ROWS.append((name, value, note))


def timed(fn, repeat: int = 1) -> float:
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def bench_import() -> None:
    code = (
        "import time;t=time.perf_counter();import src.gui_main;"
        "print(round((time.perf_counter()-t)*1000,1))"
    )
    try:
        out = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        )
        record("import src.gui_main", f"{out.stdout.strip()} ms")
    except Exception as exc:  # noqa: BLE001
        record("import src.gui_main", "n/a", str(exc)[:60])


def bench_regex() -> None:
    try:
        from src.translation.text_analyzer import TextAnalyzer
        from src.translation.response_parser import ResponseParser

        analyzer = TextAnalyzer()
        cases = {
            "percent 20k": "%" + "0" * 20000,
            "comment 8k": "<!--" * 8000,
            "bracket 20k": "[a" * 20000,
            "angle 20k": "<" * 20000,
        }
        for name, payload in cases.items():
            try:
                ms = timed(lambda p=payload: analyzer.build_protected_format_shell(p), repeat=3)
                record(f"regex {name}", f"{ms:.1f} ms")
            except Exception as exc:  # noqa: BLE001
                record(f"regex {name}", "error", str(exc)[:60])

        payload = '{"translation":"' + ("x" * 60 + '","id":1},') * 2000
        parser = ResponseParser()
        try:
            ms = timed(lambda: parser.parse(payload, "source", []), repeat=3)
            record("parser 200KB json spam", f"{ms:.1f} ms")
        except Exception as exc:  # noqa: BLE001
            record("parser 200KB json spam", "error", str(exc)[:60])
    except Exception as exc:  # noqa: BLE001
        record("regex suite", "skipped", str(exc)[:80])


def bench_text_pipeline() -> None:
    try:
        from src.translation.text_analyzer import TextAnalyzer
        from src.translation.quality_checker import QualityChecker

        analyzer = TextAnalyzer()
        checker = QualityChecker()
        source = ("The quick brown fox jumps over the lazy dog. " * 120)[:5124]
        target = "敏捷的棕色狐狸跳过了懒狗。" * 240
        target = target[: max(10, len(source) // 2)]

        record(
            "quality check 5124ch",
            f"{timed(lambda: checker.check(source, target, target_lang='zh'), 3):.1f} ms",
        )
        record(
            "protect shell 5124ch",
            f"{timed(lambda: analyzer.build_protected_format_shell(source), 3):.1f} ms",
        )

        cjk = "这是一句用于测试分块边界的中文长文本。" * 230  # ~3700 chars
        chunks = analyzer.chunk_text(cjk, 300)
        clean = sum(1 for c in chunks[:-1] if c.rstrip().endswith(("。", "！", "？", "；")))
        record(
            "CJK 3.7k chunk @300",
            f"{len(chunks)} chunks",
            f"{clean}/{max(1, len(chunks) - 1)} 句末切分",
        )
    except Exception as exc:  # noqa: BLE001
        record("text pipeline", "skipped", str(exc)[:80])


def bench_xml_fidelity() -> None:
    try:
        from src.xml_processor import XMLProcessor

        payload = (
            b'\xef\xbb\xbf<?xml version="1.0" encoding="utf-8" standalone="yes"?>\r\n'
            b"<SSTXMLRessources>\r\n"
            b'  <Params><Addon>Bench</Addon><Source>English</Source>'
            b"<Dest>Chinese</Dest></Params>\r\n"
            b"  <Content>\r\n"
            b'    <String List="0"><EDID>Bench</EDID><Rec>TEST</Rec>'
            b"<Source>Hello</Source><Dest>Hello</Dest></String>\r\n"
            b"  </Content>\r\n"
            b"</SSTXMLRessources>\r\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bench.xml")
            Path(path).write_bytes(payload)
            processor = XMLProcessor()
            assert processor.load_file(path), "load failed"
            processor.save_file(path)
            after = Path(path).read_bytes()
            same = after == payload
            record(
                "XML no-op save bytes",
                "identical" if same else "CHANGED",
                f"{len(payload)}B -> {len(after)}B",
            )
    except Exception as exc:  # noqa: BLE001
        record("XML no-op save bytes", "skipped", str(exc)[:80])


def bench_vector_batch() -> None:
    try:
        import numpy as np
        from src.rag.vector_store import VectorStore

        rng = np.random.default_rng(7)
        dims, count = 64, 20000
        vectors = rng.standard_normal((count, dims)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        with tempfile.TemporaryDirectory() as tmp:
            store = VectorStore(
                os.path.join(tmp, "vectors.npy"),
                os.path.join(tmp, "terms.json"),
                dims,
            )
            store._set_vectors(vectors)
            store.terms = [f"term {i}" for i in range(count)]
            queries = rng.standard_normal((5, dims)).astype(np.float32)
            queries /= np.linalg.norm(queries, axis=1, keepdims=True)

            if hasattr(store, "score_queries_batch"):
                batch = store.score_queries_batch(queries)
                single = np.stack([store.search_cosine_full(q) for q in queries])
                diff = float(np.max(np.abs(batch - single)))
                record(
                    "score_queries_batch 20k",
                    f"{timed(lambda: store.score_queries_batch(queries), 3):.1f} ms",
                    f"max|Δ|={diff:.2e}",
                )
            else:
                record("score_queries_batch 20k", "missing", "契约未实现")
            record(
                "search_cosine_full 20k x5",
                f"{timed(lambda: [store.search_cosine_full(q) for q in queries], 3):.1f} ms",
            )
    except Exception as exc:  # noqa: BLE001
        record("vector batch", "skipped", str(exc)[:80])


def bench_deterministic_scan() -> None:
    try:
        from src.rag.deterministic_scan import DeterministicGlossaryScanner

        scanner = DeterministicGlossaryScanner()
        terms = [f"Glossary Term Number {i}" for i in range(40000)]
        text = (
            "The Whiterun Guard patrols the road while Glossary Term Number 77 "
            "and Glossary Term Number 3999 stand near the gate. 这件护甲很重。"
        ) * 4
        terms.extend(["Whiterun Guard", "护甲"])
        build_ms = timed(lambda: scanner.rebuild(terms, "bench"), repeat=1)
        scanner.ensure_built(terms, "bench")
        hits = scanner.scan(text)
        record(
            "deterministic scan 40k",
            f"{timed(lambda: scanner.scan(text), 5):.3f} ms",
            f"build {build_ms:.0f} ms, {len(hits)} hits",
        )
    except Exception as exc:  # noqa: BLE001
        record("deterministic scan 40k", "skipped", str(exc)[:80])


def main() -> None:
    bench_import()
    bench_regex()
    bench_text_pipeline()
    bench_xml_fidelity()
    bench_vector_batch()
    bench_deterministic_scan()

    width = max(len(r[0]) for r in ROWS) + 2
    print()
    print(f"{'metric'.ljust(width)} {'value':>14}   note")
    print("-" * (width + 40))
    for name, value, note in ROWS:
        print(f"{name.ljust(width)} {value:>14}   {note}")


if __name__ == "__main__":
    main()
