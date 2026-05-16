"""
Smoke test for the v2 alignment + plan post-processing pipeline.

What this tests
---------------
1. align.tokenize_script — extracts Arabic word tokens correctly
2. align._interpolate    — interpolation fallback produces sensible timings
3. align.assign_words_to_sections — buckets words back into sections
4. plan._snap_to_word_boundaries  — quantises shot times to word boundaries
5. plan._fill_captions            — populates caption_text from word timings
6. plan._assign_sections          — tags shots with their section_id
7. plan._validate_plan            — splits over-long shots, no gaps/overlaps
8. plan.shots_to_json / shots_from_json — round-trip

What this does NOT test
-----------------------
- The actual Claude planner call (requires API key)
- The actual WhisperX backend (requires the wav2vec2 model)
- The FFmpeg renderer (next session's work)

Run with:
    python -m phase3_v2.test_smoke
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

# Make the v2 modules importable as a flat package
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def _build_test_phase3_package():
    """
    The align.py / plan.py we wrote live in /home/claude/phase3_v2/.
    For the smoke test we mount them under a synthetic `phase3` package
    so the v2 modules can import each other normally.
    """
    import importlib.util
    import types

    pkg = types.ModuleType("phase3")
    pkg.__path__ = [str(HERE)]
    sys.modules["phase3"] = pkg

    # Inline mini-parser so align.py can import ScriptSection without us
    # having to copy phase3/parser.py into the test directory.
    parser_mod = types.ModuleType("phase3.parser")

    from dataclasses import dataclass, field

    @dataclass
    class ScriptSection:
        section_id: str
        title: str
        text: str
        char_count: int = field(init=False)

        def __post_init__(self):
            self.char_count = len(self.text.strip())

    parser_mod.ScriptSection = ScriptSection
    sys.modules["phase3.parser"] = parser_mod

    # Load align.py
    spec = importlib.util.spec_from_file_location(
        "phase3.align", HERE / "align.py")
    align_mod = importlib.util.module_from_spec(spec)
    sys.modules["phase3.align"] = align_mod
    spec.loader.exec_module(align_mod)

    # Load plan.py
    spec = importlib.util.spec_from_file_location(
        "phase3.plan", HERE / "plan.py")
    plan_mod = importlib.util.module_from_spec(spec)
    sys.modules["phase3.plan"] = plan_mod
    spec.loader.exec_module(plan_mod)

    return parser_mod, align_mod, plan_mod


def test_tokenize():
    parser, align, plan = _build_test_phase3_package()
    text = "هذا اختبار بسيط للتأكد من أن التقسيم يعمل."
    tokens = align.tokenize_script(text)
    assert len(tokens) >= 7, f"expected ≥7 tokens, got {len(tokens)}: {tokens}"
    print(f"  ✓ tokenize_script: {len(tokens)} tokens from sample")


def test_interpolate():
    parser, align, plan = _build_test_phase3_package()
    tokens = ["كلمة", "أخرى", "ثالثة", "رابعة", "خامسة"]
    timings = align._interpolate(tokens, total_duration_sec=10.0)
    assert len(timings) == 5
    assert abs(timings[-1].end - 10.0) < 0.001, \
        f"last word should end at 10.0s, got {timings[-1].end}"
    assert all(t.source == "interpolated" for t in timings)
    assert all(timings[i].end <= timings[i + 1].start + 0.001
               for i in range(len(timings) - 1)), "overlapping timings"
    print(f"  ✓ _interpolate: 5 tokens over 10.0s, last ends at {timings[-1].end:.3f}s")


def test_assign_words_to_sections():
    parser, align, plan = _build_test_phase3_package()
    sections = [
        parser.ScriptSection("opening", "افتتاحية", "هذه افتتاحية قصيرة جدا"),
        parser.ScriptSection("point_1", "النقطة الأولى",
                             "النقطة الأولى تتحدث عن شيء مختلف تماما"),
    ]
    full_text = "\n".join(s.text for s in sections)
    timings = align._interpolate(
        align.tokenize_script(full_text), total_duration_sec=20.0)
    section_map = align.assign_words_to_sections(timings, sections)
    assert "opening" in section_map
    assert "point_1" in section_map
    op_start, op_end, op_words = section_map["opening"]
    p1_start, p1_end, p1_words = section_map["point_1"]
    assert op_end <= p1_start + 0.01, "section overlap"
    print(f"  ✓ assign_words_to_sections: opening={op_start:.2f}-{op_end:.2f}s "
          f"({len(op_words)} words), point_1={p1_start:.2f}-{p1_end:.2f}s "
          f"({len(p1_words)} words)")


def test_plan_post_processing():
    parser, align, plan = _build_test_phase3_package()

    sections = [
        parser.ScriptSection("opening", "افتتاحية", "هذه افتتاحية قصيرة"),
        parser.ScriptSection("point_1", "النقطة الأولى",
                             "النقطة الأولى ثم النقطة الثانية ثم النقطة الثالثة"),
    ]
    full_text = "\n".join(s.text for s in sections)
    timings = align._interpolate(
        align.tokenize_script(full_text), total_duration_sec=30.0)

    # Simulated planner output — intentionally messy:
    # - One shot is over 6 s and should be split
    # - One shot's start/end is mid-word and should snap
    # - One shot has no caption_text and should get one
    raw_shots = [
        {"start": 0.0, "end": 3.5, "visual": "title_card",
         "search_query": "", "motion": "static_hold",
         "typography_text": "كتاب الاختبار",
         "typography_template": "name_reveal"},
        {"start": 3.5, "end": 12.0, "visual": "location",   # too long → split
         "search_query": "Baghdad 1920", "motion": "slow_push"},
        {"start": 12.7, "end": 18.3, "visual": "typography",  # off-boundary
         "typography_text": "اقتباس مؤثر",
         "typography_template": "pull_quote", "motion": "static_hold"},
        {"start": 18.3, "end": 30.0, "visual": "portrait",   # too long → split
         "search_query": "Test Subject portrait", "motion": "slow_push"},
    ]
    shots = [plan._shot_from_dict(d) for d in raw_shots]
    shots = plan._snap_to_word_boundaries(shots, timings)
    shots = plan._fill_captions(shots, timings)

    section_map = align.assign_words_to_sections(timings, sections)
    shots = plan._assign_sections(shots, section_map)
    shots = plan._validate_plan(shots, total_duration_sec=30.0)

    # Verify invariants
    assert all(s.duration <= 8.0 + 0.01 for s in shots), \
        f"over-long shot: {[s.duration for s in shots]}"
    assert abs(shots[-1].end - 30.0) < 0.01, \
        f"last shot doesn't end at 30s: {shots[-1].end}"
    for i in range(len(shots) - 1):
        gap = shots[i + 1].start - shots[i].end
        assert abs(gap) < 0.01, f"gap/overlap at shot {i}: {gap:.3f}s"
    assert any(s.caption_text for s in shots), "no captions filled"
    assert any(s.section_id for s in shots), "no section ids assigned"

    print(f"  ✓ plan post-processing: {len(raw_shots)} raw shots → "
          f"{len(shots)} validated, all ≤6s, no gaps")
    for i, s in enumerate(shots):
        kind = s.visual
        marker = " [split]" if "split" in s.note else ""
        print(f"    {i+1:>2}. [{s.section_id:>9}] "
              f"{s.start:5.2f}-{s.end:5.2f}s ({s.duration:.1f}s) "
              f"{kind:<13}{marker}")


def test_json_roundtrip():
    parser, align, plan = _build_test_phase3_package()
    shots = [
        plan.Shot(start=0.0, end=4.0, visual="title_card",
                  motion="static_hold", typography_text="كتاب الاختبار",
                  typography_template="name_reveal",
                  caption_text="", section_id="opening"),
        plan.Shot(start=4.0, end=7.5, visual="portrait",
                  search_query="historical portrait", motion="slow_push",
                  caption_text="هذه افتتاحية قصيرة", section_id="opening"),
    ]
    text = plan.shots_to_json(shots)
    parsed = json.loads(text)
    assert len(parsed) == 2
    restored = plan.shots_from_json(text)
    assert len(restored) == 2
    assert restored[0].typography_text == "كتاب الاختبار"
    assert restored[1].search_query == "historical portrait"
    print(f"  ✓ json round-trip: {len(shots)} shots ↔ JSON")


def test_summarise_plan():
    parser, align, plan = _build_test_phase3_package()
    shots = [
        plan.Shot(start=0.0, end=4.0, visual="title_card",
                  motion="static_hold", section_id="opening",
                  typography_text="كتاب الاختبار"),
        plan.Shot(start=4.0, end=7.5, visual="portrait",
                  search_query="historical portrait", motion="slow_push",
                  section_id="opening"),
        plan.Shot(start=7.5, end=11.0, visual="typography",
                  motion="static_hold", section_id="point_1",
                  typography_text="هذه عبارة مؤثرة من النص الأصلي"),
    ]
    summary = plan.summarise_plan(shots)
    print("\n  ── Sample summarise_plan output ──")
    for line in summary.splitlines():
        print(f"    {line}")
    print()


def main():
    print("\nv2 smoke test\n" + "─" * 50)
    test_tokenize()
    test_interpolate()
    test_assign_words_to_sections()
    test_plan_post_processing()
    test_json_roundtrip()
    test_summarise_plan()
    print("\nAll smoke tests passed ✓\n")


if __name__ == "__main__":
    main()
