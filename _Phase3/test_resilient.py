"""
Test the resilient JSON parser in plan.py against:
  1. A clean, valid response  → should parse all shots
  2. A response truncated mid-shot with a trailing comma  → should
     salvage all complete shots and drop the last malformed one
  3. A response cut off mid-string  → should salvage everything before it
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

# Mount the v2 modules under a synthetic phase3 package, same as test_smoke
import importlib.util
import types

pkg = types.ModuleType("phase3")
pkg.__path__ = [str(HERE)]
sys.modules["phase3"] = pkg

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

spec = importlib.util.spec_from_file_location("phase3.align", HERE / "align.py")
align_mod = importlib.util.module_from_spec(spec)
sys.modules["phase3.align"] = align_mod
spec.loader.exec_module(align_mod)

spec = importlib.util.spec_from_file_location("phase3.plan", HERE / "plan.py")
plan_mod = importlib.util.module_from_spec(spec)
sys.modules["phase3.plan"] = plan_mod
spec.loader.exec_module(plan_mod)


CLEAN = """\
{
  "shots": [
    {"start": 0.0, "end": 4.0, "visual": "title_card", "search_query": "",
     "motion": "static_hold", "typography_text": "كتاب الاختبار",
     "typography_template": "name_reveal", "note": ""},
    {"start": 4.0, "end": 7.5, "visual": "portrait",
     "search_query": "historical portrait", "motion": "slow_push",
     "typography_text": "", "typography_template": null, "note": "anchor"},
    {"start": 7.5, "end": 11.2, "visual": "archive",
     "search_query": "Baghdad 1920", "motion": "pan_right",
     "typography_text": "", "typography_template": null, "note": ""}
  ]
}
"""

# This is what you actually hit: 2 complete shots, then a third object
# that's missing the closing brace, followed by a comma. The original
# parser fails on the whole thing; the resilient one should return 2.
TRUNCATED_MID_OBJECT = """\
{
  "shots": [
    {"start": 0.0, "end": 4.0, "visual": "title_card", "search_query": "",
     "motion": "static_hold", "typography_text": "كتاب",
     "typography_template": "name_reveal", "note": ""},
    {"start": 4.0, "end": 7.5, "visual": "portrait",
     "search_query": "historical portrait", "motion": "slow_push",
     "typography_text": "", "typography_template": null, "note": "anchor"},
    {"start": 7.5, "end": 11.2, "visual": "archive",
     "search_query": "Baghdad 1920", "motion": "pan_right",
     "typography_text": "", "typography_template":
"""

# Cut mid-string of the typography_text — depth tracker won't close,
# salvage should return everything before the unclosed object.
TRUNCATED_MID_STRING = """\
{
  "shots": [
    {"start": 0.0, "end": 4.0, "visual": "title_card", "search_query": "",
     "motion": "static_hold", "typography_text": "كتاب الاختبار",
     "typography_template": "name_reveal", "note": ""},
    {"start": 4.0, "end": 7.5, "visual": "typography",
     "search_query": "", "motion": "static_hold",
     "typography_text": "هذه جملة لم تكتمل بسبب
"""


def test_clean():
    data = plan_mod._extract_json_resilient(CLEAN)
    assert len(data["shots"]) == 3, f"expected 3 shots, got {len(data['shots'])}"
    assert data["shots"][0]["visual"] == "title_card"
    print(f"  ✓ clean response: 3/3 shots parsed")


def test_truncated_mid_object():
    data = plan_mod._extract_json_resilient(TRUNCATED_MID_OBJECT)
    n = len(data["shots"])
    assert n == 2, f"expected 2 salvaged shots, got {n}"
    assert data["shots"][-1]["visual"] == "portrait"
    print(f"  ✓ truncated mid-object: salvaged {n}/3 shots "
          f"(last complete = {data['shots'][-1]['visual']!r})")


def test_truncated_mid_string():
    data = plan_mod._extract_json_resilient(TRUNCATED_MID_STRING)
    n = len(data["shots"])
    assert n == 1, f"expected 1 salvaged shot, got {n}"
    print(f"  ✓ truncated mid-string: salvaged {n}/2 shots")


def test_target_sizing():
    # 337s script (Abdol's actual run): the old code computed 96 shots,
    # which blew the output budget. The new sizing should cap at 70.
    n = plan_mod._sized_target_shots(337.1, target_shot_duration=4.5)
    assert n <= 70, f"target should cap at 70, got {n}"
    assert n >= 40, f"target should still be substantial, got {n}"
    print(f"  ✓ 337s @ 4.5s avg → {n} shots (was 96 in v1; cap is 70)")

    n_short = plan_mod._sized_target_shots(90.0, target_shot_duration=3.5)
    assert n_short == 25, f"90s/3.5s = 25 expected, got {n_short}"
    print(f"  ✓ 90s @ 3.5s avg → {n_short} shots (short videos: linear)")

    n_med = plan_mod._sized_target_shots(180.0, target_shot_duration=4.5)
    assert n_med == 40, f"180s/4.5s = 40 expected, got {n_med}"
    print(f"  ✓ 180s @ 4.5s avg → {n_med} shots (boundary)")


def main():
    print("\nResilient parser tests\n" + "─" * 50)
    test_clean()
    test_truncated_mid_object()
    test_truncated_mid_string()
    test_target_sizing()
    print("\nAll resilient-parser tests passed ✓\n")


if __name__ == "__main__":
    main()
