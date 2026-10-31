"""Buffer export format, no environment needed: python -m pytest tests/test_export.py -q"""
import json

from wm.export_buffers import game_of, main, to_buffer_row
from wm.schema import read_jsonl

BUFFER_FIELDS = {"obs", "action", "obs_next", "env", "game", "cap_tag"}


def test_row_mapping():
    row = next(read_jsonl("data/sample/alfworld_sample20.jsonl"))
    b = to_buffer_row(row)
    assert BUFFER_FIELDS <= set(b)
    assert (b["obs"], b["action"], b["obs_next"]) == (row["o_t"], row["a_t"], row["o_next"])
    assert b["cap_tag"] == "" and b["game"] in {
        "pick_and_place_simple", "look_at_obj_in_light", "pick_clean_then_place_in_recep",
        "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_two_obj_and_place"}
    sci = next(read_jsonl("data/sample/sciworld_sample20.jsonl"))
    assert game_of(sci) == sci["task_type"]


def test_export_files(tmp_path):
    rows = list(read_jsonl("data/sample/alfworld_sample20.jsonl"))
    train = tmp_path / "train.jsonl"
    held = tmp_path / "held.jsonl"
    train.write_text("".join(json.dumps(r) + "\n" for r in rows))
    held.write_text("".join(json.dumps(dict(r, split="heldout")) + "\n" for r in rows[:5]))
    bdir = tmp_path / "budgets"
    bdir.mkdir()
    for s in (0, 1, 2):
        for n in (32, 64, 128, 256, 512, 1024):  # tiny stand-in budgets over the 20 sample ids
            (bdir / ("alfworld_seed%d_n%d.txt" % (s, n))).write_text(
                "\n".join(r["transition_id"] for r in rows[: min(n, 20) - s]) + "\n")
    out = tmp_path / "buffers"
    main(["--env", "alfworld", "--train", str(train), "--heldout", str(held),
          "--budget-dir", str(bdir), "--out-dir", str(out)])
    first = [json.loads(l) for l in open(out / "alfworld_seed0_n32.jsonl")]
    assert len(first) == 20 and all(BUFFER_FIELDS <= set(r) for r in first)
    assert len(open(out / "alfworld_heldout.jsonl").readlines()) == 5
