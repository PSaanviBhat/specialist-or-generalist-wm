"""Export collected transitions to the JEPA/SFT buffer format (jepa/training/buffer.py).

One buffer per budget file, plus the task-disjoint held-out set:

  python -m wm.export_buffers --env alfworld \
      --train data/raw/alfworld_train.jsonl.gz --heldout data/raw/alfworld_heldout.jsonl.gz [--provisional]

writes data/buffers/{env}_seed{s}_n{n}[_provisional].jsonl and data/buffers/{env}_heldout.jsonl.

Field mapping (buffer <- schema):
  obs <- o_t, action <- a_t, obs_next <- o_next, env <- env,
  game <- task family (ALFWorld: the 6 task types, e.g. "pick_heat_then_place_in_recep";
          ScienceWorld: the task name), the per-game unit used by scripts/probe_fidelity.py,
  cap_tag <- first capability tag, or "" while untagged (cap_tags keeps the full list).
Extra fields kept for traceability: transition_id, task_id, split, policy, feedback_next.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import yaml

from wm.schema import read_jsonl

CONFIG = "configs/shared.yaml"


def game_of(row: Dict) -> str:
    if row["env"] == "alfworld":
        return row["task_type"].split("-")[0]
    return row["task_type"]


def to_buffer_row(row: Dict) -> Dict:
    tags = row.get("capability_tag") or []
    return {
        "obs": row["o_t"],
        "action": row["a_t"],
        "obs_next": row["o_next"],
        "env": row["env"],
        "game": game_of(row),
        "cap_tag": tags[0] if tags else "",
        "cap_tags": list(tags),
        "transition_id": row["transition_id"],
        "task_id": row["task_id"],
        "split": row["split"],
        "policy": row["policy"],
        "feedback_next": row["feedback_next"],
    }


def write_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--heldout", required=True)
    p.add_argument("--budget-dir", default="data/budgets")
    p.add_argument("--out-dir", default="data/buffers")
    p.add_argument("--provisional", action="store_true")
    args = p.parse_args(argv)

    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    suffix = "_provisional" if args.provisional else ""
    by_id = {r["transition_id"]: r for r in read_jsonl(args.train)}
    os.makedirs(args.out_dir, exist_ok=True)

    for seed in cfg["seeds"]:
        for n in cfg["budgets"]:
            name = "%s_seed%d_n%d%s" % (args.env, seed, n, suffix)
            with open(os.path.join(args.budget_dir, name + ".txt")) as f:
                ids = [line.strip() for line in f if line.strip()]
            missing = [i for i in ids if i not in by_id]
            if missing:
                raise SystemExit("%s: %d ids not in %s (e.g. %s)" % (name, len(missing), args.train, missing[0]))
            write_jsonl(os.path.join(args.out_dir, name + ".jsonl"), [to_buffer_row(by_id[i]) for i in ids])

    held = [to_buffer_row(r) for r in read_jsonl(args.heldout) if r["split"] == "heldout"]
    write_jsonl(os.path.join(args.out_dir, "%s_heldout.jsonl" % args.env), held)
    print("wrote %d budget buffers + %s_heldout.jsonl (%d rows) to %s"
          % (len(cfg["seeds"]) * len(cfg["budgets"]), args.env, len(held), args.out_dir))


if __name__ == "__main__":
    main()
