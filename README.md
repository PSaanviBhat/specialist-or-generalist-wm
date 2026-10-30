# Specialist or Generalist World Model — environments and transition data

Team Nemotron (AFML). This repo holds the environment wrappers and the transition data pipeline for ALFWorld and ScienceWorld, used by the RAG (G), SFT (F) and JEPA (D) world-model arms.

- **Setup:** [docs/SETUP_VM.md](docs/SETUP_VM.md)
- **Data format:** [docs/SCHEMA.md](docs/SCHEMA.md)
- **Samples:** `data/sample/alfworld_sample20.jsonl`, `data/sample/sciworld_sample20.jsonl`
- **Full data:** team Google Drive folder (gzipped JSONL + `SHA256SUMS`); not in git.
- **Shared config (seeds, budgets, versions):** [configs/shared.yaml](configs/shared.yaml)

## Layout

```
wm/envs/        ALFWorld and ScienceWorld wrappers: reset(task_id), step(action), admissible_actions(), expert_action(), replay(task_id, prefix)
wm/obs.py       composite observation text (goal, room, location, feedback, inventory, admissible)
wm/collect.py   mixed-policy data collection (expert / random / LLM)
wm/schema.py    JSONL validator: python -m wm.schema FILE.jsonl
splits/         Word2World train/test task ids
tests/          env smoke + replay determinism, schema checks
```

## Collect data

```bash
~/w2w/uv_alfworld/bin/python -m wm.collect --env alfworld --split train --target expert=800,random=480 --seed 0 --out data/raw/alfworld_train.jsonl
```

Rerunning the same command resumes where it stopped.

## Known environment properties

- Observations contain no hidden initial state.
- Replaying a row's `action_prefix` from `reset(task_id)` reproduces `o_t`: exactly in ALFWorld; in ScienceWorld up to the order objects are listed in the room description (compare with `wm.obs.canonical`).
- ScienceWorld ambiguity prompts end the episode (their numbered options are not replayable).
- ALFWorld's handcoded expert is not optimal: it explores and solves roughly half of tasks within 30 steps.
