# Specialist or Generalist World Model — environments and transition data

Team Nemotron (AFML). This repo holds the environment wrappers and the transition data pipeline for ALFWorld and ScienceWorld, used by the RAG (G), SFT (F) and JEPA (D) world-model arms.

- **Setup:** [docs/SETUP_VM.md](docs/SETUP_VM.md)
- **Data format:** [docs/SCHEMA.md](docs/SCHEMA.md)
- **Samples:** `data/sample/alfworld_sample20.jsonl`, `data/sample/sciworld_sample20.jsonl`
- **Full data:** team Google Drive folder (gzipped JSONL + `SHA256SUMS`); not in git.
- **Shared config (seeds, budgets, versions):** [configs/shared.yaml](configs/shared.yaml)
- **Budgets:** `data/budgets/{env}_seed{s}_n{n}.txt` (transition ids; nested 32 ⊂ … ⊂ 1024 per seed, 50/30/20 expert/random/LLM). `*_provisional` files are expert/random only (5:3) until the LLM rows exist.
- **When the GPU is available:** [docs/GPU_DAY.md](docs/GPU_DAY.md)
- **Task 3 sanity check rule:** [docs/SANITY_CHECK.md](docs/SANITY_CHECK.md)

## Layout

```
wm/envs/        ALFWorld and ScienceWorld wrappers: reset(task_id), step(action), admissible_actions(), expert_action(), replay(task_id, prefix)
wm/obs.py       composite observation text (goal, room, location, feedback, inventory, admissible)
wm/collect.py   mixed-policy data collection (expert / random / LLM)
wm/policies.py  expert and random policies (random is verb-first in ScienceWorld)
wm/llm_client.py, wm/llm_policy.py   LLM_BASE_URL client with sqlite cache + retries; LLM policy
wm/budgets.py   nested, stratified budget id lists
wm/stats.py     counts + integrity checks (split leakage, duplicates, schema)
wm/replay_check.py   replays saved action_prefix for sampled rows and compares o_t / next feedback
wm/schema.py    JSONL validator: python -m wm.schema FILE.jsonl
sanity/         Task 3: 3-shot retrieval sanity check
scripts/        setup_vm.sh, fetch_data.py, collect_all.sh, package_data.sh
splits/         Word2World train/test task ids
tests/          env smoke + replay determinism, schema checks
```

## Collect data

```bash
nohup bash scripts/collect_all.sh > logs/collect_all.log 2>&1 &   # expert + random, both envs
bash scripts/package_data.sh                                        # checks, provisional budgets, gzip + SHA256SUMS
```

Rerunning resumes where it stopped. Offline tests (no env, no GPU): `python -m pytest tests/test_schema.py tests/test_budgets.py tests/test_llm.py tests/test_sanity.py -q`.

## Known environment properties

- Observations contain no hidden initial state.
- Replaying a row's `action_prefix` from `reset(task_id)` reproduces `o_t`: exactly in ALFWorld; in ScienceWorld up to the order objects are listed in the room description (compare with `wm.obs.canonical`).
- ScienceWorld ambiguity prompts end the episode (their numbered options are not replayable).
- ALFWorld: every admissible command succeeds, so about 1/3 of random steps use a well-formed non-admissible command over visible names (e.g. `take mug 1 from drawer 2` when the mug is elsewhere), which returns `Nothing happens.`. This is where ALFWorld's action-failure transitions come from.
- ScienceWorld random actions pick a verb uniformly, then an action with that verb (about 2/3 of valid commands are `connect X to Y`). The LLM sees at most 60 ScienceWorld candidates, sampled the same way.
- ALFWorld's handcoded expert is not optimal: it explores and solves roughly half of tasks within 30 steps.
