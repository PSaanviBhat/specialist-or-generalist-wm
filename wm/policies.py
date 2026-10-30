"""Data-collection policies. Expert and random need no LLM."""
from __future__ import annotations

import random
from typing import Optional, Tuple

# ALFWorld's "help" only prints the command list; never chosen randomly and never recorded.
SKIP_ACTIONS = {"help"}


class ExpertPolicy:
    name = "expert"

    def act(self, env, rng: random.Random) -> Tuple[Optional[str], Optional[dict]]:
        return env.expert_action(), None


class RandomPolicy:
    name = "random"

    def act(self, env, rng: random.Random) -> Tuple[Optional[str], Optional[dict]]:
        choices = [a for a in env.admissible_actions() if a not in SKIP_ACTIONS]
        return (rng.choice(choices) if choices else None), None


def make_policy(name: str):
    if name == "expert":
        return ExpertPolicy()
    if name == "random":
        return RandomPolicy()
    if name == "llm":
        from wm.llm_policy import LLMPolicy
        return LLMPolicy()
    raise ValueError("unknown policy: %s" % name)
