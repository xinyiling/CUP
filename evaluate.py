"""
Evaluation metrics and episode recording.
Computes: Success Rate (SR), Average Turns (avgT), success@k, per-turn stats.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from collections import defaultdict


@dataclass
class TurnRecord:
    turn_id: int
    action_type: str
    attribute_name: Optional[str] = None
    options: Optional[List[str]] = None
    user_response: Optional[str] = None
    recommended_items: Optional[List[str]] = None
    accepted: Optional[bool] = None
    agent_utterance: Optional[str] = None
    user_natural_language: Optional[str] = None
    entropy_after: float = 0.0
    info_gain: float = 0.0
    num_candidates_after: int = 0
    belief_after: Optional[Dict[str, float]] = None  # b_{t+1} top candidates
    reward: float = 0.0

    def to_dict(self):
        return asdict(self)


@dataclass
class EpisodeRecord:
    episode_id: int
    target_item_id: str
    target_item_name: str
    method: str
    success: bool = False
    num_turns: int = 0
    total_reward: float = 0.0
    total_info_gain: float = 0.0
    initial_candidates: int = 0
    initial_entropy: float = 0.0
    initial_belief: Optional[Dict[str, float]] = None  # b_0 top candidates
    turns: List[TurnRecord] = field(default_factory=list)

    def add_turn(self, turn):
        self.turns.append(turn)
        self.num_turns = len(self.turns)
        self.total_reward += turn.reward
        self.total_info_gain += turn.info_gain

    def to_dict(self):
        return {
            "episode_id": self.episode_id,
            "target_item_id": self.target_item_id,
            "target_item_name": self.target_item_name,
            "method": self.method,
            "success": self.success,
            "num_turns": self.num_turns,
            "total_reward": self.total_reward,
            "total_info_gain": self.total_info_gain,
            "initial_candidates": self.initial_candidates,
            "initial_entropy": self.initial_entropy,
            "initial_belief": self.initial_belief,
            "turns": [t.to_dict() for t in self.turns]
        }

    @classmethod
    def from_dict(cls, data):
        turns = [TurnRecord(**t) for t in data.pop("turns", [])]
        record = cls(**data)
        record.turns = turns
        return record


def compute_metrics(records, max_turns=5):
    if not records:
        return {"num_episodes": 0, "success_rate": 0.0, "avg_turns": 0.0, "avg_reward": 0.0}

    n = len(records)
    successes = [r for r in records if r.success]
    sr = len(successes) / n
    avg_turns = sum(r.num_turns for r in records) / n
    avg_reward = sum(r.total_reward for r in records) / n
    avg_ig = sum(r.total_info_gain for r in records) / n

    success_at_k = {
        f"success@{max_turns}": sum(1 for r in records if r.success and r.num_turns <= max_turns) / n
    }

    turn_data = defaultdict(lambda: {"count": 0, "ask": 0, "rec": 0, "ig": 0.0})
    for record in records:
        for turn in record.turns:
            t = turn_data[turn.turn_id]
            t["count"] += 1
            if turn.action_type == "ask":
                t["ask"] += 1
                t["ig"] += turn.info_gain
            else:
                t["rec"] += 1

    turn_stats = {}
    for tid, s in sorted(turn_data.items()):
        turn_stats[f"turn_{tid+1}"] = {
            "count": s["count"], "ask_count": s["ask"], "recommend_count": s["rec"],
            "avg_info_gain": s["ig"] / s["ask"] if s["ask"] > 0 else 0
        }

    return {
        "success_rate": sr, "avg_turns": avg_turns,
        "num_conversations": n, "num_successes": len(successes),
        **success_at_k,
        "turn_stats": turn_stats,
    }
