"""
Belief state and Expected Information Gain (Section 3.3).
- BeliefState: probability distribution b_t over candidates
  - init via softmax(sim) (Eq.1), entropy H(b_t) (Eq.2)
  - Bayesian update b_{t+1} ∝ b_t · sim^δ (Eq.10)
- compute_eig: EIG for ask actions over attribute options (Eq.4)
- compute_eig_commit: EIG for commit actions over accept/reject
- compute_eig_all: EIG for all un-asked attributes
"""

import math
import numpy as np
from dataclasses import dataclass, field
from copy import deepcopy
from typing import Dict, Any


@dataclass
class BeliefState:
    """Probability distribution b_t over candidate set C_t."""
    probs: Dict[str, float] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)

    def initialize_uniform(self, candidates):
        """Uniform prior when no conversation context is available."""
        n = len(candidates)
        if n > 0:
            p = 1.0 / n
            self.probs = {cid: p for cid in candidates}

    def initialize_from_similarity(self, candidates, scores):
        """Eq.1: b_0(c_i) = softmax(sim(c_i, h_0)). Scores are pre-softmaxed."""
        self.probs = {cid: float(s) for cid, s in zip(candidates, scores)}
        self._normalize()

    def _normalize(self):
        total = sum(self.probs.values())
        if total <= 0:
            n = len(self.probs)
            if n > 0:
                self.probs = {k: 1.0 / n for k in self.probs}
        else:
            self.probs = {k: v / total for k, v in self.probs.items()}

    def entropy(self, eps=1e-12):
        """Eq.2: H(b_t) = -Σ b_t(c_i) log b_t(c_i)"""
        h = 0.0
        for p in self.probs.values():
            if p > eps:
                h -= p * math.log(p)
        return h

    def max_entropy(self):
        """Maximum entropy for current candidate count: log|C_t|."""
        n = len(self.probs)
        return math.log(n) if n > 1 else 0.0

    def normalized_entropy(self):
        """H(b_t) / log|C_t|, used in commitment trigger (compared against ε)."""
        me = self.max_entropy()
        return self.entropy() / me if me > 0 else 0.0

    def update_bayesian(self, similarity_scores, remaining_candidates, alpha=1.0):
        """Eq.10: b_{t+1}(c_i) ∝ b_t(c_i) · sim(c_i, h_{t+1})^δ
        Candidates not in remaining set are removed."""
        eps = 1e-12
        remaining = set(remaining_candidates)
        new_probs = {}
        for item_id, prob in self.probs.items():
            if item_id in remaining:
                sim = similarity_scores.get(item_id, eps)
                new_probs[item_id] = prob * max(sim, eps) ** alpha
        self.probs = new_probs
        self._normalize()

    def filter_candidates(self, remaining):
        """Hard elimination: remove candidates and renormalize."""
        remaining_set = set(remaining)
        self.probs = {k: v for k, v in self.probs.items() if k in remaining_set}
        self._normalize()

    def add_constraint(self, attr_name, value):
        """Record user's attribute choice (prevents re-asking)."""
        self.constraints[attr_name] = value

    def get_asked_attributes(self):
        return set(self.constraints.keys())

    def top_k(self, k):
        """Top-k candidates sorted by belief probability (descending)."""
        return sorted(self.probs.items(), key=lambda x: x[1], reverse=True)[:k]

    def get_prob(self, item_id):
        return self.probs.get(item_id, 0.0)

    def copy(self):
        new = BeliefState()
        new.probs = self.probs.copy()
        new.constraints = deepcopy(self.constraints)
        return new


def compute_eig(belief, candidates, data_loader, attr_name, options):
    """Eq.4: EIG(a, b_t) = H(b_t) - Σ P(o|a) · H(b_t|o) for ask actions.

    Each candidate is assigned to exactly one option (first match),
    forming a partition. This mirrors _find_matching_option in the
    simulator, ensuring EIG predictions match actual user behavior."""
    current_entropy = belief.entropy()
    NO_PREF = "No preference"

    # assign each candidate to exactly one option
    option_members = {o: [] for o in options}
    specific_options = [o for o in options if o != "Others" and o != NO_PREF]

    for cid in candidates:
        item = data_loader.get_item(cid)
        if not item:
            continue
        val = item.get_attribute(attr_name)

        if val is None:
            if NO_PREF in option_members:
                option_members[NO_PREF].append(cid)
            elif "Others" in option_members:
                option_members["Others"].append(cid)
            continue

        # first-match for multi-valued attrs (e.g., genre=["Action","Comedy"])
        matched = False
        if isinstance(val, list):
            for opt in specific_options:
                if opt in val:
                    option_members[opt].append(cid)
                    matched = True
                    break
        else:
            if val in specific_options:
                option_members[val].append(cid)
                matched = True

        if not matched and "Others" in option_members:
            option_members["Others"].append(cid)

    # EIG = H(before) - Σ P(option) · H(after|option)
    expected_after = 0.0
    for option in options:
        members = option_members[option]
        prob = sum(belief.get_prob(cid) for cid in members)
        if prob > 0 and members:
            h = 0.0
            for cid in members:
                p = belief.get_prob(cid) / prob
                if p > 1e-12:
                    h -= p * math.log(p)
            expected_after += prob * h

    return current_entropy - expected_after


def compute_eig_all(belief, candidates, data_loader, asked_attributes, max_options=4):
    """Compute EIG for all un-asked attributes. Returns {attr: {eig, options}}."""
    askable = data_loader.get_askable_attributes()
    results = {}

    for attr_name in askable:
        if attr_name in asked_attributes:
            continue
        dist = data_loader.get_attribute_distribution(candidates, attr_name)
        if not dist:
            continue

        # build options: top values by frequency, fill to max_options
        sorted_vals = sorted(dist.items(), key=lambda x: -x[1])
        if len(sorted_vals) <= max_options:
            options = [v for v, _ in sorted_vals]
        else:
            n_specific = max_options - 1
            options = [v for v, _ in sorted_vals[:n_specific]]
            if "Others" not in options:
                options.append("Others")
            else:
                # "Others" already in top, take one more specific value
                for v, _ in sorted_vals[n_specific:]:
                    if v != "Others":
                        options.append(v)
                        break

        if len(options) < 2:
            continue

        eig = compute_eig(belief, candidates, data_loader, attr_name, options)
        results[attr_name] = {"eig": eig, "options": options}

    return results


def compute_eig_commit(belief, commit_item_id):
    """EIG for commit actions. Observations: {accept, reject}.
    P(accept) = b_t(c*), H(accept) = 0 (conversation ends).
    P(reject) = 1 - b_t(c*), H(reject) = entropy after removing c*."""
    current_entropy = belief.entropy()
    p_accept = belief.get_prob(commit_item_id)
    p_reject = 1.0 - p_accept

    # H(b_t | reject) = entropy after removing committed item
    if p_reject > 1e-12:
        h_reject = 0.0
        for cid, prob in belief.probs.items():
            if cid == commit_item_id:
                continue
            p = prob / p_reject
            if p > 1e-12:
                h_reject -= p * math.log(p)
    else:
        h_reject = 0.0

    return current_entropy - p_reject * h_reject
