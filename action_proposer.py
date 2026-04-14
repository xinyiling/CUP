"""
LLM-based action proposal: A_t = LLM(h_t, C_t) (Section 3.3).
LLM proposes options for EVERY available attribute, grounded against
real candidate distributions. EIG+MCTS handles attribute selection.
"""

import re
from environment import AskAction, get_ask_actions, get_recommend_action
from prompts import get_domain, ACTION_PROPOSAL_SYSTEM, ACTION_PROPOSAL_TEMPLATE


def _build_candidate_list(data_loader, candidates, belief):
    """Build candidate list C_t with names and belief probabilities."""
    lines = []
    sorted_cands = sorted(candidates, key=lambda c: belief.get_prob(c), reverse=True)
    for cid in sorted_cands:
        item = data_loader.get_item(cid)
        if item:
            prob = belief.get_prob(cid)
            lines.append(f"  - {item.name} (belief: {prob:.4f})")
    return "\n".join(lines) if lines else "  (none)"


def _build_attr_summary(data_loader, candidates, asked):
    lines = []
    for attr_name in data_loader.get_askable_attributes():
        if attr_name in asked:
            continue
        dist = data_loader.get_attribute_distribution(candidates, attr_name)
        if not dist:
            continue
        sorted_vals = sorted(dist.items(), key=lambda x: -x[1])
        top = sorted_vals[:6]
        rest = sum(c for _, c in sorted_vals[6:])
        parts = [f"{v}({c})" for v, c in top]
        if rest > 0:
            parts.append(f"others({rest})")
        lines.append(f"  {attr_name}: {', '.join(parts)}")
    return "\n".join(lines) if lines else "  (none available)"


def _build_history_str(state):
    lines = []
    for turn in state.history:
        if turn.agent_action_type == "ask":
            lines.append(f"  Agent asked about {turn.agent_action.get('attribute_name', '?')}: "
                         f"{turn.agent_action.get('question_text', '')}")
            if turn.user_natural_language:
                lines.append(f"  User: {turn.user_natural_language}")
            elif turn.user_response:
                lines.append(f"  User chose: {turn.user_response}")
        else:
            items = turn.agent_action.get("recommended_items", [])
            lines.append(f"  Agent recommended: {', '.join(items[:3])}")
            lines.append(f"  User: {turn.user_natural_language or turn.user_response or '?'}")
    return "\n".join(lines) if lines else "  (start of conversation)"


def _parse_proposals(text):
    proposals = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line.upper().startswith("ATTRIBUTE"):
            continue
        m = re.match(r"ATTRIBUTE:\s*(.+?)\s*\|\s*OPTIONS:\s*(.+)", line, re.IGNORECASE)
        if m:
            attr = m.group(1).strip().lower().replace(" ", "_")
            options = [o.strip() for o in m.group(2).split(",") if o.strip()]
            if attr and options:
                proposals.append((attr, options))
    return proposals


def _ground_proposal(attr_name, proposed_options, data_loader, candidates, max_options=4):
    """Ground LLM-proposed options against real candidate data.
    Keeps valid proposed options, fills gaps from distribution, adds Others for coverage."""
    dist = data_loader.get_attribute_distribution(candidates, attr_name)
    if not dist:
        return None

    real_values = set(dist.keys())
    valid = [o for o in proposed_options if o in real_values]

    # fill from distribution if LLM missed important values
    sorted_vals = sorted(dist.items(), key=lambda x: -x[1])
    for val, _ in sorted_vals:
        if len(valid) >= max_options:
            break
        if val not in valid:
            valid.append(val)

    if not valid:
        return None

    # ensure Others covers any uncovered values
    covered = set(valid)
    uncovered = sum(c for v, c in dist.items() if v not in covered)
    if uncovered > 0 and "Others" not in valid:
        if len(valid) < max_options:
            valid.append("Others")
        else:
            valid[-1] = "Others"

    if len(valid) < 2:
        return None
    if not data_loader.attributes.get(attr_name):
        return None
    return AskAction(attribute_name=attr_name, options=valid)


def _structural_action(attr_name, data_loader, candidates, max_options=4):
    """Build a structural AskAction from data distribution (no LLM)."""
    dist = data_loader.get_attribute_distribution(candidates, attr_name)
    if not dist:
        return None

    sorted_vals = sorted(dist.items(), key=lambda x: -x[1])
    if len(sorted_vals) <= max_options:
        options = [v for v, _ in sorted_vals]
    else:
        n_specific = max_options - 1
        options = [v for v, _ in sorted_vals[:n_specific]]
        if "Others" not in options:
            options.append("Others")
        else:
            for v, _ in sorted_vals[n_specific:]:
                if v != "Others":
                    options.append(v)
                    break

    if len(options) < 2:
        return None
    if not data_loader.attributes.get(attr_name):
        return None
    return AskAction(attribute_name=attr_name, options=options)


class ActionProposer:
    def __init__(self, model_manager, data_loader, dataset="inspired"):
        self.model_manager = model_manager
        self.data_loader = data_loader
        self.dataset = dataset
        self.item_noun = get_domain(dataset)["item_noun"]

    def propose_actions(self, state, candidates):
        """Propose actions for ALL available attributes.
        LLM proposes options for each, then grounded against real data.
        Any attribute the LLM missed gets structural fallback options."""
        asked = state.asked_attributes

        prompt = ACTION_PROPOSAL_TEMPLATE.format(
            item_noun=self.item_noun,
            history=_build_history_str(state),
            candidate_list=_build_candidate_list(self.data_loader, candidates, state.belief),
            attr_summary=_build_attr_summary(self.data_loader, candidates, asked),
            asked=", ".join(asked) if asked else "(none)")

        messages = [{"role": "system", "content": ACTION_PROPOSAL_SYSTEM},
                    {"role": "user", "content": prompt}]

        response = self.model_manager.generate(messages, max_new_tokens=256)
        proposals = _parse_proposals(response)

        # ground LLM proposals
        actions = []
        covered_attrs = set()
        for attr_name, proposed_opts in proposals:
            if attr_name in asked or attr_name in covered_attrs:
                continue
            action = _ground_proposal(attr_name, proposed_opts, self.data_loader, candidates)
            if action:
                actions.append(action)
                covered_attrs.add(attr_name)

        # fill in any attributes the LLM missed with structural fallback
        for attr_name in self.data_loader.get_askable_attributes():
            if attr_name in asked or attr_name in covered_attrs:
                continue
            action = _structural_action(attr_name, self.data_loader, candidates)
            if action:
                actions.append(action)
                covered_attrs.add(attr_name)

        return actions
