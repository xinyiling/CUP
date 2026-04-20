"""
Conversation environment for CUP.
- AskAction / RecommendAction: structured action types
- Turn: one agent-user exchange with entropy tracking
- State: conversation state s_t = {h_t, C_t, b_t}
- ConversationEnvironment: manages state transitions, reward (Eq.8), belief updates (Eq.10), and commitment trigger
"""

import math
from dataclasses import dataclass, field
from copy import deepcopy
from typing import Dict, List, Any, Optional
from belief import BeliefState
from simulator import AgentAction


@dataclass
class AskAction:
    """Ask about an attribute with multiple-choice options."""
    attribute_name: str
    options: List[str]

    def to_dict(self):
        return {"action_type": "ask", "attribute_name": self.attribute_name,
                "options": self.options}


@dataclass
class RecommendAction:
    """Commit: recommend item(s) to the user."""
    recommended_items: List[str]

    def to_dict(self):
        return {"action_type": "recommend", "recommended_items": self.recommended_items}


@dataclass
class Turn:
    """One agent-user exchange in the conversation."""
    turn_id: int
    agent_action_type: str          # "ask" or "recommend"
    agent_action: Dict[str, Any]    # serialized action
    agent_utterance: str = ""       # LLM-verbalized utterance (utt_t)
    user_response: Optional[str] = None      # structured response
    user_natural_language: str = ""           # free-form response (r_t)
    entropy_before: float = 0.0     # H(b_t) before this turn
    entropy_after: float = 0.0      # H(b_{t+1}) after this turn
    info_gain: float = 0.0          # entropy_before - entropy_after


@dataclass
class State:
    """Conversation state s_t = {h_t, C_t, b_t}."""
    history: List[Turn] = field(default_factory=list)      # h_t
    candidates: List[str] = field(default_factory=list)    # C_t
    belief: BeliefState = field(default_factory=BeliefState)  # b_t
    turn_count: int = 0
    terminated: bool = False
    success: bool = False

    @property
    def num_candidates(self):
        return len(self.candidates)

    @property
    def asked_attributes(self):
        return self.belief.get_asked_attributes()

    @property
    def current_entropy(self):
        return self.belief.entropy()

    def get_history_string(self):
        """Format h_t as text for LLM prompts."""
        lines = []
        for turn in self.history:
            if turn.agent_action_type == "ask":
                lines.append(f"Recommender: {turn.agent_action.get('question_text', '')}")
            else:
                items = turn.agent_action.get("recommended_items", [])
                lines.append(f"Recommender: I recommend {', '.join(items[:3])}")
            if turn.user_natural_language:
                lines.append(f"Seeker: {turn.user_natural_language}")
        return "\n".join(lines)

    def copy(self):
        return State(
            history=deepcopy(self.history), candidates=self.candidates.copy(),
            belief=self.belief.copy(), turn_count=self.turn_count,
            terminated=self.terminated, success=self.success)

NO_PREFERENCE = "No preference"

def get_ask_actions(data_loader, candidates, asked_attributes, max_options=4):
    """Build structural AskActions from candidate attribute distributions."""
    actions = []
    for attr_name in data_loader.get_askable_attributes():
        if attr_name in asked_attributes:
            continue
        attr_info = data_loader.attributes.get(attr_name)
        if not attr_info:
            continue
        dist = data_loader.get_attribute_distribution(candidates, attr_name)
        if not dist:
            continue

        # top values by frequency, fill to max_options, add "Others" for uncovered
        sorted_vals = sorted(dist.items(), key=lambda x: -x[1])
        if len(sorted_vals) <= max_options:
            options = [v for v, _ in sorted_vals]
        else:
            # reserve one slot for "Others" if not already in top values
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

        if len(options) >= 2:
            actions.append(AskAction(attribute_name=attr_name, options=options))
    return actions


def get_recommend_action(candidates, top_k=1):
    """Build RecommendAction with top-k candidates by belief."""
    return RecommendAction(recommended_items=candidates[:top_k])


class ConversationEnvironment:
    """Manages one conversation episode: state, transitions, rewards."""

    def __init__(self, data_loader, T=5, max_recommendations=1,
                 lam=0.1, alpha=0.2, beta=0.5,
                 similarity_manager=None, softmax_temperature=0.1,
                 delta=1.0, dataset_name="inspired",
                 epsilon=0.5, theta=0.8):
        self.data_loader = data_loader
        self.T = T
        self.max_recommendations = max_recommendations
        self.lam = lam
        self.alpha = alpha
        self.beta = beta
        self.similarity_manager = similarity_manager  # SBERT for sim(·)
        self.softmax_temperature = softmax_temperature
        self.delta = delta
        self.dataset_name = dataset_name
        self.epsilon = epsilon
        self.theta = theta

        self.state = None
        self.user_simulator = None
        self.target_item = None
        self.conversation_text = ""   # accumulated conversation text

    def reset(self, target_item, user_simulator, initial_candidates=None, conversation_idx=None):
        """Initialize conversation: set up candidates, belief b_0 via SBERT (Eq.1)."""
        self.target_item = target_item
        self.user_simulator = user_simulator
        candidates = initial_candidates or self.data_loader.get_candidates()

        # belief initialization
        belief = BeliefState()
        if self.similarity_manager:
            # get conversation context h_0 for similarity computation
            query_text = None
            if self.dataset_name == "inspired":
                query_text = self.data_loader.get_conversation_text(target_item.item_id)
            elif conversation_idx is not None:
                query_text = self.data_loader.get_conversation_text(conversation_idx)

            if query_text:
                self.conversation_text = query_text
                # Eq.1: b_0(c_i) = softmax(sim(c_i, h_0))
                scores = self.similarity_manager.compute_similarity(
                    query_text, candidates, temperature=self.softmax_temperature)
                belief.initialize_from_similarity(candidates, scores)
            else:
                belief.initialize_uniform(candidates)
        else:
            belief.initialize_uniform(candidates)

        self.state = State(history=[], candidates=candidates.copy(), belief=belief,
                           turn_count=0, terminated=False, success=False)
        return self.state.copy()

    def step(self, action, utterance=None):
        """Execute action, get user response, update state.
        Returns (next_state, reward, done)."""
        if isinstance(action, AskAction):
            return self._execute_ask(action, utterance)
        else:
            return self._execute_recommend(action, utterance)

    def _execute_ask(self, action, utterance=None):
        """Execute ask action: user responds, filter candidates, update belief."""
        entropy_before = self.state.belief.entropy()

        # use verbalized utterance if available
        if utterance:
            agent_text = utterance
        else:
            attr_info = self.data_loader.attributes.get(action.attribute_name)
            opts_str = ", ".join(action.options)
            agent_text = attr_info.question_template.format(options=opts_str) if attr_info and attr_info.question_template else f"What {action.attribute_name}? Options: {opts_str}"
        agent_action = AgentAction(action_type="ask", attribute_name=action.attribute_name, options=action.options, question_text=agent_text)
        response = self.user_simulator.respond(agent_action)

        # record constraint and filter candidates
        self.state.belief.add_constraint(action.attribute_name, response.selected_option)
        self._filter_candidates(action.attribute_name, response.selected_option, action.options)

        # update conversation history
        turn_text = f"{agent_text} {response.selected_option}"
        if response.natural_language:
            turn_text += f" {response.natural_language}"
        self.conversation_text += f" {turn_text}"

        # Eq.10: update belief state
        if self.similarity_manager and self.state.candidates:
            sim_array = self.similarity_manager.compute_similarity(
                self.conversation_text, self.state.candidates, temperature=self.softmax_temperature)
            sim_dict = {cid: float(s) for cid, s in zip(self.state.candidates, sim_array)}
            self.state.belief.update_bayesian(sim_dict, self.state.candidates, self.delta)
        else:
            self.state.belief.filter_candidates(self.state.candidates)

        entropy_after = self.state.belief.entropy()
        ig = max(0.0, entropy_before - entropy_after)
        turn = Turn(self.state.turn_count, "ask", action.to_dict(), agent_text,
                    response.selected_option, response.natural_language,
                    entropy_before, entropy_after, ig)
        self.state.history.append(turn)
        self.state.turn_count += 1

        done = self._check_termination()
        reward = self._compute_reward(done, "ask", info_gain=ig)
        return self.state.copy(), reward, done

    def _execute_recommend(self, action, utterance=None):
        """Execute commit action: user accepts or rejects."""
        entropy_before = self.state.belief.entropy()
        agent_text = utterance or f"I recommend: {', '.join(action.recommended_items[:3])}"
        agent_action = AgentAction(action_type="recommend", recommended_items=action.recommended_items)
        response = self.user_simulator.respond(agent_action)

        if response.is_accept:
            self.state.success = True
            self.state.terminated = True
        else:
            # failed commit: remove recommended item from candidates
            self.state.candidates = [c for c in self.state.candidates if c not in set(action.recommended_items)]

            # update conversation history
            turn_text = f"{agent_text} {response.natural_language}"
            self.conversation_text += f" {turn_text}"

            # Eq.10: update belief state
            if self.similarity_manager and self.state.candidates:
                sim_array = self.similarity_manager.compute_similarity(
                    self.conversation_text, self.state.candidates, temperature=self.softmax_temperature)
                sim_dict = {cid: float(s) for cid, s in zip(self.state.candidates, sim_array)}
                self.state.belief.update_bayesian(sim_dict, self.state.candidates, self.delta)
            else:
                self.state.belief.filter_candidates(self.state.candidates)

        entropy_after = self.state.belief.entropy()
        turn = Turn(self.state.turn_count, "recommend", action.to_dict(), agent_text,
                    "accept" if response.is_accept else "reject",
                    response.natural_language, entropy_before, entropy_after, 0.0)
        self.state.history.append(turn)
        self.state.turn_count += 1

        done = self._check_termination()
        reward = self._compute_reward(done, "recommend", success=response.is_accept)
        return self.state.copy(), reward, done

    def _filter_candidates(self, attr_name, attr_value, options):
        """Filter C_t based on user's attribute selection."""
        if attr_value == NO_PREFERENCE:
            self.state.candidates = [
                cid for cid in self.state.candidates
                if self.data_loader.get_item(cid) and
                   self.data_loader.get_item(cid).get_attribute(attr_name) is None]
        elif attr_value == "Others":
            eliminate = [o for o in options if o != "Others" and o != NO_PREFERENCE]
            elim_none = NO_PREFERENCE in options
            filtered = []
            for cid in self.state.candidates:
                item = self.data_loader.get_item(cid)
                if not item: continue
                val = item.get_attribute(attr_name)
                if val is None:
                    if not elim_none: filtered.append(cid)
                    continue
                if not any(item.has_attribute_value(attr_name, o) for o in eliminate):
                    filtered.append(cid)
            self.state.candidates = filtered
        else:
            self.state.candidates = self.data_loader.filter_candidates(
                self.state.candidates, attr_name, attr_value)

    def _check_termination(self):
        """Check is the conversation should terminate."""
        if self.state.terminated:
            return True
        if self.state.turn_count >= self.T:
            self.state.terminated = True
            return True
        if self.state.num_candidates == 0:
            self.state.terminated = True
            return True
        return False

    def _compute_reward(self, done, action_type, success=False, info_gain=0.0):
        """Eq.8"""
        reward = -self.lam + self.alpha * info_gain
        if success:
            reward += 1.0
        if done and not success and not self.state.success:
            reward -= self.beta
        return reward

    def _sorted_by_belief(self):
        """Candidates sorted by belief probability (highest first)."""
        return sorted(self.state.candidates, key=lambda c: self.state.belief.get_prob(c), reverse=True)

    def get_available_actions(self, proposer=None):
        """Check commitment trigger, then return available actions.

        If triggered: returns [RecommendAction] only.
        Otherwise: returns ask actions (LLM-proposed or structural) + recommend.
        """
        sorted_cands = self._sorted_by_belief()
        probs = sorted(self.state.belief.probs.values(), reverse=True) if self.state.belief.probs else []
        top_prob = probs[0] if probs else 0.0
        n_cands = self.state.num_candidates

        # commitment trigger
        force = (n_cands <= 2 or self.state.turn_count >= self.T - 1)
        if not force and self.epsilon > 0 and self.theta > 0:
            if n_cands > 1:
                ratio = self.state.belief.entropy() / math.log(n_cands)
                if ratio < self.epsilon and top_prob >= self.theta:
                    force = True

        if force:
            return [get_recommend_action(sorted_cands, self.max_recommendations)]

        # LLM-proposed actions or structural fallback
        if proposer is not None:
            ask_actions = proposer.propose_actions(self.state, self.state.candidates)
        else:
            ask_actions = get_ask_actions(self.data_loader, self.state.candidates, self.state.asked_attributes)

        # always include commit as an option so MCTS can consider it
        ask_actions.append(get_recommend_action(sorted_cands, self.max_recommendations))
        return ask_actions

    def clone(self):
        """Deep copy for MCTS simulation. Shares data_loader and similarity_manager."""
        new = ConversationEnvironment(
            self.data_loader, self.T, self.max_recommendations,
            self.lam, self.alpha, self.beta,
            self.similarity_manager, self.softmax_temperature,
            self.delta, self.dataset_name,
            self.epsilon, self.theta)
        if self.state:
            new.state = self.state.copy()
        new.target_item = self.target_item
        new.conversation_text = self.conversation_text
        return new
