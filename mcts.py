"""
Monte Carlo Tree Search with EIG prior.
- Edge: action with EIG prior for scoring (Eq.6)
- Node: belief state node in the search tree
- MCTS: search with selection, expansion, rollout, backpropagation

At the root, actions come from LLM proposer (cached once per turn).
Inner nodes and rollouts use structural actions (no LLM, fast).
"""

import math
import random
from environment import ConversationEnvironment, State, AskAction, RecommendAction
from belief import compute_eig_all, compute_eig_commit


class Edge:
    """Action edge connecting parent -> child in the search tree."""
    def __init__(self, action, parent=None, child=None, prior=0.0):
        self.action = action
        self.parent = parent
        self.child = child
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0

    @property
    def q_value(self):
        """V(b_t, a_t) in Eq.6."""
        return self.total_value / self.visit_count if self.visit_count > 0 else 0.0

    def puct_score(self, parent_visits, c):
        """Eq.6"""
        return self.q_value + c * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)


class Node:
    """Belief state node in the search tree."""
    def __init__(self, state, parent_edge=None, is_terminal=False):
        self.state = state
        self.parent_edge = parent_edge
        self.is_terminal = is_terminal
        self.visit_count = 0        # N(b_t)
        self.total_value = 0.0
        self.edges = {}             # action_key -> Edge
        self.is_expanded = False

    def add_child(self, action, child_state, is_terminal=False, prior=0.0):
        child = Node(child_state, is_terminal=is_terminal)
        edge = Edge(action, parent=self, child=child, prior=prior)
        child.parent_edge = edge
        self.edges[_action_key(action)] = edge
        return child

    def select_child_puct(self, c):
        """Select child with highest PUCT score (used during selection phase)."""
        best_score, best_edge = float('-inf'), None
        for edge in self.edges.values():
            score = edge.puct_score(self.visit_count, c)
            if score > best_score:
                best_score = score
                best_edge = edge
        if best_edge is None:
            return None, None, None
        return best_edge.action, best_edge, best_edge.child

    def select_best_action(self):
        """Select action with highest visit count (final decision after search)."""
        best_count, best_action = -1, None
        for edge in self.edges.values():
            if edge.visit_count > best_count:
                best_count = edge.visit_count
                best_action = edge.action
        return best_action


def _action_key(action):
    """Generate hashable key for deduplication in the tree."""
    if hasattr(action, 'attribute_name'):
        return f"ask_{action.attribute_name}"
    elif hasattr(action, 'recommended_items'):
        return f"rec_{action.recommended_items[0]}"
    return str(action)


class MCTS:
    def __init__(self, env, target_item, num_simulations=50, exploration_constant=1.4,
                 max_rollout_depth=10, discount_factor=0.99, action_proposer=None):
        self.env = env
        self.target_item = target_item
        self.num_simulations = num_simulations
        self.c = exploration_constant
        self.max_depth = max_rollout_depth
        self.gamma = discount_factor
        self.action_proposer = action_proposer
        self.root = None
        self._root_actions = None  # LLM-proposed actions, cached for all simulations

    def search(self, state):
        """Run K MCTS simulations and return best action by visit count."""
        self.root = Node(state, is_terminal=state.terminated)
        # LLM proposes actions at root once; inner nodes use structural actions
        if self.action_proposer is not None:
            self._root_actions = self.env.get_available_actions(proposer=self.action_proposer)
        else:
            self._root_actions = None
        for _ in range(self.num_simulations):
            self._simulate()
        return self.root.select_best_action()

    def _simulate(self):
        """Simulation: selection -> expansion -> rollout -> backprop."""
        node = self.root
        path = [node]

        # selection: walk down tree using PUCT
        while node.is_expanded and not node.is_terminal:
            action, edge, child = node.select_child_puct(self.c)
            if child is None:
                break
            node = child
            path.append(node)

        # expansion: add one new child with EIG prior
        if not node.is_terminal:
            # root uses LLM-proposed actions, inner nodes use structural (fast)
            if node is self.root and self._root_actions is not None:
                available = self._root_actions
            else:
                temp_env = self._temp_env(node.state)
                available = temp_env.get_available_actions()
            explored = set(node.edges.keys())
            unexplored = [a for a in available if _action_key(a) not in explored]

            if unexplored:
                # sample unexplored action weighted by EIG prior
                priors = self._get_priors(node.state, available)
                action = self._sample_by_prior(unexplored, priors)
                key = _action_key(action)
                prior = priors.get(key, 1.0 / len(available))
                # simulate action using deterministic user simulator
                child_state, reward, done = self._sim_action(node.state, action)
                child = node.add_child(action, child_state, is_terminal=done, prior=prior)
                node = child
                path.append(node)

            node.is_expanded = len(node.edges) >= len(available)

        # rollout: play out with random policy to estimate value
        value = self._rollout(node.state)

        # backpropagation: update visit counts and values along path
        for n in reversed(path):
            n.visit_count += 1
            n.total_value += value
            if n.parent_edge:
                n.parent_edge.visit_count += 1
                n.parent_edge.total_value += value

    def _get_priors(self, state, available):
        """Compute EIG-based priors for all actions (Eq.4-5).
        Ask actions: EIG over attribute option partitions.
        Commit actions: EIG over accept/reject outcomes.
        Then softmax to get probability distribution."""
        eig_scores = {}

        # ask actions
        ask_actions = [a for a in available if isinstance(a, AskAction)]
        if ask_actions:
            eig_results = compute_eig_all(state.belief, state.candidates, self.env.data_loader, state.asked_attributes, max_options=4)
            for a in ask_actions:
                eig = eig_results.get(a.attribute_name, {}).get("eig", 0.0)
                eig_scores[_action_key(a)] = eig

        # commit actions
        for a in available:
            if isinstance(a, RecommendAction) and a.recommended_items:
                commit_id = a.recommended_items[0]
                eig_scores[_action_key(a)] = compute_eig_commit(state.belief, commit_id)

        # softmax normalization (Eq.5)
        if not eig_scores:
            n = len(available)
            return {_action_key(a): 1.0 / n for a in available}

        max_v = max(eig_scores.values())
        exp_vals = {k: math.exp(v - max_v) for k, v in eig_scores.items()}
        total = sum(exp_vals.values())
        if total <= 0:
            n = len(available)
            return {_action_key(a): 1.0 / n for a in available}
        return {k: v / total for k, v in exp_vals.items()}

    def _sample_by_prior(self, actions, priors):
        """Weighted random sampling during expansion."""
        weights = [priors.get(_action_key(a), 0.01) for a in actions]
        total = sum(weights)
        if total <= 0:
            return random.choice(actions)
        r = random.random() * total
        cumsum = 0
        for a, w in zip(actions, weights):
            cumsum += w
            if r <= cumsum:
                return a
        return actions[-1]

    def _sim_action(self, state, action):
        """Simulate one action in cloned environment with deterministic user."""
        temp = self._temp_env(state)
        next_state, reward, done = temp.step(action)
        return next_state, reward, done

    def _temp_env(self, state):
        """Create throwaway environment for simulation (no LLM, fast)."""
        from simulator import DeterministicUserSimulator
        temp = self.env.clone()
        temp.state = state.copy()
        temp.user_simulator = DeterministicUserSimulator(self.target_item)
        return temp

    def _rollout(self, state):
        """Eq.7 via random playout. Default commit if |C|<=2, else random action."""
        if state.terminated:
            return 1.0 if state.success else 0.0

        temp = self._temp_env(state)
        current = state.copy()
        total_reward = 0.0
        discount = 1.0

        for _ in range(self.max_depth):
            if current.terminated:
                break
            temp.state = current
            actions = temp.get_available_actions()
            if not actions:
                break

            # simple rollout policy
            rec = [a for a in actions if isinstance(a, RecommendAction)]
            if current.num_candidates <= 2 and rec:
                action = rec[0]
            else:
                action = random.choice(actions)

            current, reward, done = temp.step(action)
            total_reward += discount * reward
            discount *= self.gamma
            if done:
                break

        return total_reward
