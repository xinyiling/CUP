"""
CUP: Conversation Uncertainty-aware Planning

Main script. Integrates the three components of CUP:
  (1) Belief & Uncertainty Modeling (Section 3.3)
  (2) Uncertainty-Guided Planning (Section 3.4)
  (3) Language-Grounded Action Execution (Section 3.5)

Models used:
  - LLM: backbone language model for action proposal, verbalization, and refined
         commit (e.g., Qwen3-4B, Mistral-7B-v0.3, Llama-3.1-8B). Denoted as
         LLM in the paper.
  - User simulator: separate LLM that simulates user responses
         (Llama-3.2-3B). Returns simulated response.
  - SBERT (similarity model): sentence-BERT model (all-MiniLM-L6-v2) for computing sim(·)
         in belief initialization and updates.

Inputs:
  - Dataset: Inspired (movies) or Beauty/Fashion/Home (e-commerce products)
  - Each conversation has 1 ground-truth target + 299 SBERT-retrieved distractors

Outputs:
  - records JSON: full conversation traces with turn-by-turn details
"""

import sys
import os
import argparse
import json
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import InspiredDataLoader, LavicDataLoader
from environment import ConversationEnvironment, AskAction, RecommendAction
from simulator import ModelManager, LLMUserSimulator
from mcts import MCTS
from action_proposer import ActionProposer
from evaluate import TurnRecord, EpisodeRecord, compute_metrics
from similarity import SimilarityManager


DATASET_TO_CATEGORY = {
    "beauty": "all_beauty",
    "fashion": "amazon_fashion",
    "home": "amazon_home",
}

def load_dataset(dataset):
    """Load dataset: items, attributes, and question templates."""
    if dataset == "inspired":
        loader = InspiredDataLoader('/fs/ess/PCON0041/xinyi/current/proj/Inspired')
    elif dataset in DATASET_TO_CATEGORY:
        category = DATASET_TO_CATEGORY[dataset]
        loader = LavicDataLoader('/fs/ess/PCON0041/xinyi/current/proj/lavic', category, "test")
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Must be one of: inspired, beauty, fashion, home.")
    loader.load_data()
    loader.extract_attributes()
    loader.define_attributes()
    return loader


def create_simulator(target_item, user_llm=None, dataset="inspired", askable=None):
    """Create user simulator. LLM-based for evaluation, deterministic for MCTS rollouts."""
    if user_llm is not None:
        return LLMUserSimulator(target_item, user_llm, dataset, askable)
    else:
        raise ValueError(
            "User simulator LLM (--sim-model) is required. "
            "Please provide a valid model name via --sim-model."
        )


def refine_commit(state, env, llm, dataset, action):
    """refined commitment refinement (Section 3.5).

    When commitment is triggered but belief is not highly confident
    (max b_t(c) < theta), the LLM performs chain-of-thought reasoning
    over all remaining candidates to identify the best match c*.

    Input:  state (with b_t, h_t, C_t), env, LLM, dataset
    Output: RecommendAction with LLM's pick, or None (fallback to belief top-1)
    """
    from prompts import REFINED_COMMIT_SYSTEM, refined_commit_prompt

    try:
        # all remaining candidates sorted by belief b_t
        all_cands = state.belief.top_k(len(state.candidates))
        cands_with_names = []
        for cid, prob in all_cands:
            item = env.data_loader.get_item(cid)
            if item:
                cands_with_names.append((item.name, prob))

        if not cands_with_names:
            raise ValueError("No valid candidates found for refined commit.")

        # LLM reasons over h_t and C_t to pick c*
        h_t = state.get_history_string()
        prompt = refined_commit_prompt(dataset, cands_with_names, h_t)
        messages = [{"role": "system", "content": REFINED_COMMIT_SYSTEM},
                    {"role": "user", "content": prompt}]
        response = llm.generate(messages, max_new_tokens=512)

        if not response:
            raise ValueError("LLM returned empty response for refined commit.")

        cand_names = [n for n, _ in cands_with_names]

        # Parse strategy 1: look for explicit COMMIT marker
        for marker in ["COMMIT:", "commit:", "Commit:"]:
            if marker in response:
                name = response.split(marker, 1)[1].strip().split('\n')[0].strip()
                # strip common prefixes/quotes
                name = name.strip('"\'*`').strip()
                for cn in cand_names:
                    if name.lower() == cn.lower() or name.lower() in cn.lower() or cn.lower() in name.lower():
                        return RecommendAction(recommended_items=[cn])

        # Parse strategy 2: find the last candidate name mentioned in the response
        last_pos = -1
        last_match = None
        for cn in cand_names:
            # find last occurrence of candidate name
            pos = response.lower().rfind(cn.lower())
            if pos > last_pos:
                last_pos = pos
                last_match = cn
        if last_match is not None:
            return RecommendAction(recommended_items=[last_match])

        raise ValueError("Could not parse any candidate from LLM response.")

    except Exception as e:
        print(f"Warning: refined commit failed ({e}).")
        return action


def verbalize_action(action, state, llm, dataset):
    """Verbalize structured action into natural language utterance.

    Input:  selected action, state (h_t, b_t), system LLM, dataset
    Output: natural language utterance string utt_t
    """
    from prompts import AGENT_VERBALIZE_SYSTEM, agent_ask_prompt, agent_commit_prompt

    h_t = state.get_history_string()

    if isinstance(action, AskAction):
        # ask: LLM generates question about attribute with options
        prompt = agent_ask_prompt(dataset, action.attribute_name, action.options, h_t)
    elif isinstance(action, RecommendAction):
        # commit: LLM generates recommendation utterance for c*
        commit_name = action.recommended_items[0]
        prompt = agent_commit_prompt(dataset, commit_name, h_t)
    else:
        raise ValueError(f"Unknown action type: {type(action)}")

    messages = [{"role": "system", "content": AGENT_VERBALIZE_SYSTEM},
                {"role": "user", "content": prompt}]
    utterance = llm.generate(messages, max_new_tokens=128)
    return utterance


def execute_turn(action, state, env, record, llm=None, dataset=None):
    """Execute one turn of the conversation.

    Steps:
      1. Verbalize: utt_t = LLM(a*_t, h_t, C_t, b_t)  [Eq.9]
      2. Simulate user responds and update belief  [Eq.10]
      3. Record post-action state and metrics

    Input:  action (AskAction or RecommendAction), current state, env, record
    Output: (next_state, reward, done)
    """
    # record turn information
    turn = TurnRecord(turn_id=state.turn_count,
                      action_type="recommend" if isinstance(action, RecommendAction) else "ask")
    if isinstance(action, RecommendAction):
        turn.recommended_items = action.recommended_items
    elif isinstance(action, AskAction):
        turn.attribute_name = action.attribute_name
        turn.options = action.options
    else:
        raise ValueError(f"Unknown action type in execute_turn: {type(action)}")

    # verbalize action into natural language
    if llm:
        utterance = verbalize_action(action, state, llm, dataset)
    else:
        raise ValueError("Backbone LLM (--system-model) is required for verbalization.")

    # simulate user responds and execute candidates filter, belief updates
    next_state, reward, done = env.step(action, utterance=utterance)

    # record post-action information
    turn.num_candidates_after = next_state.num_candidates
    turn.entropy_after = next_state.current_entropy
    turn.info_gain = max(0, state.current_entropy - turn.entropy_after)
    turn.belief_after = dict(next_state.belief.probs)
    turn.reward = reward

    last = next_state.history[-1]
    turn.agent_utterance = last.agent_utterance
    turn.user_natural_language = last.user_natural_language
    if isinstance(action, RecommendAction):
        turn.accepted = (last.user_response == "accept")
    else:
        turn.user_response = last.user_response

    record.add_turn(turn)
    return next_state, reward, done



def evaluate(loader, dataset, indices, sbert, user_llm, K=50, T=5, c=1.4, 
             softmax_temp=0.1, delta=1.0, epsilon=0.5, theta=0.8, action_proposer=None, llm=None):
    """Run CUP evaluation over conversations.

    Per-conversation flow:
      1. Initialize: load candidates, init belief b_0
      2. Per-turn loop:
         (a) Section 3.3 - Belief & Uncertainty, then check commitment trigger
         (b) Section 3.4 - Planning: if not committing, MCTS search with
             proposed actions and EIG-based priors
         (c) Section 3.5 - Execution: verbalize, simulate user responds, update belief

    Output: evaluated results
    """
    if llm is None:
        raise ValueError("Backbone LLM is required. Provide --system-model.")
    if user_llm is None:
        raise ValueError("User simulator LLM is required. Provide --sim-model.")
    if sbert is None:
        raise ValueError("SBERT similarity manager is required. Provide --similarity-model.")
    if action_proposer is None:
        raise ValueError("ActionProposer is required for LLM-based action proposal.")

    records = []
    askable = loader.get_askable_attributes()

    for idx in tqdm(indices):
        """For each conversation"""
        try:
            target_id = loader.get_target_item(idx)
            if not target_id:
                continue
            candidates = loader.get_conversation_candidates(idx, sbert)
            target_item = loader.get_item(target_id)
            if not target_item:
                raise ValueError(f"Target item {target_id} not found in data loader.")
            if not candidates:
                raise ValueError(f"No candidates retrieved for conversation {idx}.")
            if target_id not in candidates:
                raise ValueError(f"Target {target_id} not in candidate set for conversation {idx}.")

            # Belief & Uncertainty Modeling
            # initialize environment with SBERT-based belief b_0
            env = ConversationEnvironment(loader, T=T, similarity_manager=sbert,
                                          softmax_temperature=softmax_temp, delta=delta, 
                                          dataset_name=dataset, epsilon=epsilon, theta=theta)

            # Initialize a user simulator with target_item as user's intent
            simulator = create_simulator(target_item, user_llm, dataset, askable)
            conv_idx = idx if dataset != "inspired" else None
            state = env.reset(target_item, simulator, candidates, conv_idx)
            initial_belief = dict(state.belief.probs)
            record = EpisodeRecord(idx, target_id, target_item.name, "CUP", 
                                   initial_entropy=state.current_entropy,
                                   initial_candidates=len(candidates),
                                   initial_belief=initial_belief)

            done = False
            while not done:
                # check commitment trigger and returns only [RecommendAction] if forced to commit
                available = env.get_available_actions(proposer=action_proposer)
                if not available:
                    raise RuntimeError(f"No available actions at turn {state.turn_count}.")
                forced_commit = not any(isinstance(a, AskAction) for a in available)

                if forced_commit:
                    # commitment execution
                    # direct commit when confident enough, refined commit otherwise
                    top_prob = max(state.belief.probs.values())
                    if top_prob < theta:
                        action = refine_commit(state, env, llm, dataset, available[0])
                    else:
                        action = available[0]
                else:
                    # Uncertainty-Guided Planning
                    mcts = MCTS(env, target_item, K, c, action_proposer=action_proposer)
                    mcts_action = mcts.search(state)
                    if mcts_action is None:
                        raise RuntimeError("MCTS returned None action.")

                    # if MCTS chose to commit, apply refined commitment
                    if isinstance(mcts_action, RecommendAction) and max(state.belief.probs.values()) < theta:
                        action = refine_commit(state, env, llm, dataset, mcts_action)
                    else:
                        action = mcts_action

                # Language-Grounded Action Execution
                state, _, done = execute_turn(action, state, env, record, llm, dataset)

            record.success = state.success
            records.append(record)

        except Exception as e:
            print(f"Error in conversation {idx}: {e}.")

    return records



def main():
    parser = argparse.ArgumentParser(description='CUP: Conversation Uncertainty-aware Planning')

    # dataset
    parser.add_argument('--dataset', type=str, required=True, choices=['inspired', 'beauty', 'fashion', 'home'])
    parser.add_argument('--T', type=int, default=5, help='Maximum conversation turns T')

    # model
    parser.add_argument('--system-model', type=str, default='meta-llama/Llama-3.1-8B-Instruct', help='System LLM')
    parser.add_argument('--sim-model', type=str, default='meta-llama/Llama-3.2-3B-Instruct', help='User simulator LLM')
    parser.add_argument('--similarity-model', type=str, default='all-MiniLM-L6-v2', help='SBERT model for sim(·)')
    parser.add_argument('--model-device', type=str, default='cuda', help='Device for all models')

    # MCTS hyperparameters
    parser.add_argument('--K', type=int, default=50, help='MCTS search budget K (num simulations per turn)')
    parser.add_argument('--c', type=float, default=1.4, help='PUCT exploration constant c')

    # belief hyperparameters
    parser.add_argument('--softmax-temperature', type=float, default=0.1, help='Temperature for softmax in belief init')
    parser.add_argument('--delta', type=float, default=1.0, help='Bayesian update exponent')
    parser.add_argument('--epsilon', type=float, default=0.5, help='Normalized entropy threshold ε for commitment trigger')
    parser.add_argument('--theta', type=float, default=0.6, help='Max belief probability threshold for commitment trigger')

    # output
    parser.add_argument('--output-dir', type=str, default='results', help='Directory for saving metrics and records')

    args = parser.parse_args()

    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"System LLM: {args.system_model}")
    print(f"User simulator LLM: {args.sim_model}")
    print(f"Device: {args.model_device}")
    print("=" * 60)

    loader = load_dataset(args.dataset)
    indices = list(range(loader.num_conversations()))

    # similarity model to compute sim(·) for belief update
    sbert = SimilarityManager(args.similarity_model, args.model_device)
    category = DATASET_TO_CATEGORY.get(args.dataset)
    cache_key = "inspired_all" if args.dataset == "inspired" else f"lavic_{category}_all"
    if not sbert.load_embeddings(cache_key):
        sbert.compute_embeddings(loader.get_item_texts(), cache_key)
    loader.retrieve_candidates_sbert(sbert, top_k=300)

    # user simulator LLM
    user_llm = ModelManager(args.sim_model, device=args.model_device)
    user_llm.load_model()

    # system LLM
    llm = ModelManager(args.system_model, device=args.model_device)
    llm.load_model()
    proposer = ActionProposer(llm, loader, args.dataset)

    # run evaluation
    records = evaluate(
        loader, args.dataset, indices, sbert, user_llm,
        args.K, args.T, args.c, args.softmax_temperature, 
        args.delta, args.epsilon, args.theta,
        action_proposer=proposer, llm=llm)

    # save record
    out_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    prefix = args.system_model.split('/')[-1][:12]

    metrics = compute_metrics(records, args.T)

    save_path = os.path.join(out_dir, f"{prefix}_T{args.T}.json")
    save_data = {
        "config": {
            "dataset": args.dataset,
            "system_model": args.system_model,
            "sim_model": args.sim_model,
            "K": args.K, "c": args.c, "T": args.T,
            "delta": args.delta, "epsilon": args.epsilon, "theta": args.theta,
        },
        "metrics": metrics,
        "num_conversations": len(records),
        "conversations": [r.to_dict() for r in records]
    }
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved to: {save_path}")

if __name__ == "__main__":
    main()
