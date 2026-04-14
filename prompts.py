"""
All prompt templates for CUP.
"""

_PRODUCT_DOMAIN = {
    "item_noun": "product",
    "seeker_role": "shopper looking for a specific product",
    "context": "product recommendation conversation",
    "leakage_replacement": "[a product I like]",
}

DOMAIN = {
    "inspired": {
        "item_noun": "movie",
        "seeker_role": "movie lover seeking a specific movie",
        "context": "movie recommendation conversation",
        "leakage_replacement": "[a movie I like]",
    },
    "beauty": _PRODUCT_DOMAIN,
    "fashion": _PRODUCT_DOMAIN,
    "home": _PRODUCT_DOMAIN,
}


def get_domain(dataset):
    return DOMAIN.get(dataset, _PRODUCT_DOMAIN)


def format_options(options):
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(options))


def format_attrs(attributes, askable):
    lines = []
    for attr in askable:
        val = attributes.get(attr)
        if val is not None:
            name = attr.replace("_", " ").title()
            v = ", ".join(str(x) for x in val) if isinstance(val, list) else str(val)
            lines.append(f"- {name}: {v}")
    return "\n".join(lines) if lines else "- No specific preferences"


def user_system_prompt(dataset, item_attributes, askable, item_name=""):
    d = get_domain(dataset)
    attrs = format_attrs(item_attributes, askable)
    return (
        f"You are a {d['seeker_role']} in a conversational {d['item_noun']} recommendation system.\n"
        f"You are looking for a specific {d['item_noun']} with these characteristics:\n"
        f"{attrs}\n\n"
        f"IMPORTANT: NEVER mention the exact name of your target {d['item_noun']}.\n"
        f"Be conversational, concise (under 50 words)."
    )


def user_question_prompt(dataset, question_text, options, selected):
    d = get_domain(dataset)
    opts = format_options(options)
    return (
        f"The recommender asks: \"{question_text}\"\n\n"
        f"Options:\n{opts}\n\n"
        f"Your preference is: {selected}\n\n"
        f"Respond naturally. Do NOT mention the exact name of your target {d['item_noun']}. Under 50 words."
    )


def user_accept_prompt(dataset, recommended_items):
    items_str = ", ".join(recommended_items[:3])
    return (
        f"The recommender suggests: {items_str}\n"
        f"This is exactly what you wanted! Accept enthusiastically. Under 30 words."
    )


def user_reject_prompt(dataset, recommended_items, item_attributes, askable):
    d = get_domain(dataset)
    items_str = ", ".join(recommended_items[:3])
    hints = []
    for attr in askable[:3]:
        v = item_attributes.get(attr)
        if v:
            hints.append(str(v[0]) if isinstance(v, list) else str(v))
    hint = ", ".join(hints) if hints else "something else"
    return (
        f"The recommender suggests: {items_str}\n"
        f"Not what you want. You want: {hint}.\n"
        f"Politely decline. Do NOT mention your target {d['item_noun']}'s name. Under 50 words."
    )


AGENT_VERBALIZE_SYSTEM = "You are a conversational recommendation assistant."

def agent_ask_prompt(dataset, attribute_name, options, conversation_history):
    d = get_domain(dataset)
    opts = format_options(options)
    hist = conversation_history if conversation_history else "(start of conversation)"
    return (
        f"You are in a {d['context']}.\n\n"
        f"Conversation so far:\n{hist}\n\n"
        f"You want to ask the user about their preference on \"{attribute_name.replace('_', ' ')}\".\n"
        f"The options are:\n{opts}\n\n"
        f"Generate a natural, conversational question that asks about this attribute "
        f"and presents these options. Be friendly and concise (1-2 sentences)."
    )


def agent_commit_prompt(dataset, candidate_name, conversation_history, top_candidates=None):
    d = get_domain(dataset)
    hist = conversation_history if conversation_history else "(start of conversation)"
    ctx = ""
    if top_candidates:
        ctx = f"\nTop candidates considered: {', '.join(top_candidates[:5])}\n"
    return (
        f"You are in a {d['context']}.\n\n"
        f"Conversation so far:\n{hist}\n{ctx}\n"
        f"Based on the conversation, you believe \"{candidate_name}\" is the best match.\n"
        f"Generate a natural recommendation utterance presenting this {d['item_noun']} to the user. "
        f"Be confident and concise (1-2 sentences)."
    )


REFINED_COMMIT_SYSTEM = "You are a conversational recommendation assistant."

def refined_commit_prompt(dataset, candidates_with_scores, conversation_history):
    d = get_domain(dataset)
    hist = conversation_history if conversation_history else "(no history)"
    cand_lines = []
    for name, prob in candidates_with_scores[:10]:
        cand_lines.append(f"  - {name} (belief: {prob:.3f})")
    cands_str = "\n".join(cand_lines)
    return (
        f"You are in a {d['context']}.\n\n"
        f"Conversation so far:\n{hist}\n\n"
        f"You must now commit to one {d['item_noun']}. "
        f"Here are the top remaining candidates with their belief probabilities:\n{cands_str}\n\n"
        f"Think step by step:\n"
        f"1. What preferences has the user expressed so far?\n"
        f"2. Which candidate best matches those preferences?\n"
        f"3. Why is it the best match?\n\n"
        f"Format:\n"
        f"REASONING: <your step-by-step analysis>\n"
        f"COMMIT: <exact {d['item_noun']} name>"
    )


ACTION_PROPOSAL_SYSTEM = "You are a planning assistant for a recommendation system."

ACTION_PROPOSAL_TEMPLATE = (
    "You are helping a conversational recommendation system decide what to ask the user.\n\n"
    "Given the conversation so far and the remaining candidate {item_noun}s, "
    "propose options for EVERY available attribute below.\n\n"
    "Conversation history:\n{history}\n\n"
    "Remaining candidates (sorted by belief probability):\n{candidate_list}\n\n"
    "Available attributes and their value distributions among candidates:\n{attr_summary}\n\n"
    "Already asked: {asked}\n\n"
    "For EACH attribute listed above, propose 2-4 options that would best help "
    "distinguish between candidates given the conversation context. "
    "Pick options that split the candidate set into roughly equal groups.\n\n"
    "Respond in this exact format (one line per attribute):\n"
    "ATTRIBUTE: <name> | OPTIONS: <option1>, <option2>, <option3>\n"
    "ATTRIBUTE: <name> | OPTIONS: <option1>, <option2>, <option3>, <option4>\n"
    "...\n\n"
    "You MUST include ALL available attributes. Do NOT skip any."
)



def baseline_dp_messages(dataset, candidates, history, turn, max_turns):
    d = get_domain(dataset)
    cands = "\n".join(f"- {c}" for c in candidates[:20])
    hist = "\n".join(history) if history else "(no history yet)"
    return [
        {"role": "system", "content":
         f"You are a {d['context']} assistant. Recommend a single {d['item_noun']} from the candidates.\n"
         f"Output format: RECOMMENDATION: <exact {d['item_noun']} name>"},
        {"role": "user", "content":
         f"Candidates:\n{cands}\n\nConversation:\n{hist}\n\n"
         f"Turn {turn+1}/{max_turns}. Recommend one {d['item_noun']}."}
    ]


def baseline_cot_messages(dataset, candidates, history, turn, max_turns):
    d = get_domain(dataset)
    cands = "\n".join(f"- {c}" for c in candidates[:20])
    hist = "\n".join(history) if history else "(no history yet)"
    return [
        {"role": "system", "content":
         f"You are a {d['context']} assistant. Think step by step, then recommend.\n"
         f"Format:\nREASONING: <your analysis>\nRECOMMENDATION: <exact {d['item_noun']} name>"},
        {"role": "user", "content":
         f"Candidates:\n{cands}\n\nConversation:\n{hist}\n\n"
         f"Turn {turn+1}/{max_turns}. Reason step by step, then recommend."}
    ]


def baseline_user_sim_messages(dataset, target_attrs, askable, question):
    d = get_domain(dataset)
    lines = []
    for a in askable[:5]:
        v = target_attrs.get(a)
        if v:
            lines.append(f"- {a}: {v[0] if isinstance(v, list) else v}")
    attrs = "\n".join(lines) if lines else "- general preferences"
    return [
        {"role": "system", "content":
         f"You are seeking a specific {d['item_noun']} with:\n{attrs}\n"
         f"Answer naturally. Never reveal the exact {d['item_noun']} name. Under 50 words."},
        {"role": "user", "content": question}
    ]