"""
User simulators and LLM model management.
"""

import re
import random
from dataclasses import dataclass, field
from typing import List, Optional
from prompts import get_domain, user_system_prompt, user_question_prompt, user_accept_prompt, user_reject_prompt

MODEL_CACHE_DIR = "."
NO_PREFERENCE = "No preference"
OTHERS = "Others"


@dataclass
class UserResponse:
    """Response from user simulator after an agent action."""
    selected_option: Optional[str] = None    # structured answer for ask actions
    is_accept: bool = False                  # True if user accepts recommendation
    is_reject: bool = False                  # True if user rejects recommendation
    natural_language: str = ""               # free-form response text
    attribute_name: str = ""                 # which attribute was asked
    attribute_value: Optional[str] = None    # ground-truth attribute value
    confidence: float = 1.0


@dataclass
class AgentAction:
    """Action passed from environment to user simulator."""
    action_type: str                         # "ask" or "recommend"
    attribute_name: str = ""                 # for ask: which attribute
    options: List[str] = field(default_factory=list)  # for ask: multiple choice options
    question_text: str = ""                  # for ask: verbalized question
    recommended_items: List[str] = field(default_factory=list)  # for recommend: item IDs


def _find_matching_option(target_value, options):
    """Find which option matches the target item's attribute value."""
    if target_value is None:
        if NO_PREFERENCE in options:
            return NO_PREFERENCE
        if OTHERS in options:
            return OTHERS
        return random.choice(options)
    if isinstance(target_value, list):
        for opt in options:
            if opt != OTHERS and opt != NO_PREFERENCE and opt in target_value:
                return opt
        return OTHERS if OTHERS in options else random.choice(options)
    if target_value in options:
        return target_value
    return OTHERS if OTHERS in options else random.choice(options)


def _check_target(target_item, items):
    """Check if the target item is among the recommended items."""
    tid = target_item.item_id
    tname = target_item.name
    for item in items:
        if item == tid or item.lower() == tname.lower():
            return True
        if tname.lower() in item.lower() or item.lower() in tname.lower():
            return True
    return False


def _extract_generated_text(output):
    """Extract generated text from HuggingFace pipeline output.
    Handles various nested output formats from different models."""
    if isinstance(output, list) and output:
        o = output[0]
        if isinstance(o, list):
            g = o[-1].get('generated_text', '')
            if isinstance(g, list): g = g[-1].get('content', '')
        else:
            g = o.get('generated_text', '')
            if isinstance(g, list): g = g[-1].get('content', '')
        return g.strip()
    return ''


class DeterministicUserSimulator:
    """Fast user simulator for MCTS rollouts. No LLM calls.
    Looks up target item's attribute and picks the matching option."""

    def __init__(self, target_item):
        self.target_item = target_item

    def respond(self, action):
        if action.action_type == "ask":
            target_val = self.target_item.get_attribute(action.attribute_name)
            selected = _find_matching_option(target_val, action.options)
            return UserResponse(
                selected_option=selected, attribute_name=action.attribute_name,
                natural_language=f"I'd say {selected}.", confidence=1.0)
        else:
            hit = _check_target(self.target_item, action.recommended_items)
            nl = "Yes, that's exactly what I was looking for!" if hit else "No, that's not what I had in mind."
            return UserResponse(is_accept=hit, is_reject=not hit, natural_language=nl, confidence=1.0)


class ModelManager:
    """Generic HuggingFace model wrapper for LLM generation.
    Used for both user simulator and system backbone LLM."""

    def __init__(self, model_name="meta-llama/Llama-3.2-3B-Instruct", device="cuda"):
        self.model_name = model_name
        self.device = device
        self._pipeline = None
        self._tokenizer = None
        self._loaded = False

    def load_model(self):
        if self._loaded:
            return
        print(f"Loading model: {self.model_name}...")
        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
        import torch
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=MODEL_CACHE_DIR, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, cache_dir=MODEL_CACHE_DIR, torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None, trust_remote_code=True)
        self._pipeline = pipeline(
            "text-generation", model=model, tokenizer=self._tokenizer,
            device_map="auto" if self.device == "cuda" else None)
        self._loaded = True
        print(f"Model loaded: {self.model_name}")

    def generate(self, messages, max_new_tokens=64):
        """Generate text from chat messages."""
        from transformers import GenerationConfig
        config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_return_sequences=1,
            pad_token_id=self._tokenizer.eos_token_id)
        out = self._pipeline(messages, generation_config=config)
        return _extract_generated_text(out)


class LLMUserSimulator:
    """LLM-based user simulator for evaluation."""

    def __init__(self, target_item, model_manager, dataset="inspired",
                 askable_attributes=None):
        self.target_item = target_item
        self.model_manager = model_manager
        self.dataset = dataset
        self.askable_attributes = askable_attributes or []
        self.conversation_history = []

        self.domain = get_domain(dataset)
        # system prompt describes the user's role and target characteristics
        self.system_prompt = user_system_prompt(
            dataset, target_item.attributes, self.askable_attributes, target_item.name)
        # name variants for leakage detection
        self.name_variants = self._get_name_variants()

    def _get_name_variants(self):
        variants = [self.target_item.name.lower()]
        name = self.target_item.name
        if '(' in name:
            t = name.split('(')[0].strip()
            if len(t) > 3:
                variants.append(t.lower())
        return list(set(variants))

    def _filter_leakage(self, text):
        """Remove target item name from LLM response to prevent information leakage."""
        r = self.domain.get("leakage_replacement", "[an item I like]")
        for v in self.name_variants:
            text = re.compile(re.escape(v), re.IGNORECASE).sub(r, text)
        return text

    def respond(self, action):
        if action.action_type == "ask":
            return self._respond_question(action)
        return self._respond_recommend(action)

    def _respond_question(self, action):
        """Respond to an ask action."""
        target_val = self.target_item.get_attribute(action.attribute_name)
        selected = _find_matching_option(target_val, action.options)

        # LLM generates natural language response for the selected option
        prompt = user_question_prompt(self.dataset, action.question_text, action.options, selected)
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}]

        nl = self.model_manager.generate(messages)
        if not nl:
            nl = f"I'd prefer {selected}."
        if any(v in nl.lower() for v in self.name_variants):
            nl = self._filter_leakage(nl)

        self.conversation_history.append(f"Agent: {action.question_text}")
        self.conversation_history.append(f"User: {nl}")

        return UserResponse(
            selected_option=selected, attribute_name=action.attribute_name,
            attribute_value=str(target_val) if target_val else None,
            natural_language=nl, confidence=1.0)

    def _respond_recommend(self, action):
        """Respond to a commit action."""
        hit = _check_target(self.target_item, action.recommended_items)

        if hit:
            prompt = user_accept_prompt(self.dataset, action.recommended_items)
        else:
            prompt = user_reject_prompt(
                self.dataset, action.recommended_items,
                self.target_item.attributes, self.askable_attributes)

        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}]

        nl = self.model_manager.generate(messages)
        if not nl:
            nl = "Yes, that's it!" if hit else "Not quite what I'm looking for."
        if any(v in nl.lower() for v in self.name_variants):
            nl = self._filter_leakage(nl)

        return UserResponse(is_accept=hit, is_reject=not hit, natural_language=nl, confidence=1.0)
