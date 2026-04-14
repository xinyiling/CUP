from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class Item:
    item_id: str
    name: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    raw_metadata: str = ""

    def get_attribute(self, attr_name):
        return self.attributes.get(attr_name)

    def has_attribute_value(self, attr_name, value):
        attr_val = self.attributes.get(attr_name)
        if attr_val is None:
            return False
        if isinstance(attr_val, list):
            return value in attr_val
        return attr_val == value


@dataclass
class Attribute:
    name: str
    display_name: str
    attr_type: str
    possible_values: List[str]
    has_others: bool = False
    question_template: str = ""


class BaseDataLoader:
    NO_PREFERENCE = "No preference"

    def __init__(self, data_path):
        self.data_path = data_path
        self.items: Dict[str, Item] = {}
        self.candidates: List[str] = []
        self.attributes: Dict[str, Attribute] = {}

    def load_data(self):
        raise NotImplementedError

    def extract_attributes(self):
        raise NotImplementedError

    def define_attributes(self):
        raise NotImplementedError

    def get_item(self, item_id):
        return self.items.get(item_id)

    def get_candidates(self):
        return list(self.items.keys())

    def filter_candidates(self, candidate_ids, attr_name, attr_value):
        return [cid for cid in candidate_ids
                if self.items.get(cid) and self.items[cid].has_attribute_value(attr_name, attr_value)]

    def get_attribute_distribution(self, candidate_ids, attr_name):
        distribution = {}
        for cid in candidate_ids:
            item = self.items.get(cid)
            if not item:
                continue
            attr_val = item.get_attribute(attr_name)
            if attr_val is None:
                distribution[self.NO_PREFERENCE] = distribution.get(self.NO_PREFERENCE, 0) + 1
                continue
            if isinstance(attr_val, list):
                for val in attr_val:
                    distribution[val] = distribution.get(val, 0) + 1
            else:
                distribution[attr_val] = distribution.get(attr_val, 0) + 1
        return distribution

    def get_askable_attributes(self):
        return list(self.attributes.keys())

    def num_conversations(self):
        return 0

    def get_target_item(self, idx):
        return None

    def get_conversation_candidates(self, idx, similarity_manager=None, top_k=300):
        return []

    def get_conversation_text(self, idx):
        return ""

    def get_item_texts(self):
        return {}
