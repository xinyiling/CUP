import json
import os
import random
from collections import Counter
from .base import BaseDataLoader, Item, Attribute
from .utils import batch_get_item_texts

YEAR_BUCKETS = [
    ("before_2000", lambda y: y < 2000),
    ("2000-2009", lambda y: 2000 <= y < 2010),
    ("2010-2015", lambda y: 2010 <= y < 2016),
    ("2016-2018", lambda y: 2016 <= y < 2019),
    ("2019-2020", lambda y: y >= 2019),
]

COUNTRY_BUCKETS = {
    "USA_only": lambda c: c == "USA",
    "Europe": lambda c: any(eu in c for eu in ["UK", "France", "Germany", "Spain", "Italy"]),
    "Asia": lambda c: any(a in c for a in ["China", "Japan", "Hong Kong", "Taiwan", "Korea"]),
    "Others": lambda c: True,
}

LANGUAGE_BUCKETS = {
    "English_only": lambda l: l == "English",
    "European": lambda l: any(e in l for e in ["French", "German", "Spanish", "Italian"]),
    "Asian": lambda l: any(a in l for a in ["Japanese", "Mandarin", "Cantonese", "Korean"]),
    "Others": lambda l: True,
}

CANDIDATE_CACHE_DIR = "./candidates"


class InspiredDataLoader(BaseDataLoader):

    def __init__(self, data_path):
        super().__init__(data_path)
        self.db_dict = {}
        self.dialogues = {}
        self.conversation_list = []
        os.makedirs(CANDIDATE_CACHE_DIR, exist_ok=True)

    def load_data(self):
        db_path = os.path.join(self.data_path, "db_dict.json")
        with open(db_path, 'r') as f:
            self.db_dict = json.load(f)

        dialogues_path = os.path.join(self.data_path, "dialogues_test.json")
        if os.path.exists(dialogues_path):
            with open(dialogues_path, 'r') as f:
                dialogues_raw = json.load(f)
            for session_id, data in dialogues_raw.items():
                movie_title = data.get("gt_items")
                dialogue = data.get("dialogue", [])
                if movie_title and dialogue:
                    dialogue_text = " ".join(dialogue)
                    self.dialogues[movie_title] = dialogue_text
                    self.conversation_list.append({
                        "session_id": session_id,
                        "gt_item": movie_title,
                        "dialogue": dialogue_text
                    })

    def _parse_metadata(self, metadata_str):
        result = {}
        for line in metadata_str.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                if value and value.lower() != 'nan':
                    result[key] = value
        return result

    def _get_year_bucket(self, year):
        for name, cond in YEAR_BUCKETS:
            if cond(year):
                return name
        return "2019-2020"

    def _get_country_bucket(self, country):
        for name, cond in COUNTRY_BUCKETS.items():
            if name != "Others" and cond(country):
                return name
        return "Others"

    def _get_language_bucket(self, language):
        for name, cond in LANGUAGE_BUCKETS.items():
            if name != "Others" and cond(language):
                return name
        return "Others"

    def extract_attributes(self):
        for item_id in self.db_dict:
            raw_meta = self.db_dict[item_id]
            parsed = self._parse_metadata(raw_meta)
            attributes = {}

            # genre: store all raw values directly (no bucketing)
            # option-building at question time picks top values + "Others"
            if 'genre' in parsed:
                attributes['genre'] = [g.strip() for g in parsed['genre'].split(',')]

            # year: raw int + bucketed range
            if 'year' in parsed:
                try:
                    year = int(parsed['year'])
                    attributes['year'] = year
                    attributes['year_range'] = self._get_year_bucket(year)
                except ValueError:
                    pass

            # actors: store all raw values directly (no bucketing)
            if 'actors' in parsed:
                attributes['actors'] = [a.strip() for a in parsed['actors'].split(',')]

            if 'director' in parsed:
                attributes['director'] = parsed['director']
            if 'rated' in parsed:
                attributes['rated'] = parsed['rated']

            # country: raw string + bucketed
            if 'country' in parsed:
                attributes['country_raw'] = parsed['country']
                attributes['country'] = self._get_country_bucket(parsed['country'])

            # language: raw string + bucketed
            if 'language' in parsed:
                attributes['language_raw'] = parsed['language']
                attributes['language'] = self._get_language_bucket(parsed['language'])

            if 'short_plot' in parsed:
                attributes['plot'] = parsed['short_plot']
            elif 'long_plot' in parsed:
                attributes['plot'] = parsed['long_plot']
            if 'title' in parsed:
                attributes['title'] = parsed['title']

            self.items[item_id] = Item(
                item_id=item_id, name=item_id,
                attributes=attributes, raw_metadata=raw_meta
            )

    def define_attributes(self):
        genre_vals, year_vals, actor_vals = set(), set(), set()
        rated_vals, country_vals, language_vals = set(), set(), set()

        for item in self.items.values():
            if 'genre' in item.attributes: genre_vals.update(item.attributes['genre'])
            if 'year_range' in item.attributes: year_vals.add(item.attributes['year_range'])
            if 'actors' in item.attributes: actor_vals.update(item.attributes['actors'])
            if 'rated' in item.attributes: rated_vals.add(item.attributes['rated'])
            if 'country' in item.attributes: country_vals.add(item.attributes['country'])
            if 'language' in item.attributes: language_vals.add(item.attributes['language'])

        self.attributes['genre'] = Attribute(
            'genre', 'Genre', 'multi_categorical', sorted(genre_vals),
            'Others' in genre_vals, "What genre of movie are you looking for? Options: {options}")
        self.attributes['year_range'] = Attribute(
            'year_range', 'Release Period', 'categorical', sorted(year_vals),
            False, "When was the movie released? Options: {options}")
        self.attributes['actors'] = Attribute(
            'actors', 'Actors', 'multi_categorical', sorted(actor_vals),
            'Others' in actor_vals, "Which actor are you interested in? Options: {options}")
        self.attributes['rated'] = Attribute(
            'rated', 'Rating', 'categorical', sorted(rated_vals),
            False, "What rating should the movie have? Options: {options}")
        self.attributes['country'] = Attribute(
            'country', 'Country', 'categorical', sorted(country_vals),
            'Others' in country_vals, "What type of production are you looking for? Options: {options}")
        self.attributes['language'] = Attribute(
            'language', 'Language', 'categorical', sorted(language_vals),
            'Others' in language_vals, "What language preference do you have? Options: {options}")

    def num_conversations(self):
        return len(self.conversation_list)

    def get_target_item(self, idx):
        if 0 <= idx < len(self.conversation_list):
            return self.conversation_list[idx]["gt_item"]
        return None

    def get_conversation_text(self, idx_or_title):
        if isinstance(idx_or_title, int):
            if 0 <= idx_or_title < len(self.conversation_list):
                return self.conversation_list[idx_or_title]["dialogue"]
            return ""
        return self.dialogues.get(idx_or_title, "")

    def _get_candidate_cache_path(self):
        return os.path.join(CANDIDATE_CACHE_DIR, "inspired_candidates.json")

    def load_conversation_candidates(self):
        path = self._get_candidate_cache_path()
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            return {int(k): v for k, v in data.items()}
        return None

    def save_conversation_candidates(self, candidates):
        path = self._get_candidate_cache_path()
        with open(path, 'w') as f:
            json.dump(candidates, f, indent=2)

    def retrieve_candidates_sbert(self, similarity_manager, top_k=300, force_recompute=False):
        if not force_recompute:
            cached = self.load_conversation_candidates()
            if cached is not None:
                return cached

        candidates = {}
        gt_miss = 0

        for idx, conv in enumerate(self.conversation_list):
            dialogue_text = conv["dialogue"]
            gt_item = conv["gt_item"]
            retrieved = similarity_manager.retrieve_top_k(dialogue_text, top_k=top_k)
            if gt_item not in retrieved:
                retrieved[random.randint(0, len(retrieved) - 1)] = gt_item
                gt_miss += 1
            candidates[idx] = retrieved

        self.save_conversation_candidates(candidates)
        return candidates

    def get_conversation_candidates(self, idx, similarity_manager=None, top_k=300):
        cached = self.load_conversation_candidates()
        if cached is not None and idx in cached:
            return cached[idx]
        if similarity_manager is None:
            raise ValueError("Candidates not cached and no similarity_manager provided")
        all_cands = self.retrieve_candidates_sbert(similarity_manager, top_k)
        return all_cands.get(idx, [])

    def get_item_texts(self):
        return batch_get_item_texts(self.items, "inspired", db_dict=self.db_dict)
