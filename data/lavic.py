import re
import json
import os
import random
from collections import Counter
from .base import BaseDataLoader, Item, Attribute
from .utils import get_lavic_conversation_text, batch_get_item_texts

CANDIDATE_CACHE_DIR = "/fs/ess/PCON0041/xinyi/current/proj/cup/candidates"

CATEGORY_CONFIG = {
    "all_beauty": {
        "askable_attrs": [
            "rating_range", "price_range", "brand", "popularity",
            "subcategory", "item_form", "is_discontinued", "product_size"
        ],
        "subcategory_level": 1,
        "top_subcategories": [
            "Skin Care", "Hair Care", "Shave & Hair Removal",
            "Foot, Hand & Nail Care", "Tools & Accessories"
        ],
        "weight_buckets": [
            ("Travel/Sample (< 57g)", 0, 57), ("Small (57-142g)", 57, 142),
            ("Medium (142-284g)", 142, 284), ("Large (284g+)", 284, float("inf")),
        ],
        "top_brands_count": 20, "min_brand_frequency": 3,
        "top_item_forms": [
            "Cream", "Liquid", "Gel", "Lotion", "Oil",
            "Powder", "Bar", "Stick", "Spray", "Serum"
        ],
    },
    "amazon_fashion": {
        "askable_attrs": [
            "rating_range", "price_range", "brand", "popularity",
            "department", "subcategory", "is_discontinued", "product_size"
        ],
        "subcategory_level": 2,
        "top_subcategories": ["Watches", "Clothing", "Shoes", "Accessories", "Jewelry"],
        "weight_buckets": [
            ("Lightweight (< 100g)", 0, 100), ("Medium (100-300g)", 100, 300),
            ("Heavy (300g+)", 300, float("inf")),
        ],
        "top_brands_count": 20, "min_brand_frequency": 3,
        "top_departments": ["men", "women", "unisex-adult", "boys", "girls"],
    },
    "amazon_home": {
        "askable_attrs": [
            "rating_range", "price_range", "brand", "popularity",
            "subcategory", "color", "material", "is_discontinued", "product_size"
        ],
        "subcategory_level": 1,
        "top_subcategories": [
            "Kitchen & Dining", "Household Supplies",
            "Heating, Cooling & Air Quality", "Vacuums & Floor Care", "Bath"
        ],
        "weight_buckets": [
            ("Light (< 200g)", 0, 200), ("Medium (200-500g)", 200, 500),
            ("Heavy (500-2000g)", 500, 2000), ("Very Heavy (2000g+)", 2000, float("inf")),
        ],
        "top_brands_count": 20, "min_brand_frequency": 2,
        "top_colors": ["Black", "White", "Silver", "Clear", "Blue",
                       "Red", "Gray", "Brown", "Gold", "Green"],
        "top_materials": ["Plastic", "Stainless Steel", "Aluminum", "Glass",
                          "Silicone", "Wood", "Ceramic", "Cotton", "Metal", "Bamboo"],
    },
}


_LB_TO_G = 453.59237
_OZ_TO_G = 28.349523125
_KG_TO_G = 1000.0

def _norm(s):
    return re.sub(r"\s+", " ", s).strip() if isinstance(s, str) else s

def _weight_to_grams(value, unit):
    u = unit.strip().lower()
    if u in ("g", "gram", "grams"): return value
    if u in ("kg", "kilogram", "kilograms"): return value * _KG_TO_G
    if u in ("oz", "ounce", "ounces"): return value * _OZ_TO_G
    if u in ("lb", "lbs", "pound", "pounds"): return value * _LB_TO_G
    return None

def _parse_weight(s):
    s = _norm(s)
    if not s: return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(kg|kilograms?|g|grams?|oz|ounces?|lb|lbs|pounds?)\b", s, re.I)
    if not m: return None
    val = float(m.group(1))
    unit_raw = m.group(2).lower()
    if unit_raw.startswith("kg"): unit = "kg"
    elif unit_raw.startswith("g"): unit = "g"
    elif unit_raw.startswith("oz") or unit_raw.startswith("ounce"): unit = "oz"
    else: unit = "lb"
    return _weight_to_grams(val, unit)

def _parse_dimensions_weight(dim_str):
    dim_str = _norm(dim_str)
    if not dim_str: return None
    for part in dim_str.split(";"):
        w = _parse_weight(part.strip())
        if w: return w
    return None

def _normalize_department(v):
    if not isinstance(v, str): return v
    s = _norm(v).lower().replace("\u2019", "'")
    if s in ("women", "womens", "women's", "woman", "ladies"): return "women"
    if s in ("men", "mens", "men's", "man"): return "men"
    if s in ("girls", "girl", "girl's"): return "girls"
    if s in ("boys", "boy", "boy's"): return "boys"
    return s

def _extract_raw(meta):
    details = meta.get("details", {}) if isinstance(meta.get("details"), dict) else {}
    brand = None
    if isinstance(details.get("Brand"), str):
        brand = _norm(details["Brand"])
    if not brand:
        brand = meta.get("store")

    weight_g = None
    for key in ("Item Weight", "Weight Limit"):
        if isinstance(details.get(key), str):
            w = _parse_weight(details[key])
            if w: weight_g = w; break
    if weight_g is None and isinstance(details.get("Product Dimensions"), str):
        weight_g = _parse_dimensions_weight(details["Product Dimensions"])

    dept = details.get("Department")
    if isinstance(dept, str):
        dept = _normalize_department(dept)

    return {
        "title": _norm(meta.get("title", "")),
        "brand": brand,
        "price": float(meta["price"]) if isinstance(meta.get("price"), (int, float)) else None,
        "average_rating": float(meta["average_rating"]) if isinstance(meta.get("average_rating"), (int, float)) else None,
        "rating_number": int(meta["rating_number"]) if isinstance(meta.get("rating_number"), (int, float)) else None,
        "categories": meta.get("categories") if isinstance(meta.get("categories"), list) else None,
        "weight_grams": weight_g,
        "department": dept,
        "color": _norm(details.get("Color", "")) or None,
        "material": _norm(details.get("Material", "")) or None,
        "item_form": _norm(details.get("Item Form", "")) or None,
        "is_discontinued_by_manufacturer": details.get("Is Discontinued By Manufacturer"),
    }


class LavicDataLoader(BaseDataLoader):

    def __init__(self, data_dir, category="all_beauty", split="test"):
        super().__init__(data_dir)
        self.data_dir = data_dir
        self.category = category
        self.split = split
        self.item_db = {}
        self.conversations = []
        self.askable_attrs = []
        self._top_values = {}
        os.makedirs(CANDIDATE_CACHE_DIR, exist_ok=True)

    def load_data(self):
        meta_path = os.path.join(self.data_dir, "item2meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                self.item_db = json.load(f)

        conv_path = os.path.join(self.data_dir, self.category, f"{self.split}.jsonl")
        if os.path.exists(conv_path):
            with open(conv_path, 'r') as f:
                for line in f:
                    if line.strip():
                        self.conversations.append(json.loads(line))

    def extract_attributes(self):
        all_ids_path = os.path.join(self.data_dir, self.category, "all_item_ids.json")
        if os.path.exists(all_ids_path):
            with open(all_ids_path, 'r') as f:
                valid_ids = set(json.load(f))
            for item_id in valid_ids:
                if item_id in self.item_db:
                    self.items[item_id] = self._create_item(item_id, self.item_db[item_id])
        else:
            cat_map = {"all_beauty": "All Beauty", "amazon_fashion": "Amazon Fashion", "amazon_home": "Amazon Home"}
            target = cat_map.get(self.category)
            for item_id, meta in self.item_db.items():
                if target and meta.get("main_category") != target:
                    continue
                self.items[item_id] = self._create_item(item_id, meta)

        self._apply_top_value_mapping()

    def _create_item(self, item_id, meta):
        config = CATEGORY_CONFIG[self.category]
        raw = _extract_raw(meta)
        attrs = {}

        # keep raw numeric values alongside bucketed versions
        if raw["average_rating"] is not None:
            r = raw["average_rating"]
            attrs["average_rating"] = r
            if r >= 4.5: attrs["rating_range"] = "4.5+"
            elif r >= 4.0: attrs["rating_range"] = "4.0-4.5"
            elif r >= 3.5: attrs["rating_range"] = "3.5-4.0"
            elif r >= 3.0: attrs["rating_range"] = "3.0-3.5"
            else: attrs["rating_range"] = "below 3.0"

        if raw["price"] is not None:
            p = raw["price"]
            attrs["price"] = p
            if p < 10: attrs["price_range"] = "Under $10"
            elif p < 25: attrs["price_range"] = "$10-$25"
            elif p < 50: attrs["price_range"] = "$25-$50"
            elif p < 100: attrs["price_range"] = "$50-$100"
            else: attrs["price_range"] = "$100+"

        if raw["rating_number"] is not None:
            n = raw["rating_number"]
            attrs["rating_number"] = n
            if n >= 1000: attrs["popularity"] = "Very Popular (1000+ ratings)"
            elif n >= 100: attrs["popularity"] = "Popular (100-999 ratings)"
            elif n >= 10: attrs["popularity"] = "Moderate (10-99 ratings)"
            else: attrs["popularity"] = "New (<10 ratings)"

        # _raw_brand preserved by _apply_top_value_mapping as brand_raw
        if raw.get("brand"): attrs["_raw_brand"] = str(raw["brand"])

        disc = raw.get("is_discontinued_by_manufacturer")
        if disc is not None:
            s = str(disc).strip().lower()
            if s in ("yes", "true"): attrs["is_discontinued"] = "Discontinued"
            elif s in ("no", "false"): attrs["is_discontinued"] = "Available"

        wg = raw.get("weight_grams")
        if wg and wg > 0:
            attrs["weight_grams"] = wg
            for bname, lo, hi in config["weight_buckets"]:
                if lo <= wg < hi:
                    attrs["product_size"] = bname
                    break

        cats = raw.get("categories")
        level = config["subcategory_level"]
        if cats and isinstance(cats, list) and len(cats) > level:
            attrs["_raw_subcategory"] = cats[level]

        if self.category == "all_beauty":
            if raw.get("item_form"): attrs["_raw_item_form"] = str(raw["item_form"])
        elif self.category == "amazon_fashion":
            if raw.get("department"): attrs["_raw_department"] = str(raw["department"])
        elif self.category == "amazon_home":
            if raw.get("color"): attrs["_raw_color"] = str(raw["color"])
            if raw.get("material"): attrs["_raw_material"] = str(raw["material"])

        if "features" in meta and meta["features"]:
            attrs["features"] = meta["features"][:5]

        title = raw.get("title") or meta.get("title", f"Product {item_id}")
        return Item(item_id=item_id, name=title, attributes=attrs, raw_metadata=json.dumps(meta))

    def _apply_top_value_mapping(self):
        config = CATEGORY_CONFIG[self.category]
        self._map_top_values("_raw_brand", "brand", config["top_brands_count"], config["min_brand_frequency"])
        self._map_to_fixed("_raw_subcategory", "subcategory", config["top_subcategories"])

        if self.category == "all_beauty":
            self._map_to_fixed("_raw_item_form", "item_form", config["top_item_forms"])
        elif self.category == "amazon_fashion":
            self._map_to_fixed("_raw_department", "department", config["top_departments"])
        elif self.category == "amazon_home":
            self._map_to_fixed("_raw_color", "color", config["top_colors"])
            self._map_to_fixed("_raw_material", "material", config["top_materials"])

    def _map_top_values(self, raw_key, final_key, top_n, min_freq):
        counts = Counter()
        for item in self.items.values():
            v = item.attributes.get(raw_key)
            if v: counts[str(v)] += 1

        top = [v for v, c in counts.most_common() if c >= min_freq][:top_n]
        top_set = set(top)

        for item in self.items.values():
            v = item.attributes.get(raw_key)
            if v is None: continue
            # keep raw value, add bucketed version
            item.attributes[f"{final_key}_raw"] = str(v)
            item.attributes[final_key] = str(v) if str(v) in top_set else "Others"
            del item.attributes[raw_key]
        self._top_values[final_key] = top

    def _map_to_fixed(self, raw_key, final_key, allowed):
        allowed_set = set(allowed)
        for item in self.items.values():
            v = item.attributes.get(raw_key)
            if v is None: continue
            # keep raw value, add bucketed version
            item.attributes[f"{final_key}_raw"] = str(v)
            item.attributes[final_key] = str(v) if str(v) in allowed_set else "Others"
            del item.attributes[raw_key]
        self._top_values[final_key] = allowed

    def define_attributes(self):
        config = CATEGORY_CONFIG[self.category]
        self.askable_attrs = list(config["askable_attrs"])

        self.attributes["rating_range"] = Attribute("rating_range", "Rating", "categorical",
            ["4.5+", "4.0-4.5", "3.5-4.0", "3.0-3.5", "below 3.0"], False, "What rating range do you prefer?")
        self.attributes["price_range"] = Attribute("price_range", "Price Range", "categorical",
            ["Under $10", "$10-$25", "$25-$50", "$50-$100", "$100+"], False, "What price range are you looking for?")
        self.attributes["brand"] = Attribute("brand", "Brand", "categorical",
            self._top_values.get("brand", []) + ["Others"], True, "Do you have a brand preference?")
        self.attributes["popularity"] = Attribute("popularity", "Popularity", "categorical",
            ["Very Popular (1000+ ratings)", "Popular (100-999 ratings)", "Moderate (10-99 ratings)", "New (<10 ratings)"],
            False, "How popular should the product be?")
        self.attributes["subcategory"] = Attribute("subcategory", "Product Type", "categorical",
            config["top_subcategories"] + ["Others"], True, "What type of product are you looking for?")
        self.attributes["is_discontinued"] = Attribute("is_discontinued", "Availability", "categorical",
            ["Available", "Discontinued", "I don't care"], False,
            "Do you prefer currently available products or don't mind discontinued ones?")
        self.attributes["product_size"] = Attribute("product_size", "Product Size", "categorical",
            [n for n, _, _ in config["weight_buckets"]], False, "What size product are you looking for?")

        if self.category == "all_beauty":
            self.attributes["item_form"] = Attribute("item_form", "Item Form", "categorical",
                config["top_item_forms"] + ["Others"], True, "What form would you prefer?")
        elif self.category == "amazon_fashion":
            self.attributes["department"] = Attribute("department", "Department", "categorical",
                config["top_departments"] + ["Others"], True, "Who is this product for?")
        elif self.category == "amazon_home":
            self.attributes["color"] = Attribute("color", "Color", "categorical",
                config["top_colors"] + ["Others"], True, "What color do you prefer?")
            self.attributes["material"] = Attribute("material", "Material", "categorical",
                config["top_materials"] + ["Others"], True, "What material do you prefer?")

    def get_askable_attributes(self):
        return self.askable_attrs

    def num_conversations(self):
        return len(self.conversations)

    def get_target_item(self, idx):
        if 0 <= idx < len(self.conversations):
            gt = self.conversations[idx].get("gt_items", [])
            valid = [g for g in gt if g in self.items]
            return valid[0] if valid else None
        return None

    def get_conversation_text(self, idx):
        if 0 <= idx < len(self.conversations):
            return get_lavic_conversation_text(self.conversations[idx])
        return ""

    def _get_candidate_cache_path(self):
        return os.path.join(CANDIDATE_CACHE_DIR, f"lavic_{self.category}_{self.split}_candidates.json")

    def load_conversation_candidates(self):
        path = self._get_candidate_cache_path()
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            return {int(k): v for k, v in data.items()}
        return None

    def retrieve_candidates_sbert(self, similarity_manager, top_k=300, force_recompute=False):
        if not force_recompute:
            cached = self.load_conversation_candidates()
            if cached is not None:
                return cached

        all_ids = list(self.items.keys())
        candidates = {}
        gt_miss = 0

        for idx in range(len(self.conversations)):
            text = self.get_conversation_text(idx)
            if not text:
                candidates[idx] = []
                continue
            gt = self.get_target_item(idx)
            retrieved = similarity_manager.retrieve_top_k(text, item_ids=all_ids, top_k=top_k)
            if gt and gt not in retrieved:
                retrieved[random.randint(0, len(retrieved) - 1)] = gt
                gt_miss += 1
            candidates[idx] = retrieved

        path = self._get_candidate_cache_path()
        with open(path, 'w') as f:
            json.dump(candidates, f, indent=2)
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
        return batch_get_item_texts(self.items, "lavic")
