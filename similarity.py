"""
SBERT similarity manager for belief initialization and candidate retrieval.

sim(·) is cosine similarity between SBERT embeddings of candidate text
and conversation/query text.
"""

import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

CACHE_DIR = "./embeddings"


class SimilarityManager:
    def __init__(self, model_name="all-MiniLM-L6-v2", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Loading sentence-BERT: {model_name}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.item_embeddings = {}  # item_id -> embedding vector
        os.makedirs(CACHE_DIR, exist_ok=True)

    def load_embeddings(self, cache_key):
        """Load pre-computed item embeddings from disk cache."""
        path = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
        if os.path.exists(path):
            print(f"Loading cached embeddings: {cache_key}")
            with open(path, 'rb') as f:
                self.item_embeddings = pickle.load(f)
            return True
        return False

    def compute_embeddings(self, item_texts, cache_key=None, batch_size=32):
        """Compute SBERT embeddings for all items and optionally cache to disk."""
        print(f"Computing embeddings for {len(item_texts)} items")
        ids = list(item_texts.keys())
        texts = [item_texts[i] for i in ids]
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
        self.item_embeddings = {i: e for i, e in zip(ids, embeddings)}
        if cache_key:
            path = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
            with open(path, 'wb') as f:
                pickle.dump(self.item_embeddings, f)

    def compute_similarity(self, query_text, candidate_ids, temperature=1.0):
        """Compute softmaxed cosine_similarity between query and candidates."""
        query_emb = self.model.encode(query_text, convert_to_numpy=True)
        cand_embs = np.array([self.item_embeddings[c] for c in candidate_ids])
        # normalize for cosine similarity
        q_norm = query_emb / np.linalg.norm(query_emb)
        c_norms = cand_embs / np.linalg.norm(cand_embs, axis=1, keepdims=True)
        # cosine similarity = dot product of normalized vectors
        sims = np.dot(c_norms, q_norm)
        # temperature-scaled softmax
        scaled = sims / temperature
        exp_s = np.exp(scaled - np.max(scaled))  # subtract max for numerical stability
        return exp_s / np.sum(exp_s)

    def retrieve_top_k(self, query_text, item_ids=None, top_k=300):
        """Retrieve top-k most similar items to query text."""
        if item_ids is None:
            item_ids = list(self.item_embeddings.keys())
        query_emb = self.model.encode(query_text, convert_to_numpy=True)
        cand_embs = np.array([self.item_embeddings[i] for i in item_ids])
        q_norm = query_emb / np.linalg.norm(query_emb)
        c_norms = cand_embs / np.linalg.norm(cand_embs, axis=1, keepdims=True)
        sims = np.dot(c_norms, q_norm)
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [item_ids[i] for i in top_idx]
