import os
import csv
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

class FacetPredictor:
    """
    Approximate facet predictor via semantic similarity between conversation and facet names.
    Uses a SentenceTransformer to embed facet definitions and conversation text.
    """
    def __init__(self, embed_model_name: str = 'sentence-transformers/all-mpnet-base-v2'):
        # Load embedding model
        self.model = SentenceTransformer(embed_model_name)
        # Load facet names from CSV
        csv_path = os.path.join(os.path.dirname(__file__), '143000.csv')
        facet_names = []
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                facet_names.append(row['facets'])
        self.facet_names = facet_names
        # Compute facet embeddings
        self.facet_embeddings = self.model.encode(facet_names, convert_to_tensor=True)
        self.n_facets = len(facet_names)

    def predict(self, texts):
        """
        Predict similarity scores between concatenated texts and each facet.
        Returns a numpy array of shape (n_facets,) with values in [0,1].
        texts: List[str]
        """
        # Concatenate all utterances
        context = ' '.join(texts)
        # Embed context
        context_emb = self.model.encode(context, convert_to_tensor=True)
        # Compute cosine similarity
        sims = torch.nn.functional.cosine_similarity(
            context_emb.unsqueeze(0), self.facet_embeddings, dim=-1
        )
        # Map from [-1,1] to [0,1]
        scores = (sims.cpu().numpy() + 1) / 2
        return scores

    def top_k(self, raw_scores, k: int = 30):
        """
        Select top-k facets by score.
        """
        idx = np.argsort(raw_scores)[::-1][:k]
        scores = raw_scores[idx]
        return list(zip(idx.tolist(), scores.tolist())) 