import os
from dotenv import load_dotenv
load_dotenv()

import time
from flask import Flask, request, jsonify
import math

from preprocess import clean_text
from emotion import EmotionExtractor
from liwc_features import LIWCExtractor
from encoder import HierarchicalEncoder
from predictor import FacetPredictor
from calibration import Calibrator
from reranker import Reranker
from kg import apply_kg_bias

import csv
# Load facet names mapping
facet_map = {}
csv_path = os.path.join(os.path.dirname(__file__), "143000.csv")
with open(csv_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for idx, row in enumerate(reader):
        facet_map[idx] = row['facets']

app = Flask(__name__)

# Initialize components
emotion_extractor = EmotionExtractor()
liwc_extractor = LIWCExtractor()
encoder = HierarchicalEncoder()
predictor = FacetPredictor()
calibrator = Calibrator()
reranker = Reranker()

@app.route('/rate_v2', methods=['POST'])
def rate_v2():
    start_time = time.time()
    data = request.get_json(force=True)
    conversation = data.get('conversation', [])
    user_id = data.get('user_id')

    # Extract and preprocess texts
    texts = [turn.get('text', '') for turn in conversation]
    clean_texts = [clean_text(t) for t in texts]
    # Only consider the human's (User) utterances for facet scoring
    user_clean_texts = [ct for ct, ut in zip(clean_texts, conversation)
                        if ut.get('speaker', '').lower() == 'user']
    if not user_clean_texts:
        user_clean_texts = clean_texts

    # Utterance-level features
    emotion_feats = [emotion_extractor.extract(t) for t in clean_texts]
    liwc_feats = [liwc_extractor.extract(t) for t in clean_texts]

    # Context encoding (unused for embedding-based predictor)
    _ = encoder.encode(clean_texts, emotion_feats, liwc_feats)

    # Facet prediction via semantic similarity embeddings on user text
    raw_scores = predictor.predict(user_clean_texts)
    # Use all facets (0 through N-1)
    candidates = list(enumerate(raw_scores))

    # Optional KG bias
    candidates = apply_kg_bias(candidates, user_id=user_id)

    # Optional LLM reranking
    reranked = reranker.rerank(candidates, clean_texts)

    # Calibration
    calibrated = calibrator.calibrate(reranked)

    # Build response
    facet_scores = []
    for fid, score in calibrated:
        # Convert score [0-1] to 1-10 scale
        bucket = max(1, math.ceil(score * 10))
        facet_scores.append({
            'facet_id': fid,
            'facet_name': facet_map.get(fid, ''),
            'score': bucket
        })
    processing_ms = int((time.time() - start_time) * 1000)

    return jsonify({'facet_scores': facet_scores, 'processing_ms': processing_ms})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
