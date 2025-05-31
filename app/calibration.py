from sklearn.linear_model import LogisticRegression  # :contentReference[oaicite:9]{index=9}
import numpy as np, joblib, pathlib

CALIB_PATH = pathlib.Path(__file__).parent / "platt.joblib"

def load_platt():
    if CALIB_PATH.exists():
        return joblib.load(CALIB_PATH)
    return LogisticRegression()

def calibrate(raw, llm, clf):
    fused = 0.7 * raw + 0.3 * llm
    return clf.predict_proba(fused.reshape(-1, 1))[:, 1]

class Calibrator:
    """
    Calibrator for scaling facet scores to [0,1] via min-max normalization.
    """
    def __init__(self, model_path: str = None):
        # No external model needed for normalization
        pass

    def calibrate(self, facet_scores):
        """
        Normalize scores to [0,1] range based on observed min and max.
        facet_scores: list of (facet_id, score)
        Returns list of (facet_id, normalized_score)
        """
        if not facet_scores:
            return []
        # Extract only the score values
        scores = [score for (_, score) in facet_scores]
        min_s = min(scores)
        max_s = max(scores)
        span = max_s - min_s if max_s > min_s else 1.0
        # Scale each score to [0,1]
        normalized = [ (fid, (score - min_s) / span) for (fid, score) in facet_scores ]
        return normalized
