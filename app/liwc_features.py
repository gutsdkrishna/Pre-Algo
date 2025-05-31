import numpy as np

class LIWCExtractor:
    """
    Stub for extracting LIWC psycholinguistic counts (90-dimensional).
    """
    def __init__(self, dict_path: str = None):
        # TODO: load LIWC dictionary from dict_path or default
        self.dim = 90

    def extract(self, text: str) -> np.ndarray:
        """
        Compute LIWC features for a single utterance.
        Currently returns a zero vector stub.
        """
        # TODO: use pyliwc to compute real counts
        return np.zeros(self.dim, dtype=float)
