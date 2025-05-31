import time, json, random, os
from typing import List, Dict

def timer():
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    return int((end - start) * 1000)

def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def read_conversation(conv: List[Dict[str, str]]) -> str:
    """Concatenate speaker-tagged utterances into a single string."""
    return "\n".join(f"{ut['speaker']}: {ut['text']}" for ut in conv)
