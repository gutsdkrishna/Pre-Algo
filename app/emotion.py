import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

class EmotionExtractor:
    """
    Utterance-level emotion logits extractor (28-dimensional) using RoBERTa-GoEmotions.
    """
    def __init__(self, model_name: str = "SamLowe/roberta-base-go_emotions"):
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def extract(self, text: str) -> np.ndarray:
        """
        Extract emotion logits for a single utterance.
        Returns a numpy array of shape (28,).
        """
        # Tokenize input text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        # Model inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.squeeze(0).cpu().numpy()
        return logits 