import torch, torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np

class HRANEncoder(nn.Module):
    def __init__(self, vocab_size: int = 50257, d_model: int = 256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.word_gru = nn.GRU(d_model, d_model, batch_first=True)
        self.utt_attn = nn.Linear(d_model, 1)
        self.sent_gru = nn.GRU(d_model, d_model, batch_first=True)
        self.dialog_attn = nn.Linear(d_model, 1)

    def forward(self, dialogues):
        """
        dialogues: list[list[LongTensor]]  – tokenised utterances
        Returns single 256-dim embedding.
        """
        # Encode each utterance
        utt_embeddings = []
        for utt in dialogues:
            emb = self.embed(utt)
            out, _ = self.word_gru(emb.unsqueeze(0))
            weights = torch.softmax(self.utt_attn(out).squeeze(-1), dim=-1)
            utt_vec = (weights.unsqueeze(-1) * out).sum(1)
            utt_embeddings.append(utt_vec)

        # Pad & pack utterances
        packed = pad_sequence(utt_embeddings, batch_first=True)
        sent_out, _ = self.sent_gru(packed)
        d_weights = torch.softmax(self.dialog_attn(sent_out).squeeze(-1), dim=-1)
        dialog_vec = (d_weights.unsqueeze(-1) * sent_out).sum(1)
        return dialog_vec.squeeze(0)

class HierarchicalEncoder:
    """
    Stub for hierarchical encoder (HRAN) producing a 256-dimensional context vector.
    """
    def __init__(self, model_path: str = None):
        # TODO: load encoder model (e.g., PyTorch Lightning or ONNX)
        self.dim = 256

    def encode(self, texts, emotion_feats, liwc_feats):
        """
        Encode a sequence of utterances into a context vector.
        Currently returns a zero vector stub.
        """
        # texts: List[str], emotion_feats: List[np.ndarray], liwc_feats: List[np.ndarray]
        # TODO: implement real hierarchical encoding
        return np.zeros(self.dim, dtype=float)
