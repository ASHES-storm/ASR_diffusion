import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCHead(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        """
        x: (B, T, hidden_dim)

        returns:
            log_probs: (T, B, vocab_size)
        """

        logits = self.classifier(x)          # (B, T, V)

        log_probs = F.log_softmax(logits, dim=-1)

        # CTC format: (T, B, V)
        return log_probs.transpose(0, 1)