
import torch
import torch.nn as nn
class CTCHead(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        """
        x: (B, T, hidden_dim)
        return: (T, B, vocab_size) log_probs
        """
        logits = self.classifier(x)          # (B, T, V)
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs.permute(1, 0, 2)    # (T, B, V)