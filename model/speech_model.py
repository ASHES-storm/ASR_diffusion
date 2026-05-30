import torch.nn as nn

from .audio_encoder import AudioEncoder
from .ctc_head import CTCHead


class SpeechModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.encoder = AudioEncoder()
        self.ctc_head = CTCHead(
            hidden_dim=512,   # transformer output dim
            vocab_size=vocab_size
        )

    def forward(self, waveform):
        """
        waveform: (B, samples)
        """
        enc_out = self.encoder(waveform)   # (B, T, 512)
        log_probs = self.ctc_head(enc_out)

        return log_probs