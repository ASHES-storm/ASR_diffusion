import torch
import torch.nn as nn
import torchaudio


# -------------------------
# 1. Mel Spectrogram
# -------------------------
class AudioPreprocessor(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=80):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=n_mels
        )

    def forward(self, waveform):
        """
        waveform: (B, T)
        returns: (B, 80, time)
        """
        mel = self.mel(waveform)
        mel = torch.log(mel + 1e-5)
        return mel


# -------------------------
# 2. CNN Frontend (stable)
# -------------------------
class ConvFrontend(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        """
        x: (B, 80, T)
        returns: (B, T', hidden_dim)
        """

        x = x.unsqueeze(1)  # (B, 1, 80, T)
        x = self.net(x)     # (B, C, F, T)

        B, C, F, T = x.shape

        # IMPORTANT FIX:
        # Instead of flattening F * C (unsafe),
        # we pool frequency dimension → stable ASR representation
        x = x.mean(dim=2)   # (B, C, T)

        x = x.permute(0, 2, 1)  # (B, T, C)

        return x


# -------------------------
# 3. Positional Encoding
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len).unsqueeze(1)

        div = torch.exp(
            torch.arange(0, dim, 2) * (-torch.log(torch.tensor(10000.0)) / dim)
        )

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# -------------------------
# 4. Transformer Encoder
# -------------------------
class AudioTransformer(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512, layers=4):
        super().__init__()

        self.proj = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, layers)
        self.pos = PositionalEncoding(hidden_dim)

    def forward(self, x):
        x = self.proj(x)
        x = self.pos(x)
        return self.encoder(x)


# -------------------------
# 5. Full Audio Encoder
# -------------------------
class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.prep = AudioPreprocessor()
        self.frontend = ConvFrontend()
        self.encoder = AudioTransformer(input_dim=256)

    def forward(self, waveform):
        """
        waveform: (B, T)
        returns: (B, T', 512)
        """

        mel = self.prep(waveform)
        feats = self.frontend(mel)
        enc_out = self.encoder(feats)

        # Debug prints (you can remove later)
        #print("Mel:", mel.shape)
        #print("Features:", feats.shape)
        #print("Encoder:", enc_out.shape)

        return enc_out

