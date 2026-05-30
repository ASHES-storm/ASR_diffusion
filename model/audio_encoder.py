import torch
import torchaudio
import torch.nn as nn

""" Mel Spectrogram"""
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
        return: (B, 80, time)
        """
        mel = self.mel(waveform)
        mel = torch.log(mel + 1e-5)
        return mel
#CNN to get phonemes,sharp sounds
class ConvFrontend(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        """
        x: (B, 80, T)
        return: (B, T', hidden_dim * freq)
        """
        x = x.unsqueeze(1)        # (B, 1, 80, T)
        x = self.net(x)
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * F)
        return x
#Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
#self attention-Transformer
class AudioTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, layers=4):
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
#Audio-Encoder-Calling Block
class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.prep = AudioPreprocessor()
        self.frontend = ConvFrontend()
        self.encoder = AudioTransformer(input_dim=256 * 20)

    def forward(self, waveform):
        mel = self.prep(waveform)
        feats = self.frontend(mel)
        enc_out = self.encoder(feats)
        print("Mel:", mel.shape)
        print("Features:", feats.shape)
        print("Encoder:", enc_out.shape)
        return enc_out
""" Testing Code 
if __name__ == "__main__":
    model = AudioEncoder()

    # Fake audio: 1 sec @ 16kHz
    dummy_wave = torch.randn(2, 16000)

    out = model(dummy_wave)
    print("Output shape:", out.shape)
    print(out[0, :, :5])"""



