from pathlib import Path
import wave
import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent

def load_wave(filename="audio.wav"):
    path = THIS_DIR / filename

    with wave.open(str(path), "rb") as f:
        n_channels = f.getnchannels()
        sample_rate = f.getframerate()
        n_frames = f.getnframes()
        audio = f.readframes(n_frames)

    # Convert bytes → numpy
    waveform = np.frombuffer(audio, dtype=np.int16).astype(np.float32)

    # Normalize to [-1, 1]
    waveform /= 32768.0

    # Stereo → mono
    if n_channels > 1:
        waveform = waveform.reshape(-1, n_channels).mean(axis=1)

    # Torch tensor (1, T)
    waveform = torch.from_numpy(waveform).unsqueeze(0)

    return waveform, sample_rate
