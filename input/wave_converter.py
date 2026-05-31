import wave
import numpy as np
import torch
import torchaudio

TARGET_SR = 16000


def load_wave(audio_path):

    with wave.open(str(audio_path), "rb") as f:
        n_channels = f.getnchannels()
        sample_rate = f.getframerate()
        n_frames = f.getnframes()

        audio = f.readframes(n_frames)

    # Convert bytes -> numpy
    waveform = np.frombuffer(
        audio,
        dtype=np.int16
    ).astype(np.float32)

    # Normalize to [-1, 1]
    waveform /= 32768.0

    # Stereo -> mono
    if n_channels > 1:
        waveform = waveform.reshape(
            -1,
            n_channels
        ).mean(axis=1)

    # NumPy -> Torch
    waveform = torch.from_numpy(
        waveform
    ).unsqueeze(0)

    # Resample to target sample rate
    if sample_rate != TARGET_SR:

        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=TARGET_SR
        )

        waveform = resampler(waveform)

        sample_rate = TARGET_SR

    return waveform, sample_rate