from pathlib import Path
from torch.utils.data import Dataset
from input.wave_converter import load_wave
from vocab import text_to_ids

class LJSpeechDataset(Dataset):

    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)

        self.wav_dir = self.root_dir / "wavs"
        self.metadata_path = self.root_dir / "metadata.csv"

        self.samples = []

        self.load_metadata()

    def load_metadata(self):

        with open(self.metadata_path, "r", encoding="utf-8") as f:

            for line in f:

                parts = line.strip().split("|")

                audio_id = parts[0]
                transcript = parts[1]
                transcript = transcript.lower()

                audio_path = self.wav_dir / f"{audio_id}.wav"

                self.samples.append(
                    (audio_path, transcript)
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, transcript = self.samples[idx]
        token_ids = text_to_ids(transcript)
        waveform, sample_rate = load_wave(audio_path)

        return {
            "waveform": waveform,
            "transcript": transcript,
            "tokens": token_ids,
            "sample_rate": sample_rate
        }
    