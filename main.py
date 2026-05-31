import torch
from torch.utils.data import DataLoader

from dataset import LJSpeechDataset
from collate_fn import collate_fn
from model.speech_model import SpeechModel
from model.decoder import CTCDecoder
from vocab import VOCAB_SIZE


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Load dataset
    # -------------------------
    dataset = LJSpeechDataset("data/LJSpeech-1.1")

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )

    # -------------------------
    # Take SAME TYPE of batch
    # -------------------------
    batch = next(iter(loader))

    waveforms = batch["waveforms"].to(device)
    transcripts = batch["transcripts"]
    tokens = batch["tokens"]

    print("\n===== BATCH INPUT =====")
    print(f"Waveforms shape: {waveforms.shape}")
    print(f"Tokens shape: {tokens.shape}")
    print(f"Sample transcript: {transcripts[0]}")

    # -------------------------
    # Load MODEL (IMPORTANT FIX)
    # -------------------------
    model = SpeechModel(VOCAB_SIZE).to(device)

    model.load_state_dict(
        torch.load("overfit_ctc_model.pt", map_location=device)
    )

    model.eval()

    print("\n✅ Loaded overfit model")

    # -------------------------
    # Forward pass
    # -------------------------
    with torch.no_grad():
        log_probs = model(waveforms)   # (T, B, V)

    print("\n===== MODEL OUTPUT =====")
    print(f"Output shape: {log_probs.shape}")

    # -------------------------
    # Decode
    # -------------------------
    decoder = CTCDecoder()

    predicted_text = decoder.decode(log_probs)  # NO PERMUTE FIX

    print("\n===== PREDICTION =====")

    min_len = min(len(predicted_text), len(transcripts))

    for i in range(min_len):
        print("\n--- SAMPLE ---")
        print("PRED:", predicted_text[i])
        print("TRUE:", transcripts[i])


if __name__ == "__main__":
    main()