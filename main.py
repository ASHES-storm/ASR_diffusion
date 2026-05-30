import torch

from input.wave_converter import load_wave
from model.speech_model import SpeechModel
from model.decoder import CTCDecoder
from vocab import VOCAB_SIZE


def main():
    # Load audio
    waveform, sample_rate = load_wave()

    print(f"Sample Rate: {sample_rate}")
    print(f"Waveform Shape: {waveform.shape}")

    # Create model
    model = SpeechModel(VOCAB_SIZE)

    # Inference mode
    model.eval()

    # Disable gradients
    with torch.no_grad():
        log_probs = model(waveform)

    print(f"Model Output Shape: {log_probs.shape}")

    # Decode prediction
    decoder= CTCDecoder()
    decoder_input = log_probs.permute(1, 0, 2)
    text = decoder.decode(decoder_input)

    print("\nPredicted Text:")
    print(text)


if __name__ == "__main__":
    main()