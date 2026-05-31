import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    """
    Returns a CTC-ready batch:
        waveforms: (B, T)
        waveform_lengths: (B,)
        tokens: (B, L)
        token_lengths: (B,)
        transcripts: list[str]
    """

    waveforms = []
    waveform_lengths = []

    tokens = []
    token_lengths = []

    transcripts = []

    for sample in batch:

        # -------------------------
        # SAFE waveform handling
        # -------------------------
        waveform = sample["waveform"]

        # ensure shape is (T,)
        waveform = waveform.squeeze()

        if waveform.dim() != 1:
            raise ValueError(f"Invalid waveform shape after squeeze: {waveform.shape}")

        waveforms.append(waveform)
        waveform_lengths.append(waveform.size(0))

        # -------------------------
        # tokens
        # -------------------------
        token_ids = torch.tensor(sample["tokens"], dtype=torch.long)

        tokens.append(token_ids)
        token_lengths.append(token_ids.size(0))

        transcripts.append(sample["transcript"])

    # -------------------------
    # PAD WAVES
    # -------------------------
    padded_waveforms = pad_sequence(
        waveforms,
        batch_first=True,
        padding_value=0.0
    )

    # -------------------------
    # PAD TOKENS
    # -------------------------
    padded_tokens = pad_sequence(
        tokens,
        batch_first=True,
        padding_value=0  # CTC blank-safe padding (OK if 0 is NOT a real char)
    )

    return {
        "waveforms": padded_waveforms,  # (B, T)
        "waveform_lengths": torch.tensor(waveform_lengths, dtype=torch.long),

        "tokens": padded_tokens,        # (B, L)
        "token_lengths": torch.tensor(token_lengths, dtype=torch.long),

        "transcripts": transcripts
    }