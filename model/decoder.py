import torch
from vocab import idx2char, BLANK_IDX


class CTCDecoder:
    """
    Simple Greedy CTC Decoder

    Converts model output logits -> text.
    """

    def __init__(self, idx_to_char=idx2char, blank_idx=BLANK_IDX):
        self.idx_to_char = idx_to_char
        self.blank_idx = blank_idx

    def decode(self, logits):
        """
        Args:
            logits: Tensor of shape [B, T, C]
                    B = batch size
                    T = time steps
                    C = vocab size

        Returns:
            List[str]
        """

        pred_ids = torch.argmax(logits, dim=-1)

        decoded_texts = []

        for sequence in pred_ids:
            prev_token = None
            text = []

            for token in sequence:
                token = token.item()

                if token == self.blank_idx:
                    prev_token = token
                    continue

                if token == prev_token:
                    continue

                text.append(self.idx_to_char[token])
                prev_token = token

            decoded_texts.append("".join(text))

        return decoded_texts