import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model.decoder import CTCDecoder
from dataset import LJSpeechDataset
from collate_fn import collate_fn
from model.speech_model import SpeechModel
from vocab import VOCAB_SIZE, BLANK_IDX


# -------------------------
# CTC LOSS
# -------------------------
ctc_loss_fn = nn.CTCLoss(
    blank=BLANK_IDX,
    zero_infinity=True
)


# -------------------------
# TRAIN FUNCTION
# -------------------------
def train(overfit=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = LJSpeechDataset(
    "/content/drive/MyDrive/Python_ML_project/Dataset/LJSpeech-1.1"
)

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )

    model = SpeechModel(vocab_size=VOCAB_SIZE).to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print("🚀 Starting Training...\n")

    # ============================================================
    # 🧪 OVERFIT MODE
    # ============================================================
    if overfit:

        print("🔥 OVERFIT MODE ENABLED (1 batch only)\n")

        decoder = CTCDecoder()   # ✅ FIXED

        batch = next(iter(loader))

        waveforms = batch["waveforms"].to(device)
        tokens = batch["tokens"].to(device)
        token_lengths = batch["token_lengths"].to(device)
        transcripts = batch["transcripts"]

        for step in range(200):

            optimizer.zero_grad()

            # forward pass
            log_probs = model(waveforms)

            T = log_probs.size(0)

            input_lengths = torch.full(
                (waveforms.size(0),),
                T,
                dtype=torch.long,
                device=device
            )

            # build targets
            targets_list = []
            for i in range(tokens.size(0)):
                targets_list.append(tokens[i, :token_lengths[i]])

            targets = torch.cat(targets_list).to(device)

            # loss
            loss = ctc_loss_fn(
                log_probs,
                targets,
                input_lengths,
                token_lengths
            )

            loss.backward()
            optimizer.step()

            # ----------------------------------------------------
            # 🔍 LIVE DEBUG DECODING
            # ----------------------------------------------------
            if step % 10 == 0:

                model.eval()
                with torch.no_grad():

                    pred_logits = model(waveforms)
                    pred_text = decoder.decode(pred_logits)

                print(f"\n[OVERFIT] Step {step} | Loss: {loss.item():.4f}")
                print(f"PRED: {pred_text[0]}")
                print(f"TRUE: {transcripts[0]}")

                model.train()

        print("\n✅ OVERFIT COMPLETE\n")

        # ============================================================
        # 💾 SAVE MODEL
        # ============================================================
        torch.save(model.state_dict(), "overfit_ctc_model.pt")
        print("💾 Saved: overfit_ctc_model.pt")

        return

    # ============================================================
    # 🚀 NORMAL TRAINING MODE
    # ============================================================

    best_loss = float("inf")

    for epoch in range(10):

        total_loss = 0.0

        for batch_idx, batch in enumerate(loader):

            waveforms = batch["waveforms"].to(device)
            tokens = batch["tokens"].to(device)
            token_lengths = batch["token_lengths"].to(device)

            optimizer.zero_grad()

            log_probs = model(waveforms)

            T = log_probs.size(0)

            input_lengths = torch.full(
                (waveforms.size(0),),
                T,
                dtype=torch.long,
                device=device
            )

            targets_list = []
            for i in range(tokens.size(0)):
                targets_list.append(tokens[i, :token_lengths[i]])

            targets = torch.cat(targets_list).to(device)

            loss = ctc_loss_fn(
                log_probs,
                targets,
                input_lengths,
                token_lengths
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch} | Batch {batch_idx} | "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(loader)

        print(f"\n🔥 Epoch {epoch} completed | Avg Loss: {avg_loss:.4f}\n")

        # ============================================================
        # 💾 SAVE BEST MODEL
        # ============================================================
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_ctc_model.pt")
            print("🏆 New best model saved!\n")


# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    train(overfit=True)