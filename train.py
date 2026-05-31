import os
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
# LOSS
# -------------------------
ctc_loss_fn = nn.CTCLoss(
    blank=BLANK_IDX,
    zero_infinity=True
)


# -------------------------
# CHECKPOINT DIR
# -------------------------
CHECKPOINT_DIR = "/content/drive/MyDrive/ASR_project/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# -------------------------
# SAVE FUNCTION
# -------------------------
def save_checkpoint(model, optimizer, epoch, batch_idx, best_loss, tag):
    path = os.path.join(
        CHECKPOINT_DIR,
        f"{tag}_epoch{epoch}_step{batch_idx}.pt"
    )

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "batch_idx": batch_idx,
        "best_loss": best_loss
    }, path)

    print(f"💾 Saved checkpoint: {path}")


# -------------------------
# TRAIN
# -------------------------
def train(overfit=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = LJSpeechDataset(
        "/content/drive/MyDrive/Python_ML_project/Dataset/LJSpeech-1.1"
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )

    model = SpeechModel(vocab_size=VOCAB_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.train()

    print("🚀 Starting Training...\n")

    # ============================================================
    # 🧪 OVERFIT MODE
    # ============================================================
    if overfit:

        print("🔥 OVERFIT MODE ENABLED\n")

        decoder = CTCDecoder()
        batch = next(iter(loader))

        waveforms = batch["waveforms"].to(device)
        tokens = batch["tokens"].to(device)
        token_lengths = batch["token_lengths"].to(device)
        transcripts = batch["transcripts"]

        for step in range(200):

            optimizer.zero_grad()

            log_probs = model(waveforms)
            T = log_probs.size(0)

            input_lengths = torch.full(
                (waveforms.size(0),),
                T,
                dtype=torch.long,
                device=device
            )

            targets = torch.cat([
                tokens[i, :token_lengths[i]]
                for i in range(tokens.size(0))
            ]).to(device)

            loss = ctc_loss_fn(
                log_probs,
                targets,
                input_lengths,
                token_lengths
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            if step % 10 == 0:
                model.eval()
                with torch.no_grad():
                    pred_logits = model(waveforms)
                    pred_text = decoder.decode(pred_logits)

                print(f"\n[OVERFIT] Step {step} | Loss: {loss.item():.4f}")
                print(f"PRED: {pred_text[0]}")
                print(f"TRUE: {transcripts[0]}")

                model.train()

        save_checkpoint(model, optimizer, 0, 200, 0, "overfit_final")
        print("\n💾 Overfit model saved\n")
        return


    # ============================================================
    # 🚀 NORMAL TRAINING
    # ============================================================

    best_loss = float("inf")
    patience = 3
    no_improve_epochs = 0

    for epoch in range(30):   # ✔ UPDATED TO 30

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

            targets = torch.cat([
                tokens[i, :token_lengths[i]]
                for i in range(tokens.size(0))
            ]).to(device)

            loss = ctc_loss_fn(
                log_probs,
                targets,
                input_lengths,
                token_lengths
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")

            if batch_idx % 200 == 0:
                save_checkpoint(model, optimizer, epoch, batch_idx, best_loss, "step")

        avg_loss = total_loss / len(loader)

        print(f"\n🔥 Epoch {epoch} completed | Avg Loss: {avg_loss:.4f}\n")

        # -------------------------
        # BEST MODEL SAVE
        # -------------------------
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve_epochs = 0

            save_checkpoint(model, optimizer, epoch, -1, best_loss, "best")
            print("🏆 New best model saved!\n")

        else:
            no_improve_epochs += 1

        # -------------------------
        # EARLY STOPPING
        # -------------------------
        if no_improve_epochs >= patience:
            print("\n⛔ Early stopping triggered\n")
            break

        # -------------------------
        # EPOCH CHECKPOINT
        # -------------------------
        save_checkpoint(model, optimizer, epoch, -1, best_loss, "epoch")


# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    train(overfit=False)