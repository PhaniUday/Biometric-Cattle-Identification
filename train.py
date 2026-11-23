import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import CattleMuzzleNet
from data_loader import CattleTripletDataset


def evaluate(model, loader, criterion, device):

    model.eval()

    total_loss = 0

    with torch.no_grad():

        for anchor, positive, negative in loader:

            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)

            loss = criterion(

                anchor_emb,
                positive_emb,
                negative_emb
            )

            total_loss += loss.item()

    return total_loss / len(loader)


def train():

    device = torch.device(

        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(

        "\nRunning on:",

        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    )

    DATA_PATH = os.path.join(

        "Cattle Muzzle - DB",
        "Original"
    )

    BATCH_SIZE = 8
    LR = 5e-5
    EPOCHS = 30
    PATIENCE = 10

    train_dataset = CattleTripletDataset(
        DATA_PATH,
        split='train'
    )

    val_dataset = CattleTripletDataset(
        DATA_PATH,
        split='val'
    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=2,

        pin_memory=True
    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=2,

        pin_memory=True
    )

    model = CattleMuzzleNet().to(device)

    criterion = nn.TripletMarginLoss(
        margin=0.5
    )

    optimizer = optim.Adam(

        model.parameters(),

        lr=LR,

        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.StepLR(

        optimizer,

        step_size=20,

        gamma=0.5
    )

    best_val = float('inf')

    patience_counter = 0

    for epoch in range(EPOCHS):

        model.train()

        total_train = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}"
        )

        for anchor, positive, negative in progress:

            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()

            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)

            loss = criterion(

                anchor_emb,
                positive_emb,
                negative_emb
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(

                model.parameters(),

                max_norm=1.0
            )

            optimizer.step()

            total_train += loss.item()

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        scheduler.step()

        train_loss = (
            total_train /
            len(train_loader)
        )

        val_loss = evaluate(

            model,

            val_loader,

            criterion,

            device
        )

        print(

            f"\nEpoch {epoch+1}"

            f" | Train={train_loss:.4f}"

            f" | Val={val_loss:.4f}"
        )

        if val_loss < best_val:

            best_val = val_loss

            torch.save(

                model.state_dict(),

                "triplet_best_model.pth"
            )

            print(
                "Saved best model"
            )

            patience_counter = 0

        else:

            patience_counter += 1

        if patience_counter >= PATIENCE:

            print(
                "Early stopping"
            )

            break


if __name__ == "__main__":
    train()