from math import inf
import os
import random
import argparse
import numpy as np
import pandas as pd
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.types import Device
import tqdm
from typing import Sized

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import librosa

SR = 24000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512

MAX_SECONDS = 6.0
MAX_SAMPLES = int(SR * MAX_SECONDS)

BATCH_SIZE = 256
EPOCHS = 256
LR = 1e-4

SEED = 321

AUDIO_DIR = 'nEMO-main/samples/'
TSV_PATH = 'nEMO-main/data.tsv'

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pad_or_trunc(audio, max_len):
    if len(audio) > max_len:
        return audio[:max_len]
    else:
        return np.pad(audio, (0, max_len - len(audio)))


class NemoDataset(Dataset):
    def __init__(self, tsv_path: str, audio_dir: str):
        self.df = pd.read_csv(tsv_path, sep="\t")
        self.audio_dir = audio_dir

        self.label_encoder = LabelEncoder()
        self.df["label"] = self.label_encoder.fit_transform(self.df["emotion"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        wav_path = os.path.join(self.audio_dir, row["file_id"])

        audio, _ = librosa.load(wav_path, sr=SR)
        audio = pad_or_trunc(audio, MAX_SAMPLES)

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=SR,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )

        mel = librosa.power_to_db(mel, ref=np.max)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(row["label"], dtype=torch.long)

        return mel, label


class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.squeeze(-1).squeeze(-1)
        return self.classifier(x)

class ResNetInspired(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(

            # Initial conv
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Block 1
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Block 3
            nn.Conv2d(128,256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            ## Output
            #nn.AdaptiveAvgPool2d((1,1)),
            ##nn.Flat,
            #nn.Linear(256,512), # this layer causes an error if enabled with (256,256) or (256,512) or (512,512), regardless of layers ahead or before it
            #nn.Dropout(0.5),
            #nn.Softmax2d(),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256,512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        #x = nn.ReLU(inplace=True)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: Optimizer, scheduler: LRScheduler, criterion, device: Device):
    model.train()
    total_loss = 0

    batches = 0

    with torch.no_grad():
        lr = np.mean(scheduler.get_last_lr())


    with tqdm.tqdm(loader, unit=" batch") as bar:
        for x, y in bar:
            batches += 1
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            bar.set_postfix({
                'total loss:': total_loss / batches,
                'lr:': lr            
            })

        scheduler.step()
        return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, device: Device):
    model.eval()
    correct = 0
    total = 0

    with tqdm.tqdm(loader, unit=" batch") as bar:
        for x, y in bar:
            x = x.to(device)
            y = y.to(device)

            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            bar.set_postfix({'accuracy:': correct / total})

        return correct / total


def main():
    set_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    dataset = NemoDataset(TSV_PATH, AUDIO_DIR)

    train_idx, test_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=0.2,
        stratify=dataset.df["label"],
        random_state=SEED
    )

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    assert isinstance(dataset.label_encoder.classes_, Sized)
    #model = Model(num_classes=len(dataset.label_encoder.classes_)).to(device)
    model = ResNetInspired(num_classes=len(dataset.label_encoder.classes_)).to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    #scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.01, total_iters=EPOCHS)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_test_acc = -inf

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        test_acc = eval_epoch(model, test_loader, device)

        print(f'Epoch: {epoch}')
        print(f'Train loss: {train_loss}')
        print(f'Test accuracy: {test_acc}')

        if test_acc > best_test_acc:
            best_test_acc = test_acc

            torch.save({
                "model": model.state_dict(),
                "labels": dataset.label_encoder.classes_
            }, "best.pt")

            print("This is a new best epoch - saving model to best.pt")


if __name__ == "__main__":
    main()

