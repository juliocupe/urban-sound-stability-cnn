# Shared code for the urban sound classifier: dataset, data augmentation,
# the CNN, and the training helpers. The other scripts import from here.
#
# Adapted from mariostrbac/environmental-sound-classification (MIT license).

import copy
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # save plots to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# UrbanSound8K class IDs
CLASS_MAP = {
    0: "air_conditioner",
    1: "car_horn",
    2: "children_playing",
    3: "dog_bark",
    4: "drilling",
    5: "engine_idling",
    6: "gun_shot",
    7: "jackhammer",
    8: "siren",
    9: "street_music",
}
N_CLASSES = len(CLASS_MAP)


class UrbanSound8kDataset(Dataset):
    """Wraps a dataframe with columns melspectrogram, label, fold, fname."""

    def __init__(self, us8k_df, transform=None):
        assert isinstance(us8k_df, pd.DataFrame)
        self.us8k_df = us8k_df
        self.transform = transform

    def __len__(self):
        return len(self.us8k_df)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        row = self.us8k_df.iloc[index]
        spectrogram = row["melspectrogram"]

        if self.transform is not None:
            spectrogram = self.transform(spectrogram)

        return {"spectrogram": spectrogram, "label": row["label"], "fname": row["fname"]}


class MyRightShift(object):
    """Shift the spectrogram to the right in time (data augmentation)."""

    def __init__(self, input_size, width_shift_range, shift_probability=1.0):
        assert isinstance(input_size, (int, tuple))
        assert isinstance(width_shift_range, (int, float))
        assert isinstance(shift_probability, float)

        if isinstance(input_size, int):
            self.input_size = (input_size, input_size)
        else:
            assert len(input_size) == 2
            self.input_size = input_size

        if isinstance(width_shift_range, int):
            assert 0 < width_shift_range <= self.input_size[1]
            self.width_shift_range = width_shift_range
        else:
            assert 0.0 < width_shift_range <= 1.0
            self.width_shift_range = int(width_shift_range * self.input_size[1])

        assert 0.0 < shift_probability <= 1.0
        self.shift_prob = shift_probability

    def __call__(self, image):
        if np.random.random() > self.shift_prob:
            return image

        # new array filled with the quietest value, then paste the shifted image in
        shifted_image = np.full(self.input_size, np.min(image), dtype="float32")
        rand_position = np.random.randint(1, self.width_shift_range)
        shifted_image[:, rand_position:] = copy.deepcopy(image[:, :-rand_position])

        return shifted_image


class MyAddGaussNoise(object):
    """Add Gaussian noise to the spectrogram (data augmentation)."""

    def __init__(self, input_size, mean=0.0, std=None, add_noise_probability=1.0):
        assert isinstance(input_size, (int, tuple))
        assert isinstance(mean, (int, float))
        assert isinstance(std, (int, float)) or std is None
        assert isinstance(add_noise_probability, float)

        if isinstance(input_size, int):
            self.input_size = (input_size, input_size)
        else:
            assert len(input_size) == 2
            self.input_size = input_size

        if std is not None:
            assert std > 0.0
        self.mean = mean
        self.std = std

        assert 0.0 < add_noise_probability <= 1.0
        self.add_noise_prob = add_noise_probability

    def __call__(self, spectrogram):
        if np.random.random() > self.add_noise_prob:
            return spectrogram

        if self.std is None:
            std = np.abs(np.min(spectrogram) * 0.03)
        else:
            std = self.std

        noise = np.random.normal(self.mean, std, size=self.input_size).astype("float32")
        return spectrogram + noise


class MyReshape(object):
    """Reshape the spectrogram, e.g. (128, 128) to (1, 128, 128) for the CNN."""

    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, image):
        return image.reshape(self.output_size)


# augmentation is only used on the training set
train_transforms = transforms.Compose([
    MyRightShift(input_size=128, width_shift_range=13, shift_probability=0.9),
    MyAddGaussNoise(input_size=128, add_noise_probability=0.55),
    MyReshape(output_size=(1, 128, 128)),
])
test_transforms = transforms.Compose([MyReshape(output_size=(1, 128, 128))])


class Net(nn.Module):
    """3 conv layers, global average pooling, 1 dense layer, 10 outputs."""

    def __init__(self, device):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=24, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=4, padding=0)
        self.conv3 = nn.Conv2d(in_channels=36, out_channels=48, kernel_size=3, padding=0)

        self.fc1 = nn.Linear(in_features=48, out_features=60)
        self.fc2 = nn.Linear(in_features=60, out_features=10)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001, eps=1e-07, weight_decay=1e-3)

        self.device = device

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=(3, 3), stride=3))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=(2, 2), stride=2))
        x = F.relu(self.conv3(x))

        # global average pooling, so the model also works on narrower windows
        x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        x = x.view(-1, 48)

        x = F.relu(self.fc1(x))
        # training=self.training turns dropout off when the model is in eval mode.
        # Without it, dropout stays on during testing and predictions get random.
        x = F.dropout(x, p=0.5, training=self.training)

        return self.fc2(x)

    def fit(self, train_loader, epochs, val_loader=None):
        history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

        for epoch in range(epochs):
            self.train()
            print("\nEpoch {}/{}".format(epoch + 1, epochs))

            with tqdm(total=len(train_loader), file=sys.stdout) as pbar:
                for batch in train_loader:
                    X_batch = batch["spectrogram"].to(self.device)
                    y_batch = batch["label"].to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.forward(X_batch)
                    batch_loss = self.criterion(outputs, y_batch)
                    batch_loss.backward()
                    self.optimizer.step()

                    pbar.update(1)

            train_loss, train_acc, _, _, _ = self.evaluate(train_loader)
            print("loss: %.4f - accuracy: %.4f" % (train_loss, train_acc), end="")

            val_loss, val_acc = None, None
            if val_loader is not None:
                val_loss, val_acc, _, _, _ = self.evaluate(val_loader)
                print(" - val_loss: %.4f - val_accuracy: %.4f" % (val_loss, val_acc))

            history["loss"].append(train_loss)
            history["accuracy"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)

        return history

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            return self.forward(X)

    def evaluate(self, data_loader):
        """Returns loss, accuracy and the per-clip predictions, truths and filenames."""
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        predictions_out = []
        truths_out = []
        fnames_out = []

        for batch in data_loader:
            X_batch = batch["spectrogram"].to(self.device)
            y_batch = batch["label"].to(self.device)

            outputs = self.predict(X_batch)
            predictions = torch.argmax(outputs, dim=1)

            # weight by the real batch size, since the last batch can be smaller
            n = len(y_batch)
            total_loss += self.criterion(outputs, y_batch).item() * n
            total_correct += (predictions == y_batch).sum().item()
            total_count += n

            predictions_out.extend(predictions.cpu().tolist())
            truths_out.extend(y_batch.cpu().tolist())
            fnames_out.extend(batch["fname"])

        return (total_loss / total_count, total_correct / total_count,
                predictions_out, truths_out, fnames_out)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def init_model():
    device = get_device()
    return Net(device).to(device)


def normalize_data(train_df, test_df):
    """Normalize every pixel with the training set's mean and std."""
    train_stack = np.stack(train_df["melspectrogram"])
    mean = train_stack.mean(axis=0)
    std = train_stack.std(axis=0)

    def make_df(df):
        # float32 because the model's weights are float32
        spectrograms = ((np.stack(df["melspectrogram"]) - mean) / std).astype(np.float32)
        return pd.DataFrame({
            "melspectrogram": list(spectrograms),
            "label": df["label"].to_numpy(),
            "fold": df["fold"].to_numpy(),
            "fname": df["fname"].to_numpy(),
        })

    return make_df(train_df), make_df(test_df)


def process_fold(fold_k, dataset_df, epochs=100, batch_size=32, num_of_workers=0):
    """Train on every fold except fold_k, then test on fold_k."""
    train_df = dataset_df[dataset_df["fold"] != fold_k]
    test_df = dataset_df[dataset_df["fold"] == fold_k]
    train_df, test_df = normalize_data(train_df, test_df)

    train_loader = DataLoader(UrbanSound8kDataset(train_df, transform=train_transforms),
                              batch_size=batch_size, shuffle=True,
                              num_workers=num_of_workers)
    test_loader = DataLoader(UrbanSound8kDataset(test_df, transform=test_transforms),
                             batch_size=batch_size, shuffle=False,
                             num_workers=num_of_workers)

    model = init_model()

    _, accuracy, _, _, _ = model.evaluate(test_loader)
    print("Pre-training accuracy: %.2f%%" % (100 * accuracy))

    start_time = datetime.now()
    history = model.fit(train_loader, epochs=epochs, val_loader=test_loader)
    print("\nTraining completed in time: {}".format(datetime.now() - start_time))

    loss, accuracy, predictions, truths, fnames = model.evaluate(test_loader)
    print("Post-training accuracy: %.2f%%" % (100 * accuracy))

    return history, model, predictions, truths, fnames


def show_results(history, out_path="outputs/training_curves.png"):
    """Save accuracy and loss curves for the train and test sets."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=(15, 5))

    plt.subplot(121)
    plt.plot(history["accuracy"])
    plt.plot(history["val_accuracy"])
    plt.grid(linestyle="--")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["train", "test"], loc="upper left")

    plt.subplot(122)
    plt.plot(history["loss"])
    plt.plot(history["val_loss"])
    plt.grid(linestyle="--")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["train", "test"], loc="upper left")

    plt.savefig(out_path)
    plt.close()

    print("Max test accuracy: %.2f%%" % (np.max(history["val_accuracy"]) * 100))
    print("Min test loss: %.5f" % np.min(history["val_loss"]))
