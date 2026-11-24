# Temporal stability: does the model keep giving the same answer as it
# listens to a clip over time, or does it flip between labels?
#
# For every clip in the test fold, a window slides across the spectrogram and
# the model predicts a class for each window. Then for each clip:
#   label switching frequency   how often the prediction changes between neighboring windows (0 = never)
#   dominant-class consistency  share of windows that agree with the most common prediction (1 = all agree)
#   mean prediction entropy     how unsure the model is on average (0 = fully confident)
#
# Run evaluate_classifier.py first (it saves model.pt).

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import entropy

from envsoundclasses import CLASS_MAP, N_CLASSES, Net, get_device, normalize_data

DATA_PATH = "us8k_df_up.pkl"
MODEL_PATH = "model.pt"
OUTPUT_DIR = "stability_outputs"

TEST_FOLD = 1        # must match evaluate_classifier.py
WINDOW_FRAMES = 64   # about 1.5 s of audio (each frame is 512 samples at 22050 Hz)
HOP_FRAMES = 8       # about 0.19 s between windows, so 9 windows per 128-frame clip


# load data and model
if not os.path.exists(DATA_PATH):
    sys.exit(f"Could not find {DATA_PATH}. Run data_preprocessing.py first.")
if not os.path.exists(MODEL_PATH):
    sys.exit(f"Could not find {MODEL_PATH}. Run evaluate_classifier.py first.")

us8k_df = pd.read_pickle(DATA_PATH)

# normalize exactly the same way as in training
train_df, test_df = normalize_data(us8k_df[us8k_df["fold"] != TEST_FOLD],
                                   us8k_df[us8k_df["fold"] == TEST_FOLD])
print(f"test fold {TEST_FOLD}: {len(test_df)} clips")

device = get_device()
model = Net(device).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()  # turns dropout off

os.makedirs(OUTPUT_DIR, exist_ok=True)


def sliding_window_predictions(mel):
    """Predict a class for every window. Returns classes, probabilities and window starts."""
    n_frames = mel.shape[1]
    starts = list(range(0, n_frames - WINDOW_FRAMES + 1, HOP_FRAMES))

    # the CNN ends in global average pooling, so it can take windows narrower than 128 frames
    windows = np.stack([mel[:, s:s + WINDOW_FRAMES] for s in starts])
    x = torch.tensor(windows, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        probs = F.softmax(model(x), dim=1).cpu().numpy()

    return probs.argmax(axis=1), probs, starts


def label_switching_frequency(classes):
    if len(classes) < 2:
        return 0.0
    switches = sum(1 for a, b in zip(classes[:-1], classes[1:]) if a != b)
    return switches / (len(classes) - 1)


def dominant_class_consistency(classes):
    counts = np.bincount(classes, minlength=N_CLASSES)
    return counts.max() / len(classes)


def mean_prediction_entropy(probs):
    return float(np.mean([entropy(p + 1e-12) for p in probs]))


# run the analysis on every test clip
records = []
for i, row in test_df.iterrows():
    classes, probs, _ = sliding_window_predictions(row["melspectrogram"])
    records.append({
        "fname": row["fname"],
        "true_label": CLASS_MAP[int(row["label"])],
        "dominant_pred": CLASS_MAP[int(np.bincount(classes, minlength=N_CLASSES).argmax())],
        "label_switching_freq": label_switching_frequency(classes),
        "dominant_class_consistency": dominant_class_consistency(classes),
        "mean_prediction_entropy": mean_prediction_entropy(probs),
        "windows_correct": float(np.mean(classes == int(row["label"]))),
    })

stability_df = pd.DataFrame(records)
stability_df.to_csv(os.path.join(OUTPUT_DIR, "stability_results.csv"), index=False)

metrics = ["dominant_class_consistency", "label_switching_freq",
           "mean_prediction_entropy", "windows_correct"]
by_class = (stability_df.groupby("true_label")[metrics].mean()
            .sort_values("dominant_class_consistency", ascending=False))
by_class.to_csv(os.path.join(OUTPUT_DIR, "stability_by_class.csv"))


# plot: consistency and switching per class, steadiest class on top
plot_df = by_class.iloc[::-1]
labels = [name.replace("_", " ") for name in plot_df.index]

fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
axes[0].barh(labels, plot_df["dominant_class_consistency"], color="steelblue")
axes[0].set_xlim(0, 1)
axes[0].set_xlabel("dominant-class consistency (higher = steadier)")
axes[1].barh(labels, plot_df["label_switching_freq"], color="salmon")
axes[1].set_xlim(0, max(0.5, plot_df["label_switching_freq"].max() * 1.15))
axes[1].set_xlabel("label switching frequency (lower = steadier)")

for ax, col in zip(axes, ["dominant_class_consistency", "label_switching_freq"]):
    for y, value in enumerate(plot_df[col]):
        ax.text(value + 0.01, y, f"{value:.2f}", va="center", fontsize=8)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

n_windows = len(range(0, 128 - WINDOW_FRAMES + 1, HOP_FRAMES))
fig.suptitle(f"Temporal stability by class, test fold {TEST_FOLD} "
             f"({n_windows} windows of {WINDOW_FRAMES} frames per clip)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "stability_by_class.png"), dpi=150)
plt.close()


def plot_prediction_timeline(clip, out_path):
    """Window-by-window predictions for one clip."""
    mel = test_df.loc[test_df["fname"] == clip["fname"], "melspectrogram"].iloc[0]
    classes, probs, starts = sliding_window_predictions(mel)
    true_id = [k for k, v in CLASS_MAP.items() if v == clip["true_label"]][0]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={"height_ratios": [2, 1]})

    colors = ["green" if c == true_id else "red" for c in classes]
    axes[0].scatter(starts, classes, c=colors, s=60, zorder=3)
    axes[0].axhline(true_id, color="blue", linestyle="--", linewidth=1)
    axes[0].set_yticks(range(N_CLASSES))
    axes[0].set_yticklabels([CLASS_MAP[k] for k in range(N_CLASSES)], fontsize=7)
    axes[0].set_title(f"{os.path.basename(clip['fname'])} (true: {clip['true_label']})")

    cmap = plt.get_cmap("tab10")
    axes[1].stackplot(starts, probs.T, colors=[cmap(k) for k in range(N_CLASSES)], alpha=0.8)
    axes[1].set_xlabel("window start (frames)")
    axes[1].set_ylabel("probability")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


plot_prediction_timeline(stability_df.loc[stability_df["dominant_class_consistency"].idxmax()],
                         os.path.join(OUTPUT_DIR, "timeline_most_stable.png"))
plot_prediction_timeline(stability_df.loc[stability_df["dominant_class_consistency"].idxmin()],
                         os.path.join(OUTPUT_DIR, "timeline_most_unstable.png"))


# summary
print("\n" + "=" * 78)
print(f"{'class':<20}{'consistency':>13}{'switching':>12}{'entropy':>10}{'win correct':>14}")
print("=" * 78)
for label, r in by_class.iterrows():
    print(f"{label:<20}{r['dominant_class_consistency']:>13.3f}{r['label_switching_freq']:>12.3f}"
          f"{r['mean_prediction_entropy']:>10.3f}{r['windows_correct']:>14.3f}")
print("=" * 78)
print(f"\noutputs saved in {os.path.abspath(OUTPUT_DIR)}/")
