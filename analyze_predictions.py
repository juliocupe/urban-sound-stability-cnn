# Look at what the model got wrong on the test fold: confusion matrix,
# most common mix-ups, accuracy per class, and a few misclassified clips
# saved as .wav files so you can listen to them.
# Run evaluate_classifier.py first.

import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.io.wavfile import write
from sklearn.metrics import confusion_matrix

from envsoundclasses import CLASS_MAP

RESULTS_PATH = "prediction_results.pkl"
DATA_PATH = "us8k_df_up.pkl"
OUTPUT_FOLDER = "error_samples"
EXPORT_LIMIT = 5
SAMPLE_RATE = 22050  # librosa.load default, used in data_preprocessing.py

try:
    with open(RESULTS_PATH, "rb") as fp:
        results = pickle.load(fp)
except FileNotFoundError:
    sys.exit(f"Error: could not find {RESULTS_PATH}. Run evaluate_classifier.py first.")

try:
    us8k_df = pd.read_pickle(DATA_PATH)
except FileNotFoundError:
    sys.exit(f"Error: could not find {DATA_PATH}. Run data_preprocessing.py first.")

preds = results["predictions"]
truths = results["truths"]
fnames = results["fnames"]
print(f"Test fold {results['test_fold']}: {len(preds)} clips")

print("\n--- Confusion matrix (rows = true class, columns = predicted) ---")
print(confusion_matrix(truths, preds, labels=list(CLASS_MAP)))

results_df = pd.DataFrame({
    "filename": fnames,
    "true_label": [CLASS_MAP[t] for t in truths],
    "pred_label": [CLASS_MAP[p] for p in preds],
    "correct": [p == t for p, t in zip(preds, truths)],
})

print("\nOverall accuracy: %.2f%%" % (100 * results_df["correct"].mean()))

mistakes = results_df[~results_df["correct"]]

if mistakes.empty:
    print("\nNo mistakes found")
else:
    confusion_counts = (mistakes.groupby(["true_label", "pred_label"]).size()
                        .sort_values(ascending=False))
    print("\n--- Most common mistakes (true class, predicted as) ---")
    print(confusion_counts.head(10))

class_accuracy = results_df.groupby("true_label")["correct"].mean().sort_values(ascending=False)
print("\n--- Accuracy per class ---")
print((class_accuracy * 100).round(2).astype(str) + "%")

# save a few misclassified clips so you can hear what the model got wrong
if not mistakes.empty:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    audio_by_fname = dict(zip(us8k_df["fname"], us8k_df["audio"]))

    print(f"\n--- Saving {min(len(mistakes), EXPORT_LIMIT)} misclassified clips to {OUTPUT_FOLDER}/ ---")
    for _, row in mistakes.head(EXPORT_LIMIT).iterrows():
        audio = audio_by_fname.get(row["filename"])
        if audio is None:
            print(f"Could not find audio for {row['filename']}")
            continue

        # librosa audio is float in [-1, 1]; wav files want 16-bit integers
        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

        base = os.path.splitext(os.path.basename(row["filename"]))[0]
        out_name = f"{base}_true_{row['true_label']}_pred_{row['pred_label']}.wav"
        write(os.path.join(OUTPUT_FOLDER, out_name), SAMPLE_RATE, audio_int16)
        print("  saved", out_name)
