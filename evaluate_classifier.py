# Train the CNN on 9 folds of UrbanSound8K and test it on the held-out fold.
# Run data_preprocessing.py first.
#
# Saves:
#   model.pt                  trained weights (used by temporal_stability.py)
#   prediction_results.pkl    per-clip predictions on the test fold (used by analyze_predictions.py)
#   training_history.pkl      loss and accuracy for every epoch
#   outputs/training_curves.png
#
# Adapted from mariostrbac/environmental-sound-classification (MIT license).

import pickle

import pandas as pd
import torch

from envsoundclasses import process_fold, show_results

TEST_FOLD = 1
EPOCHS = 100

us8k_df = pd.read_pickle("us8k_df_up.pkl")

history, model, predictions, truths, fnames = process_fold(
    TEST_FOLD,
    us8k_df,
    epochs=EPOCHS,
    num_of_workers=4,
)

torch.save(model.state_dict(), "model.pt")

with open("prediction_results.pkl", "wb") as fp:
    pickle.dump({
        "test_fold": TEST_FOLD,
        "predictions": predictions,
        "truths": truths,
        "fnames": fnames,
    }, fp)

with open("training_history.pkl", "wb") as fp:
    pickle.dump(history, fp)

show_results(history)
print("saved model.pt, prediction_results.pkl, training_history.pkl")
