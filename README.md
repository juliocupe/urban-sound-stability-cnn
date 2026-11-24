# When Machines Listen

Urban sound classification with a small CNN, plus a test of how steady its predictions stay over time. I did this during my 2026 ASPIRE internship at the Johns Hopkins Applied Physics Laboratory.

<img width="799" height="599" alt="postermachineslisten" src="https://github.com/user-attachments/assets/97e49493-dedb-43ac-8295-203bc958d5aa" />

## Why I built it

Sound classifiers are usually scored one clip at a time, like each clip has nothing to do with the one before it. Real audio isn't like that. It's continuous and sounds overlap, especially on a city street. If a classifier is going to be part of an assistive device for someone who's blind or hard of hearing, it can't just be right on average, it also has to stay steady. A siren alert that flickers on and off isn't useful.

So on top of normal accuracy I added a second test: slide a window across each clip and check how often the model changes its mind.

## What I did vs. what I built on

- **Base classifier (adapted).** `data_preprocessing.py`, `envsoundclasses.py` and `evaluate_classifier.py` are adapted from [mariostrbac/environmental-sound-classification](https://github.com/mariostrbac/environmental-sound-classification) (MIT license). I moved it from notebooks into scripts and saved per-clip predictions so I could analyze them afterward.
- **Error analysis (mine).** `analyze_predictions.py` builds the confusion matrix and per-class accuracy, and saves misclassified clips as .wav files so I could actually listen to what the model got wrong.
- **Temporal stability (mine).** `temporal_stability.py` runs the trained model over overlapping windows of each clip and measures label switching, dominant-class consistency and prediction entropy.
- I used Claude (an AI assistant) to help write and debug parts of the code.

## How it works

1. Each audio clip (first 2.95 s) is turned into a 128 x 128 log-mel spectrogram. Shorter clips are padded with silence.
2. Training spectrograms get a random time shift and Gaussian noise. Every spectrogram is normalized with the training set's mean and std.
3. A small CNN classifies them: 3 conv layers (24, 36, 48 filters), global average pooling, a 60-unit dense layer with dropout, and a 10-class output. Adam optimizer, cross-entropy loss, 100 epochs.
4. The model trains on 9 of the dataset's 10 folds and is tested on fold 1.
5. For stability, a 64-frame window (about 1.5 s) slides across each test spectrogram in 8-frame steps (about 0.19 s), so every clip gets 9 predictions.

The whole model is 33,634 parameters, about 134 KB as 32-bit floats, which is tiny for an audio model. The weights aren't the hard part for a small device though: the first conv layer's output alone is about 1.5 MB of 32-bit numbers (24 x 124 x 124), more memory than most microcontrollers have.

Dataset: [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) (8,732 labeled clips, 10 classes). It isn't included here, you have to download it yourself.

## Results

Tested on fold 1 (873 clips): **72.9% overall accuracy**, 75.4% when you average the 10 per-class accuracies.

| Class | Accuracy |
|---|---|
| gun shot | 97.1% |
| siren | 93.0% |
| children playing | 93.0% |
| car horn | 83.3% |
| dog bark | 83.0% |
| jackhammer | 70.8% |
| drilling | 65.0% |
| street music | 62.0% |
| engine idling | 59.4% |
| air conditioner | 47.0% |

The most common mistakes were between sounds that are similar: jackhammer called drilling (27 clips), engine idling called jackhammer (23), and air conditioner called engine idling (21).

### Temporal stability

![Temporal stability by class](stability_by_class.png)

Sirens were the steadiest: on average 87% of a siren clip's windows agreed with each other, and the label switched between neighboring windows only 8% of the time. Air conditioner and drilling had the lowest consistency, and air conditioner switched the most (20%) and was also the least accurate class.

Steady doesn't mean right though. Engine idling was fairly steady (0.83 consistency) but only 53% of its windows were correct, so the model was often confidently and consistently wrong on it, mostly calling it jackhammer.

### A bug that changed the numbers

The original code called dropout without telling it whether the model was training, so dropout stayed on during testing too. That means every test prediction had random noise in it. The poster's numbers (69.6% accuracy, and its stability chart) were made with that bug. After fixing it, the same trained model scores 72.9%, and the average label switching dropped from 0.29 to 0.15, so about half of the instability I measured before was the bug and not the model.

The results above come from the model trained in my original run, re-tested with the fix. I haven't retrained from scratch with the cleaned-up code.

## Running it

The full pipeline was last run end to end (on a small sample dataset, to check that it works) with Python 3.13, PyTorch 2.15, torchvision 0.29, NumPy 2.5, pandas 3.1, librosa 1.0, scikit-learn 1.9, SciPy 1.18 and Matplotlib 3.11.

```
pip install -r requirements.txt
```

1. Download UrbanSound8K and put it at `data/UrbanSound8K/` (so you have `data/UrbanSound8K/audio/fold1/` and `data/UrbanSound8K/metadata/UrbanSound8K.csv`).
2. `python data_preprocessing.py` makes the spectrograms and saves `us8k_df_up.pkl`.
3. `python evaluate_classifier.py` trains the CNN and saves `model.pt`, `prediction_results.pkl` and the training curves in `outputs/`.
4. `python analyze_predictions.py` prints the confusion analysis and saves a few misclassified clips in `error_samples/`.
5. `python temporal_stability.py` runs the sliding-window test and saves CSVs and charts in `stability_outputs/`.

Training takes a while on a laptop CPU. The generated files (.pkl, .pt, outputs) aren't in the repo.

## What I learned and what's next

Accuracy by itself hid a lot. The model can get a clip right overall and still flip between labels inside it, and for an assistive device that flipping is what the user would notice. Sirens came out the most stable and drilling and air conditioner the least. Stable also isn't the same as correct, engine idling was steady and still wrong almost half the time. I also learned to check the code behind my own numbers, since the dropout bug was changing my results the whole time and I didn't catch it until I went back through everything.

There are real limits though. All of this is from one test fold, not all 10, so the numbers could move on a different split. The dataset is short isolated clips of 4 seconds or less, so 9 windows in under 3 seconds isn't the same as listening to a real street for minutes. Next I'd want to run all 10 folds, test on longer recordings where sounds actually overlap, and see how much smaller the model can get (like 8-bit weights and a smaller input) and still work, since a wearable can't spare much memory.

## Credits

- J. Salamon, C. Jacoby and J. P. Bello, "A Dataset and Taxonomy for Urban Sound Research," ACM Multimedia, 2014.
- M. Strbac, [environmental-sound-classification](https://github.com/mariostrbac/environmental-sound-classification), MIT license.

License: MIT (see `LICENSE`).
