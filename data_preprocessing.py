# Turn every UrbanSound8K clip into a 128 x 128 log-mel spectrogram and save
# everything to us8k_df_up.pkl for the other scripts.
#
# Expects the dataset at data/UrbanSound8K/ (see the README).
# Adapted from mariostrbac/environmental-sound-classification (MIT license).

import os

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

US8K_AUDIO_PATH = os.path.abspath("data/UrbanSound8K/audio/")
US8K_METADATA_PATH = os.path.abspath("data/UrbanSound8K/metadata/UrbanSound8K.csv")

HOP_LENGTH = 512        # number of samples between successive frames
WINDOW_LENGTH = 512     # length of the window in samples
N_MEL = 128             # number of mel bands
N_FRAMES = 128          # every spectrogram is padded or cut to this many frames
SOUND_DURATION = 2.95   # seconds of audio to load from each clip


def compute_melspectrogram_with_fixed_length(audio, sampling_rate, num_of_frames=N_FRAMES):
    melspectrogram = librosa.feature.melspectrogram(y=audio,
                                                    sr=sampling_rate,
                                                    hop_length=HOP_LENGTH,
                                                    win_length=WINDOW_LENGTH,
                                                    n_mels=N_MEL)

    # power spectrogram to decibels (log-mel spectrogram)
    melspectrogram_db = librosa.power_to_db(melspectrogram, ref=np.max)

    # short clips get padded with -80 dB (silence)
    if melspectrogram_db.shape[1] != num_of_frames:
        melspectrogram_db = librosa.util.fix_length(melspectrogram_db,
                                                    size=num_of_frames,
                                                    axis=1,
                                                    constant_values=(0, -80.0))
    return melspectrogram_db.astype(np.float32)


us8k_metadata_df = pd.read_csv(US8K_METADATA_PATH,
                               usecols=["slice_file_name", "fold", "classID"],
                               dtype={"fold": "uint8", "classID": "uint8"})

features = []

for _, row in tqdm(us8k_metadata_df.iterrows(), total=len(us8k_metadata_df)):
    file_path = f'{US8K_AUDIO_PATH}/fold{row["fold"]}/{row["slice_file_name"]}'
    try:
        audio, sample_rate = librosa.load(file_path, duration=SOUND_DURATION, res_type="kaiser_fast")
        melspectrogram = compute_melspectrogram_with_fixed_length(audio, sample_rate)
    except Exception as error:
        print(f"\nskipping {file_path}: {error}")
        continue

    features.append([melspectrogram, row["classID"], row["fold"], file_path, audio])

us8k_df = pd.DataFrame(features, columns=["melspectrogram", "label", "fold", "fname", "audio"])
us8k_df.to_pickle("us8k_df_up.pkl")

print(f"data preprocessing is done: {len(us8k_df)} clips saved to us8k_df_up.pkl")
