import librosa
import numpy as np


def process_audio(file_path):

    y, sr = librosa.load(file_path, sr=16000)

    y, _ = librosa.effects.trim(y)
    y = librosa.util.normalize(y)

    # MFCC (13)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    # Spectral rolloff
    rolloff = np.mean(
        librosa.feature.spectral_rolloff(y=y, sr=sr)
    )

    # Energy
    energy = np.mean(y**2)

    # Combine into 15-feature vector
    feature_vector = np.append(mfcc_mean, [rolloff, energy])

    return feature_vector
