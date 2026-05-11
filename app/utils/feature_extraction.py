import librosa
import numpy as np

def extract_features(file_path):

    audio, sr = librosa.load(file_path, sr=22050)

    features = []

    # ============================================
    # MFCC
    # ============================================

    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=20
    )

    # ============================================
    # Spectral Features
    # ============================================

    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr
    )

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sr
    )

    zcr = librosa.feature.zero_crossing_rate(audio)

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sr
    )

    # ============================================
    # Statistical Aggregation
    # ============================================

    def add_statistics(feature):

        features.extend([
            np.mean(feature),
            np.std(feature),
            np.min(feature),
            np.max(feature)
        ])

    # MFCC statistics
    for mfcc in mfccs:
        add_statistics(mfcc)

    # Other features
    add_statistics(spectral_centroid)
    add_statistics(spectral_rolloff)
    add_statistics(zcr)

    # Chroma statistics
    for ch in chroma:
        add_statistics(ch)

    return np.array(features)