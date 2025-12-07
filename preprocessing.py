import librosa
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import librosa.display
import time
from typing import Literal

TARGET_SR = 16000  # 16 kHz divisiones del audio
FIXED_DURATION = 2  # 2 segundos longitud de audio max
FIXED_SAMPLES = TARGET_SR * FIXED_DURATION  # 32000 muestras

N_MELS = 128        # N de bandas mel
N_FFT = 1024        # tamaño ventana FFT
HOP_LENGTH = 256    # desplazamiento entre frames

def load_audio(path):
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    return y, sr

def fix_length(y, fixed_samples=FIXED_SAMPLES):
    if len(y) > fixed_samples:
        y = y[:fixed_samples]

    else:
        y = np.pad(y, (0, fixed_samples - len(y)), mode='constant', constant_values=0)

    return y

def apply_gaussian_noise(y):
    blur = np.random.randn(len(y))
    y_noisy = y + (blur * 0.01)

    y_noisy = np.clip(y_noisy, -1.0, 1.0)
    y_noisy = y_noisy.astype(np.float32)

    return y_noisy

def audio_to_mel(y, sr=TARGET_SR):
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )

    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db

def save_mel(S, file, save_path, type: Literal['original', 'noisy']):
    if type == 'original':
        base = os.path.basename(file).replace(".wav", ".npy")

    elif type == 'noisy':
        base = os.path.basename(file).replace(".wav", "_noisy.npy")

    out = os.path.join(save_path, base)

    np.save(out, S)

    print("Guardado:", out, "shape:", S.shape)
    return

def plot_mel_spectrogram(S_db, sr=16000, hop_length=256, title="Mel spectrogram"):
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel"
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()
    plt.show()


input_path = "data/MESD dataset"
output_path = "data/MEL_augmentation"

y_list = []

for wav_file in  glob(os.path.join(input_path, "*.wav")):
    print(wav_file)
    
    y, sr = load_audio(f"{wav_file}")
    y = fix_length(y)
    y_list.append(y)

    # Gaussian noise augmentation

    y_blur = apply_gaussian_noise(y)
    y_list.append(y_blur)

    for idx, y_var in enumerate(y_list):
        S_db = audio_to_mel(y_var)
        #plot_mel_spectrogram(S_db)

        if idx == 0:
            save_mel(S_db, wav_file, output_path, type='original')

        else:
            save_mel(S_db, wav_file, output_path, type='noisy')

    y_list = []