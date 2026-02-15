import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def visualize_hinglish(audio_path):
    # Load audio at 16kHz (Standard for Whisper)
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Generate Mel-Spectrogram
    # Why 80? Because that's the input dimension for Whisper's Encoder.
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram: Analyzing Phonetic Collisions')
    plt.tight_layout()
    plt.show()

# Usage:
visualize_hinglish("data/raw/sample.wav")