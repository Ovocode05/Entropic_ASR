import sounddevice as sd
from scipy.io.wavfile import write
import os

def record_sample(filename="data/raw/collision_test.wav", duration=5):
    fs = 16000  # Whisper's native frequency
    print(f"Recording for {duration} seconds...")
    print("SAY: 'Bhaiya, do rupaye pay kardo'")
    
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    write(filename, fs, recording) 
    print(f"Saved to {filename}")

record_sample()