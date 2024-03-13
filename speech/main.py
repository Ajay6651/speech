import speech_recognition as sr
import pyaudio
from vosk import Model, KaldiRecognizer
import wave
import numpy as np
import noisereduce as nr
import time

# Initialize speech recognition
recognizer = sr.Recognizer()

# Define the microphone parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 4
RATE = 16000

# Load the noise reduction model
def denoise_audio(audio_data):
    noise_sample = audio_data[:RATE * 2]  # Take a 2-second sample for noise profile
    noise_sample = np.mean(noise_sample)  # Compute the mean without specifying axis
    audio_data = nr.reduce_noise(audio_clip=audio_data, noise_clip=noise_sample, verbose=False)
    return audio_data

# Initialize Vosk speech recognition model
model = Model("C:/Users/HP/Downloads/vosk-model-small-en-us-0.15/vosk-model-small-en-us-0.15")

# Create a recognizer with the model
recognizer = KaldiRecognizer(model, RATE)

# Speech transcription loop
while True:
    print("Listening... Say something!")
    with sr.Microphone(device_index=0, sample_rate=RATE, chunk_size=CHUNK) as source:
        recognizer = sr.Recognizer()
        recognizer.adjust_for_ambient_noise(source)
        print("Speak now!")
        audio = recognizer.listen(source, phrase_time_limit=5)
    
    # Convert audio data to numpy array
    audio_data = np.frombuffer(audio.frame_data, dtype=np.int16)
    
    # Perform noise reduction
    audio_data = denoise_audio(audio_data)
    
    # Process the audio data for speech recognition
    recognizer.AcceptWaveform(audio_data)
    result = recognizer.Result()
    
    # Print the recognized text
    if result:
        print("Recognized text:", result["text"])
