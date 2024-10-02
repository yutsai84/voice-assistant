import streamlit as st
import numpy as np
import sounddevice as sd
import wave
import subprocess
import os


# Path to the compiled whisper.cpp binary
WHISPER_BINARY_PATH = "/Users/yucheng.tsai/Documents/whisper.cpp/main"
MODEL_PATH = "/Users/yucheng.tsai/Documents/whisper.cpp/models/ggml-base.en.bin"

# Streamlit app
st.title("Live Audio Transcription using Whisper.cpp")
st.write("Press the Start button to record audio and then transcribe it using Whisper.cpp.")

# Audio recording parameters
duration = st.slider("Recording Duration (seconds)", min_value=1, max_value=30, value=5)
sampling_rate = 16000  # set sample rate to 16 kHz for compatibility with whisper.cpp

# Start recording button
if st.button("Start Recording"):
    st.write("Recording...")

    # Record audio using sounddevice
    recorded_audio = sd.rec(int(duration * sampling_rate), samplerate=sampling_rate, channels=1, dtype=np.int16)
    sd.wait()  # Wait until recording is finished

    # Save audio to WAV file
    audio_file = "/Users/yucheng.tsai/Documents/whisper-live-transcription/recorded_audio.wav"
    with wave.open(audio_file, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(sampling_rate)
        wf.writeframes(recorded_audio.tobytes())

    st.write("Recording finished. Audio saved as 'recorded_audio.wav'.")

    # Transcribe the audio using whisper.cpp
    st.write("Transcribing audio using Whisper.cpp...")

    try:
        result = subprocess.run(
            [WHISPER_BINARY_PATH, "-m", MODEL_PATH, "-f", audio_file, "-l", "zh"],
            capture_output=True,
            text=True
        )
        # Display the transcription
        transcription = result.stdout
        st.text_area("Transcription", transcription)
    except FileNotFoundError:
        st.error("Whisper.cpp binary not found. Make sure the path to the binary is correct.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Clean up the recorded audio file
if os.path.exists("/Users/yucheng.tsai/Documents/recorded_audio.wav"):
    os.remove("recorded_audio.wav")

