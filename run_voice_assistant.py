import streamlit as st
import numpy as np
import sounddevice as sd
import wave
import subprocess
import os
import nemo.collections.tts as nemo_tts
import torch
import torchaudio
from io import BytesIO
import re
import sys

sys.path.append("/Users/yucheng.tsai/Documents/voice-assistant")
from call_ollama import run_ollama_command


# Path to the compiled whisper.cpp binary
WHISPER_BINARY_PATH = "/Users/yucheng.tsai/Documents/whisper.cpp/main"
MODEL_PATH = "/Users/yucheng.tsai/Documents/whisper.cpp/models/ggml-base.en.bin"

# Streamlit app
st.title("Live Voice Assistant")
st.write(
    "Press the Start button to record audio and then transcribe it using Whisper.cpp."
)

# Select language for transcription
language = st.selectbox(
    "Select Language for Transcription", ["English", "Traditional Chinese"]
)
language_code = "en" if language == "English" else "zh"

# Audio recording parameters
duration = st.slider("Recording Duration (seconds)", min_value=1, max_value=30, value=5)
sampling_rate = 16000  # set sample rate to 16 kHz for compatibility with whisper.cpp

# Start recording button
if st.button("Start Recording"):
    st.write("Recording...")

    # Record audio using sounddevice
    recorded_audio = sd.rec(
        int(duration * sampling_rate),
        samplerate=sampling_rate,
        channels=1,
        dtype=np.int16,
    )
    sd.wait()  # Wait until recording is finished

    # Save audio to WAV file
    audio_file = "/Users/yucheng.tsai/Documents/voice-assistant/recorded_audio.wav"
    with wave.open(audio_file, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(sampling_rate)
        wf.writeframes(recorded_audio.tobytes())

    st.write("Recording finished. Audio saved as 'recorded_audio.wav'.")

    # Transcribe the audio using whisper.cpp
    st.write("Transcribing audio using Whisper.cpp...")

    try:
        result = subprocess.run(
            [
                WHISPER_BINARY_PATH,
                "-m",
                MODEL_PATH,
                "-f",
                audio_file,
                "-l",
                language_code,
                "-otxt",
            ],
            capture_output=True,
            text=True,
        )
        # Display the transcription
        transcription = result.stdout.strip()
        st.text_area("Transcription", transcription)
    except FileNotFoundError:
        st.error(
            "Whisper.cpp binary not found. Make sure the path to the binary is correct."
        )
    except Exception as e:
        st.error(f"An error occurred: {e}")

    # Parse the transcription text
    match = re.search(r"\] *(.*)", transcription)

    # Extract and print the result
    # Use regular expression to extract the spoken text after the timestamp
    # This regex matches anything after "]", which is the start of the actual transcription
    if match:
        extracted_text = match.group(1)

    # Call ollama to get an answer
    prompt = f"""
    Given this question: "{extracted_text}, please answer it in less than 15 words."
    """

    answer = run_ollama_command(model="llama2", prompt=prompt)

    # Integrate NVIDIA NeMo TTS to read the answer from ollama
    if answer:
        st.write("Generating speech from the answer from ollama using NVIDIA NeMo...")

        try:
            # Load the FastPitch and HiFi-GAN models from NeMo
            fastpitch_model = nemo_tts.models.FastPitchModel.from_pretrained(
                model_name="tts_en_fastpitch"
            )
            hifigan_model = nemo_tts.models.HifiGanModel.from_pretrained(
                model_name="tts_en_lj_hifigan_ft_mixerttsx"
            )

            # Set the FastPitch model to evaluation mode
            fastpitch_model.eval()
            parsed_text = fastpitch_model.parse(answer)
            spectrogram = fastpitch_model.generate_spectrogram(tokens=parsed_text)

            # Convert the spectrogram into an audio waveform using HiFi-GAN vocoder
            hifigan_model.eval()
            audio = hifigan_model.convert_spectrogram_to_audio(spec=spectrogram)

            # Save the audio to a byte stream
            audio_buffer = BytesIO()
            torchaudio.save(audio_buffer, audio.cpu(), sample_rate=22050, format="wav")
            audio_buffer.seek(0)

            # Play the generated audio using Streamlit's audio player
            st.audio(audio_buffer, format="audio/wav")
            st.success("Speech synthesis complete!")

        except Exception as e:
            st.error(f"An error occurred during speech synthesis: {e}")

# Clean up the recorded audio file
if os.path.exists("/Users/yucheng.tsai/Documents/recorded_audio.wav"):
    os.remove("recorded_audio.wav")
