## whisper cpp installation
```
conda create -n whisper_env python=3.10
conda activate whisper_env
conda install cmake
conda install ffmpeg
git clone https://github.com/ggerganov/whisper.cpp
brew install gcc
conda install make
cd whisper.cpp/
sh ./models/download-ggml-model.sh base.en
```

## run voice assistant
1. follow steps for whisper_cpp
2. run poetry install
3. some packages required conda 
    ```
    conda install -c conda-forge pynini
    pip install nemo_text_processing
    pip install cython
    pip install youtokentome
    ```
4. Run `ollam serve` first.
5. Run `streamlit run run_voice_assistant.py`