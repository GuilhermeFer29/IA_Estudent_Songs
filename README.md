Audio Dataset Visualization and Export

This Python project demonstrates how to work with an audio dataset from the Hugging Face datasets library. It includes features for visualizing audio data, extracting metadata, and saving audio files locally.

Features

Audio Dataset Loading: Loads an audio dataset from the Hugging Face library.

Waveform Visualization: Plots the waveform of audio samples with their respective categories.

Audio Export: Exports audio samples to .wav format and organizes them in a structured directory.

Audio Resampling: Resamples audio data to a specified sampling rate.

Prerequisites

Make sure you have the following Python packages installed:

datasets

matplotlib

soundfile

IPython

pathlib

You can install these packages using pip:

pip install datasets matplotlib soundfile ipython

How to Run

Clone this repository or copy the script to your local machine.

Run the Python script:

python script_name.py

Replace script_name.py with the name of your Python file.

Script Overview

The script includes the following steps:

Import the Required Libraries:

from datasets import load_dataset
import matplotlib.pyplot as plt
from pathlib import Path
import soundfile as sf
import IPython
from datasets import Audio

Load the Dataset:

The dataset is loaded using the Hugging Face datasets library:

dataset = load_dataset("ashraq/esc50")
dados = dataset["train"]

Visualize Audio Waveforms:

A waveform of an audio sample is plotted along with its category:

idx_dados = 1
linha = dados[idx_dados]

plt.subplots(figsize=(30, 5))
plt.plot(linha["audio"]["array"])
plt.suptitle(linha["category"])
plt.show()

Export Audio Files:

Audio samples are exported to a structured directory:

pasta_saida = Path("audios") / 'Objetos'
pasta_saida.mkdir(exist_ok=True, parents=True)

for i, linha in enumerate(primeiras_linhas):
    objeto = linha["category"]
    dados_som = linha ["audio"]["array"]
    taxa_amostragem = linha ["audio"]["sampling_rate"]
    caminho_saida = pasta_saida / f"{i}_{objeto}.wav"
    sf.write(file=caminho_saida, data=dados_som, samplerate=taxa_amostragem)

Resample Audio:

Audio data is resampled to a specified sampling rate:

dados = dados.cast_column("audio", Audio(sampling_rate=48000))

Output

Waveform Plots: Visualized in a matplotlib figure.

Audio Files: Exported to the audios/Objetos directory with the format {index}_{category}.wav.

Notes

The script uses the ashraq/esc50 dataset as an example. You can replace it with another dataset available on the Hugging Face platform.

Modify the sampling rate as needed in the resampling step.

License

This project is licensed under the MIT License. Feel free to use and adapt it as needed!
