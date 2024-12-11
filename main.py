from datasets import load_dataset
import matplotlib.pyplot as plt
from pathlib import Path
import soundfile as sf
import IPython 
from datasets import Audio


# Importação do datasent de audio do huggingface
dataset = load_dataset("ashraq/esc50")

dados = dataset["train"]

primeira_linha = dados[0]   
primeiras_linhas = dados.select(range(10))

# Captura das  informações e gera o gráfico

idx_dados = 1
linha = dados[idx_dados]

plt.subplots(figsize=(30, 5))
plt.plot(linha["audio"]["array"])
plt.suptitle(linha["category"])

plt.show() 

# Captura de audio e escreve em uma pasta 
pasta_saida = Path("audios")/ 'Objetos'
pasta_saida.mkdir(exist_ok=True, parents=True)

# loop para capturar todos os audios
for i , linha in enumerate(primeiras_linhas):
    objeto = linha["category"]
    dados_som = linha ["audio"]["array"]
    taxa_amostragem = linha ["audio"]["sampling_rate"]
    caminho_saida = pasta_saida / f"{i}_{objeto}.wav"
    #sf.write(file=caminho_saida,data=dados_som, samplerate=taxa_amostragem)
    IPython.display.Audio(data=dados_som,rate=taxa_amostragem)

# Funçao da captura de audios

sr = dados[0]["audio"]["sampling_rate"]
dados = dados.cast_column("audio", Audio(sampling_rate=48000))