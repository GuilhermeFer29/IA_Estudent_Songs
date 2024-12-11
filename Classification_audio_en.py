from datasets import load_dataset
from transformers import pipeline
import sounddevice as sd
import IPython

# Carregando o dataset
name_dataset = 'google/fleurs'

# Idioma do dataset
lingua_dataset = 'en_us'

# Carregando Dados do dataset e convertendo para streaming
dados = load_dataset(name_dataset, name=lingua_dataset, split='train' , streaming=True)

# Criando o Modelo de Classificação
model = 'sanchit-gandhi/whisper-medium-fleurs-lang-id'

classificador = pipeline('audio-classification', model=model)

for linha in dados.take(5):
    predicao = classificador(linha['audio']['array'])
    #print(predicao)
 
 # Captura de audio   
duracao = 10 # em segundos
taxa_amostragem = 16000 # amostras por segundo
tamanho_vetor = int(duracao * taxa_amostragem) # cantidade de amostras

# Gravacão de audio
gravacao = sd.rec(tamanho_vetor, samplerate=taxa_amostragem, channels=1)

sd.wait()

print(gravacao)

gravacao = gravacao.ravel()

print(gravacao.shape) # gravacao.shape

IPython.display.display(IPython.display.Audio(data=gravacao, rate=taxa_amostragem))

    