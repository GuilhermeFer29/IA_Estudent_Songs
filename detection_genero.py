from transformers import pipeline
import IPython
from datasets import load_dataset
from transformers import pipeline
from IPython.display import display
# Carregando o dataset
name_dataset = 'google/fleurs'

# Idioma do dataset
lingua_dataset = 'pt_br'

# Carregando Dados do dataset e convertendo para streaming
dados = load_dataset(name_dataset, name=lingua_dataset, split='train' , streaming=True)


# Modelo de Classificação
model = 'audeering/wav2vec2-large-robust-24-ft-age-gender'
classification = pipeline('audio-classification', model=model)

for linha in dados.take(5):
    audio = linha['audio']
    prediction = classification(audio.copy())
    print(prediction)    
    display(IPython.display.Audio(data=audio['array'], rate=audio['sampling_rate']) )