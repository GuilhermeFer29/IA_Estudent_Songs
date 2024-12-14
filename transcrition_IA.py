from datasets import load_dataset
import IPython
from IPython.display import display

# Importando pipeline
from transformers import pipeline

# Carregando o dataset
name_dataset = 'PolyAI/minds14'

# Idioma do dataset
language_dataset = 'pt-PT'

# Carregando Dados do dataset
dades = load_dataset(name_dataset, name=language_dataset, split='train[:10]' )

for line in dades:
    dades_song = line['audio']['array']
    sampling_rate = line['audio']['sampling_rate']
    
    #display(IPython.display.Audio(data=dades_song, rate=sampling_rate))

# Modelo de transcrição de audio


model = 'openai/whisper-medium'

# Realizando a transcrição

speech_recognizer = pipeline('automatic-speech-recognition', model=model)

# selecionando o audio
speech_recognizer(dades[0]['audio'])

print(speech_recognizer(dades[0]['audio']))
