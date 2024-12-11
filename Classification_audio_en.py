from datasets import load_dataset
from transformers import pipeline


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
    print(predicao)
 


    