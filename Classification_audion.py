#libs importadas
from datasets import load_dataset
from transformers import pipeline
# Carregando o dataset
name_dataset = 'google/fleurs'

# Idioma do dataset
lingua_dataset = 'pt_br'

# Carregando Dados do dataset e convertendo para streaming
dados = load_dataset(name_dataset, name=lingua_dataset, split='train' , streaming=True)

primeiras_linhas = dados.take(5)

# Interação com os dados
for linha in primeiras_linhas:
    print(linha)
    
# Criando o Modelo de Classificação
model = 'sanchit-gandhi/whisper-medium-fleurs-lang-id'

classificador = pipeline('audio-classification', model=model)

classificador.feature_extractor.sampling_rate = 16000

primeiras_linhas = list(primeiras_linhas)
primeira_linha = primeiras_linhas[0]

# Predição do modelo
predicao = classificador(linha['audio']['array'])



