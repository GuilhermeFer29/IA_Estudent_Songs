from transformers import pipeline
import time
import IPython
import torch

# Verificar se GPU esta disponivel
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Modelo
model = 'suno/bark-small'
reader = pipeline('text-to-speech', model=model,
model_kwargs={'torch_dtype': torch.float16},          forward_params={'max_new_tokens': 50})

# Move o modelo para a GPU 
reader.model = reader.model.to(device)
reader.model = reader.model.to_bettertransformer()
reader.model.enable_cpu_offload()

# Texto para falar
text = 'Olá, meu nome é Guilherme e estou aprendendo Python'
speaks = reader(text)

# temporizador de execução
start = time.time()
speaks = reader(text)
end = time.time()
print(f'levou {end - start} segundos para gerar o audio')

# Display audio
IPython.display.Audio(data= speaks['audio'], rate= speaks['sampling_rate'])

