from transformers import pipeline
import time
import IPython


# Modelo
model = 'facebook/mms-tts-por'
reader = pipeline('text-to-speech', model=model)

text = 'Olá, meu nome é Guilherme e estou aprendendo Python'

speaks = reader(text)

# temporizador de execução
start = time.time()
speaks = reader(text)
end = time.time()
print(f'levou {end - start} segundos para gerar o audio')

# Display audio
IPython.display.Audio(data= speaks['audio'], rate= speaks['sampling_rate'])