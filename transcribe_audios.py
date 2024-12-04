from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import soundfile as sf
import os

# Inicializar el modelo y el procesador de Wav2Vec 2.0
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53-spanish")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53-spanish")

# Carpeta con los audios
audio_dir = "audios/"
output = []

# Iterar sobre los archivos de audio
for filename in os.listdir(audio_dir):
    if filename.endswith(".wav"):  # Asegúrate de que los audios sean .wav
        audio_path = os.path.join(audio_dir, filename)
        
        # Cargar el audio
        audio_input, sr = sf.read(audio_path)

        # Procesar el audio para el modelo
        input_values = processor(audio_input, return_tensors="pt", sampling_rate=sr).input_values

        # Obtener la transcripción
        with torch.no_grad():
            logits = model(input_values).logits

        # Decodificar la salida a texto
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]

        # Guardar resultados
        output.append({"audio_filename": filename, "transcription": transcription})
        print(f"Transcripción de {filename}: {transcription}")

# Crear un DataFrame con las transcripciones
import pandas as pd
transcriptions = pd.DataFrame(output)
transcriptions.to_csv("transcriptions.csv", index=False)
