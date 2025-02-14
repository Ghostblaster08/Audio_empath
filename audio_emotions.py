import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
from datasets import load_dataset
import librosa

# Load the dataset
dataset = load_dataset("stapesai/ssi-speech-emotion-recognition")

# Load the pre-trained Wav2Vec 2.0 model and processor for emotion recognition
model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

# Function to predict emotion from audio
def predict_emotion(audio_file):
    # Load audio
    speech, sr = librosa.load(audio_file, sr=16000)
    
    # Process audio
    inputs = processor(speech, sampling_rate=sr, return_tensors="pt", padding=True)
    
    # Perform inference
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # Get predicted emotion
    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_emotion = model.config.id2label[predicted_ids.item()]
    
    return predicted_emotion

# Example usage
audio_file = dataset['train'][0]['audio']['path']
emotion = predict_emotion(audio_file)
print(f"Predicted emotion: {emotion}")