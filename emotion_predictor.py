import torch
import librosa
import numpy as np
from datetime import datetime
import sounddevice as sd
import soundfile as sf
import os
from audio_emotions_mkII import EmotionCNN

# class EmotionCNN(torch.nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.conv1 = torch.nn.Conv2d(1, 32, 3, padding=1)
#         self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
#         self.pool = torch.nn.MaxPool2d(2, 2)
#         self.fc1 = torch.nn.Linear(64 * 16 * 3, 128)
#         self.fc2 = torch.nn.Linear(128, num_classes)
#         self.dropout = torch.nn.Dropout(0.5)
        
#     def forward(self, x):
#         x = x.unsqueeze(1)  # Add channel dimension
#         x = self.pool(torch.nn.functional.relu(self.conv1(x)))
#         x = self.pool(torch.nn.functional.relu(self.conv2(x)))
#         x = x.view(-1, 64 * 16 * 3)
#         x = torch.nn.functional.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x

class EmotionPredictor:
    def __init__(self, model_path='emotion_model.pth'):
        # Load the model and emotion labels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize model architecture (same as training)
        self.model = EmotionCNN(num_classes=8)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Emotion labels
        self.emotion_labels = ['ANG', 'CAL', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD', 'SUR']

    def preprocess_audio(self, audio_data, sr):
        # Ensure 3 seconds of audio
        target_length = 3 * sr
        if len(audio_data) > target_length:
            audio_data = audio_data[:target_length]
        else:
            audio_data = np.pad(audio_data, (0, target_length - len(audio_data)))
            
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(
            y=audio_data,
            sr=sr,
            n_mfcc=13,
            n_fft=1024,
            hop_length=256,
            win_length=1024
        )
        
        # Resize to fixed dimensions
        target_length = 64
        if mfccs.shape[1] > target_length:
            mfccs = mfccs[:, :target_length]
        else:
            pad_width = ((0, 0), (0, target_length - mfccs.shape[1]))
            mfccs = np.pad(mfccs, pad_width, mode='constant')
            
        # Normalize
        mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-8)
        
        return torch.FloatTensor(mfccs)

    def predict_file(self, audio_path):
        """Predict emotion from audio file"""
        # Load audio file
        audio_data, sr = librosa.load(audio_path, sr=16000)
        
        # Preprocess
        features = self.preprocess_audio(audio_data, sr)
        
        # Add batch dimension
        features = features.unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(features.to(self.device))
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
            
        emotion = self.emotion_labels[prediction.item()]
        confidence = probabilities[0][prediction].item()
        
        return emotion, confidence

    def record_and_predict(self, duration=3, sr=16000):
        """Record audio and predict emotion"""
        print("Recording...")
        audio_data = sd.rec(int(duration * sr), samplerate=sr, channels=1)
        sd.wait()
        audio_data = audio_data.flatten()
        
        # Save recording with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.wav"
        sf.write(filename, audio_data, sr)
        
        # Predict
        features = self.preprocess_audio(audio_data, sr)
        features = features.unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.model(features.to(self.device))
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
            
        emotion = self.emotion_labels[prediction.item()]
        confidence = probabilities[0][prediction].item()
        
        return emotion, confidence, filename

def main():
    # Create a directory for recordings if it doesn't exist
    os.makedirs("recordings", exist_ok=True)
    
    # Initialize predictor
    predictor = EmotionPredictor('best_emotion_model.pth')
    
    while True:
        print("\nEmotion Recognition Menu:")
        print("1. Predict from audio file")
        print("2. Record and predict")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == '1':
            audio_path = input("Enter the path to audio file: ")
            try:
                emotion, confidence = predictor.predict_file(audio_path)
                print(f"\nPredicted emotion: {emotion}")
                print(f"Confidence: {confidence:.2%}")
            except Exception as e:
                print(f"Error: {str(e)}")
                
        elif choice == '2':
            try:
                emotion, confidence, filename = predictor.record_and_predict()
                print(f"\nRecording saved as: {filename}")
                print(f"Predicted emotion: {emotion}")
                print(f"Confidence: {confidence:.2%}")
            except Exception as e:
                print(f"Error: {str(e)}")
                
        elif choice == '3':
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()