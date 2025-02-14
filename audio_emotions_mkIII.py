import torch
import librosa
import numpy as np
import sounddevice as sd
import queue  # Built-in Python module, no installation needed
import threading
import time
from datetime import datetime
import pygame# For visual feedback
from audio_emotions_mkII import EmotionCNN

class RealtimeEmotionAnalyzer:
    def __init__(self, model_path='Audio_empath/best_emotion_model.pth', buffer_duration=3, sample_rate=16000):
        # Initialize PyGame for visualization
        pygame.init()
        self.screen = pygame.display.set_mode((800, 400))
        pygame.display.set_caption('Real-time Emotion Analysis')
        
        # Audio parameters
        self.sample_rate = sample_rate
        self.buffer_duration = buffer_duration
        self.buffer_size = int(self.sample_rate * buffer_duration)
        
        # Initialize model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = EmotionCNN(num_classes=8)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Emotion labels and colors
        self.emotion_labels = ['ANG', 'CAL', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD', 'SUR']
        self.emotion_colors = {
            'ANG': (255, 0, 0),    # Red
            'CAL': (0, 255, 255),  # Cyan
            'DIS': (128, 0, 128),  # Purple
            'FEA': (128, 128, 0),  # Olive
            'HAP': (255, 255, 0),  # Yellow
            'NEU': (128, 128, 128),# Gray
            'SAD': (0, 0, 255),    # Blue
            'SUR': (255, 165, 0)   # Orange
        }
        
        # Initialize audio buffer and processing queue
        self.audio_buffer = np.zeros(self.buffer_size)
        self.audio_queue = queue.Queue()
        self.running = False
        
        # Statistics
        self.emotion_history = []
        self.confidence_history = []
        
    def audio_callback(self, indata, frames, time, status):
        """This is called for each audio block"""
        if status:
            print(status)
        self.audio_queue.put(indata.copy())
        
    def process_audio(self, audio_data):
        """Process audio data and return emotion prediction"""
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(
            y=audio_data.flatten(),
            sr=self.sample_rate,
            n_mfcc=13,
            n_fft=1024,
            hop_length=256,
            win_length=1024
        )
        
        # Resize and normalize
        target_length = 64
        if mfccs.shape[1] > target_length:
            mfccs = mfccs[:, :target_length]
        else:
            pad_width = ((0, 0), (0, target_length - mfccs.shape[1]))
            mfccs = np.pad(mfccs, pad_width, mode='constant')
            
        mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-8)
        
        # Prepare for model
        features = torch.FloatTensor(mfccs).unsqueeze(0)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(features.to(self.device))
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
            
        emotion = self.emotion_labels[prediction.item()]
        confidence = probabilities[0][prediction].item()
        
        return emotion, confidence
        
    def update_display(self, emotion, confidence):
        """Update the PyGame display"""
        self.screen.fill((0, 0, 0))  # Black background
        
        # Draw emotion bar
        bar_height = 50
        bar_width = int(confidence * 780)  # Scale to window width
        color = self.emotion_colors[emotion]
        pygame.draw.rect(self.screen, color, (10, 10, bar_width, bar_height))
        
        # Draw text
        font = pygame.font.Font(None, 36)
        text = f"{emotion}: {confidence:.2%}"
        text_surface = font.render(text, True, (255, 255, 255))
        self.screen.blit(text_surface, (10, 70))
        
        # Draw emotion history
        history_start_y = 150
        for i, (hist_emotion, hist_conf) in enumerate(zip(self.emotion_history[-10:], 
                                                        self.confidence_history[-10:])):
            bar_width = int(hist_conf * 780)
            color = self.emotion_colors[hist_emotion]
            pygame.draw.rect(self.screen, color, (10, history_start_y + i*20, bar_width, 15))
        
        pygame.display.flip()
        
    def run(self):
        """Start real-time emotion analysis"""
        self.running = True
        
        try:
            with sd.InputStream(channels=1,
                              samplerate=self.sample_rate,
                              callback=self.audio_callback):
                print("Real-time emotion analysis started. Press Ctrl+C to stop.")
                
                while self.running:
                    # Process events
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.running = False
                            break
                    
                    # Get audio data
                    try:
                        audio_data = self.audio_queue.get_nowait()
                        
                        # Update buffer
                        self.audio_buffer = np.roll(self.audio_buffer, -len(audio_data))
                        self.audio_buffer[-len(audio_data):] = audio_data.flatten()
                        
                        # Process and display
                        emotion, confidence = self.process_audio(self.audio_buffer)
                        
                        # Update history
                        self.emotion_history.append(emotion)
                        self.confidence_history.append(confidence)
                        if len(self.emotion_history) > 50:  # Keep last 50 predictions
                            self.emotion_history.pop(0)
                            self.confidence_history.pop(0)
                        
                        # Update display
                        self.update_display(emotion, confidence)
                        
                    except queue.Empty:
                        continue
                    
                    time.sleep(0.1)  # Prevent excessive CPU usage
                    
        except KeyboardInterrupt:
            print("\nStopping emotion analysis...")
        finally:
            pygame.quit()
            self.running = False

def main():
    analyzer = RealtimeEmotionAnalyzer('Audio_empath/best_emotion_model.pth')
    analyzer.run()

if __name__ == "__main__":
    main()