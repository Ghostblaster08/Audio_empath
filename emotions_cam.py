import cv2
import time
from deepface import DeepFace

# Open the webcam
cap = cv2.VideoCapture(0)

# Set interval (in seconds) for capturing screenshots
capture_interval = 5  # Capture an image every 5 seconds
last_capture_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get the current time
    current_time = time.time()

    # Capture and analyze emotion at specified intervals
    if current_time - last_capture_time >= capture_interval:
        last_capture_time = current_time  # Reset timer

        try:
            # Analyze emotion
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            print(f"Detected Emotion: {emotion}")
        except Exception as e:
            print(f"Error analyzing emotion: {e}")

    # Display the video feed
    cv2.imshow("Emotion Recognition", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()