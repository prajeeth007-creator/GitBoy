import cv2
from fer import FER

# Initialize the detector
detector = FER(mtcnn=True)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Webcam didn't open!")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect emotions
    results = detector.detect_emotions(frame)

    # Draw bounding boxes and top emotion
    for face in results:
        x, y, w, h = face["box"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Get the top emotion
        emotion, score = detector.top_emotion(frame)
        if emotion:
            cv2.putText(frame, f"{emotion} ({score:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("GitBoy Emotion Detector", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




