import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# --- Setup model ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Must match your training setup
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 7)
model.load_state_dict(torch.load("models/fer_resnet18.pth", map_location=device))
model = model.to(device)
model.eval()

# Emotion labels (same order as dataset folders)
class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# --- Define preprocessing ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # or (48,48) if trained that way
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# --- Setup face detector ---
face_cascade = cv2.CascadeClassifier('data/models/haarcascade_frontalface_default.xml')

# --- Start webcam ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        # Crop face region
        face = frame[y:y+h, x:x+w]
        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

        # Preprocess
        input_tensor = transform(face_pil).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            _, pred = torch.max(output, 1)
            emotion = class_names[pred.item()]

        # Draw box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2, cv2.LINE_AA)

    # Display
    cv2.imshow("Facial Emotion Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
