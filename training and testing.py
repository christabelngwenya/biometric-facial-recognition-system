import os
import cv2
import pickle
import glob
from PIL import  Image
from numpy import asarray

face_cascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

current_id = 0
labels_names = {}

# List to hold all subject faces and labels
x_train = []
y_train = []

# Get directories containing images
image_files = glob.glob(os.path.join(image_dir, '**', '*.jp*g'), recursive=True)

for path in image_files:
    label = os.path.basename(os.path.dirname(path))

    if label not in labels_names:
        labels_names[label] = current_id
        current_id += 1

    person_identity = labels_names[label]
    pil_image    = Image.open(path).convert('L')#CONVERTING TO GRAYSCALE USING PILLOW

    data = asarray(pil_image)
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(data, scaleFactor=1.1, minNeighbors=9)

    for (x, y, w, h) in faces:
        roi = data[y:y + h, x:x + w]
        x_train.append(roi)
        y_train.append(person_identity)

# Save labels to a pickle file
with open("labels.pickle", "wb") as f:
    pickle.dump(labels_names, f)

# Train the face recognizer
y_labels = asarray(y_train)
face_recognizer.train(x_train, y_labels)

# Save the trained recognizer
face_recognizer.save("trainer.yml")