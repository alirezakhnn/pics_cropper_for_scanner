import cv2
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Load pre-trained face detection cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained Random Forest model
model = RandomForestClassifier(n_estimators=100)

# Load the data
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Function to crop and save detected face images
def crop_and_save_faces(image_path, save_directory, desired_height):
    # Read image from file
    img = cv2.imread(image_path)
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
    # Create the save directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    for index, (x, y, w, h) in enumerate(faces):
        # Crop face image with padding
        pad = int(0.2 * w)  # 20% padding around the face
        x_start = max(x - pad, 0)
        y_start = max(y - pad, 0)
        x_end = min(x + w + pad, img.shape[1])
        y_end = min(y + h + pad, img.shape[0])
        face_img = img[y_start:y_end, x_start:x_end]

        # Resize face image to desired size
        desired_width = int(desired_height * w / h)
        face_img = cv2.resize(face_img, (desired_width, desired_height))

        # Save face image as JPEG with desired quality
        face_img_path = os.path.join(save_directory, f"face_{index+1}.jpg")
        cv2.imwrite(face_img_path, face_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"Saved face_{index+1}.jpg")

# Test the function on an example image and save faces in the given directory
image_path = './input_pics/demo.png'
save_directory = './cropped_faces'
desired_height = 400
crop_and_save_faces(image_path, save_directory, desired_height)
