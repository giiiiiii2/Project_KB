import cv2
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Function to get face images and labels from all subfolders
def get_images_and_labels(main_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = []
    labels = []
    label_names = {}
    current_label = 0

    # Ensure the dataset folder exists
    if not os.path.exists(main_path):
        print(f"Error: Folder '{main_path}' tidak ditemukan.")
        return faces, labels, label_names

    # Iterate through all subfolders in the main folder
    for folder_name in os.listdir(main_path):
        folder_path = os.path.join(main_path, folder_name)

        if os.path.isdir(folder_path):
            label_names[current_label] = folder_name
            print(f"Processing folder: {folder_name} with label {current_label}")

            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)

                # Read the image in grayscale
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Warning: Gambar '{image_name}' tidak valid, dilewati.")
                    continue  # Skip if the image is not valid

                # Detect faces
                faces_detected = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))

                for (x, y, w, h) in faces_detected:
                    # Resize the detected face and add to list
                    face_resized = cv2.resize(img[y:y+h, x:x+w], (150, 150))
                    faces.append(face_resized)
                    labels.append(current_label)

            current_label += 1

    return faces, labels, label_names

# Path to the dataset
dataset_path = r'C:\Users\0S\OneDrive\KULIAH\KULIAH\SEMESTER 3\Kecerdasan Buatan\UAS_KB_Face_Clasification'

# Get face images and labels
faces, labels, label_names = get_images_and_labels(dataset_path)

# Validate the dataset
if len(faces) > 0:
    print("Dataset berhasil dimuat.")
    # Flatten faces for SVM training
    faces_flattened = [face.flatten() for face in faces]

    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    # Train SVM classifier
    clf = SVC(kernel='linear', probability=True)
    clf.fit(faces_flattened, labels_encoded)

    # Save the trained model and label encoder
    np.save('svm_model.npy', clf)
    np.save('label_encoder.npy', le)
    print("Model berhasil disimpan.")
else:
    print("Error: Dataset kosong atau tidak valid. Pastikan dataset berisi gambar wajah.")
    exit()

# Load the trained model and label encoder
clf = np.load('svm_model.npy', allow_pickle=True).item()
le = np.load('label_encoder.npy', allow_pickle=True).item()

# Initialize face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Tidak dapat mengakses webcam. Pastikan webcam terhubung.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))

    for (x, y, w, h) in faces_detected:
        # Resize the detected face for prediction
        face_resized = cv2.resize(gray[y:y+h, x:x+w], (150, 150)).flatten().reshape(1, -1)

        # Predict face identity
        label_encoded = clf.predict(face_resized)

        # Get probabilities from the SVM model
        proba = clf.predict_proba(face_resized)
        confidence = np.max(proba)

        # Decode label
        label = le.inverse_transform(label_encoded)[0]
        name = label_names.get(label, "Unknown")

        # Display prediction result on the image
        cv2.putText(frame, f"{name} ({int(confidence * 100)}%)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Show video frame
    cv2.imshow('Face Recognition', frame)

    # Exit if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()