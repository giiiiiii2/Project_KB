import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Fungsi untuk memuat gambar dari folder
def load_images_from_folder(folder):
    images = []
    labels = []
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            img_path = os.path.join(subdir, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(os.path.basename(subdir))
    return images, labels

# Fungsi untuk preprocessing gambar
def preprocess_images(images, size=(100, 100)):
    processed_images = []
    for img in images:
        img_resized = cv2.resize(img, size)
        processed_images.append(img_resized)
    return np.array(processed_images)

# Path ke dataset
folder_path = 'dataset/'

# Muat dan preprocess dataset
images, labels = load_images_from_folder(folder_path)
images = preprocess_images(images)
images = images.reshape(len(images), -1)  # Flatten images untuk input SVM

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Bagi data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=42)

# Latih model SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Prediksi pada test set
y_pred = svm_model.predict(X_test)

# Hitung akurasi
accuracy = accuracy_score(y_test, y_pred)
print(f'Akurasi: {accuracy * 100:.2f}%')

# Tampilkan laporan klasifikasi
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Simpan model
model_path = 'models/face_recognition_model.pkl'
joblib.dump(svm_model, model_path)
print(f'Model disimpan ke {model_path}')
