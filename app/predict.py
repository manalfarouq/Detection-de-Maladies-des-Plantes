import tensorflow as tf
import numpy as np
import cv2
import os

model = tf.keras.models.load_model('notebooks/plantes_detection_model.keras')

class_names = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
              'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot',
              'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold',
              'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite',
              'Tomato__Target_Spot', 'Tomato__Tomato_Yellow_Leaf_Curl_Virus',
              'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']

def predict_disease(image_path):
    # Lire l'image en niveaux de gris
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None

    # Redimensionner à 48x48 (comme dans le modèle) 
    img = cv2.resize(img, (48, 48))

    # Normalisation
    img = img / 255.0

    # Ajouter dimension batch et canal : shape (1, 48, 48, 1) 
    img = np.expand_dims(img, axis=(0, -1))

    # Prédiction
    predictions = model.predict(img)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    return confidence, predicted_class