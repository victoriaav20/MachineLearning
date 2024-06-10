import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Chemin vers le dossier de dataset
data_dir = './data_set/asl_alphabet_train/asl_alphabet_train'  # Remplacez par le chemin d'accès réel au dossier de votre dataset
# Préparation des générateurs de données
# Ajoutez rescale=1./255 pour normaliser les images
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,       # Rotation aléatoire de l'image de 0 à 20 degrés
    width_shift_range=0.2,   # Décalage horizontal aléatoire
    height_shift_range=0.2,  # Décalage vertical aléatoire
    shear_range=0.2,         # Cisaillement de l'image
    zoom_range=0.2,          # Zoom avant ou arrière dans l'image
    horizontal_flip=True,    # Retournement horizontal des images
    fill_mode='nearest',     # Stratégie de remplissage des pixels manquants
    validation_split=0.2     # 20% des données pour la validation
)

# Générateur pour les données d'entraînement
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(200, 200),  # Taille des images d'entrée
    batch_size=32,
    class_mode='categorical',  # Mode catégorique car nous avons plusieurs classes (26 lettres)
    subset='training'          # Sous-ensemble d'entraînement
)

# Générateur pour les données de validation
validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(200, 200),
    batch_size=32,
    class_mode='categorical',
    subset='validation'        # Sous-ensemble de validation
)
