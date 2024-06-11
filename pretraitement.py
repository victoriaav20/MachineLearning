import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_dir = './data_set/asl_alphabet_train/asl_alphabet_train'  

# Ajouts du rescale=1./255 pour normaliser les images
train_datagen = ImageDataGenerator(
    rescale=1./255,
    fill_mode='nearest',     # Stratégie de remplissage des pixels manquants
    validation_split=0.2     # 20% des données pour la validation
)

# Générateur pour les données d'entraînement
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(200, 200),  # Taille des images d'entrée
    batch_size=16,
    class_mode='categorical',  # Mode catégorique car nous avons plusieurs classes (26 lettres)
    subset='training'          
)

# Générateur pour les données de validation
validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(200, 200),
    batch_size=16,
    class_mode='categorical',
    subset='validation'        
)

  # Afficher quelques images d'entraînement pour validation
for images, labels in train_generator:
     for i in range(5):
        plt.imshow(images[i])
        plt.title(np.argmax(labels[i]))
        plt.show()
     break
