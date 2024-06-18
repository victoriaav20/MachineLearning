from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

model = load_model('model_sign_language.h5')

test_datagen = ImageDataGenerator(rescale=0.2)
test_generator = test_datagen.flow_from_directory(
    './data_set/asl_alphabet_train/asl_alphabet_train',
    target_size=(200, 200),
    batch_size=32,
    class_mode='categorical'
)

# Évaluer le modèle sur les données de test
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
