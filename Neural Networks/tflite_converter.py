import tensorflow as tf
from tensorflow import keras
from tensorflow import lite

# loading the model
model = keras.models.load_model('text_classifier_model.h5')

# Converting a tf.Keras model to a TensorFlow Lite model.
converter = lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

open('text_classifier_model_lite.tflite', 'wb').write(tflite_model)
