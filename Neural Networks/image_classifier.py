# import libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# load dataset
data = keras.datasets.fashion_mnist

# splitting data into training and testing data
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# defining a list of class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# pre-processing images
# scaling the pixel values down to make computations easier
train_images = train_images/255.0
test_images = test_images/255.0

# creating our model
# a sequential model with 3 layers from input to output layer
# input layer of 784 neurons representing the 28*28 pixels in a picture
# hidden layer of 128 neurons
# output layer of 10 neurons for each of the 10 classes
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# compiling and training the model
# compiling is picking the optimizer, loss function and metrics to keep track of
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

# testing the accuracy of our model
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('\nTest accuracy:', test_acc)

# using the model to make predictions
predictions = model.predict(test_images)

# displaying first 5 images and their predictions
plt.figure(figsize=(5, 5))
for i in range(5):
    plt.grid(False)
    # plotting an image
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel('Actual: ' + class_names[test_labels[i]])
    plt.title('Prediction: ' + class_names[np.argmax(predictions[i])])
    plt.show()
