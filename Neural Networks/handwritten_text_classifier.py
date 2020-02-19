# importing mnist libraries using the terminal
# !git clone https://github.com/sorki/python-mnist
# !./python-mnist/get_data.sh
# !pip3 install emnist

# import libraries
from emnist import extract_training_samples
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import pickle

# getting data from the OpenML website
# x will be our images and y will be the labels
x, y = extract_training_samples('letters')

# make sure that every pixel in all of the images is a value between 0 and 1
x = x / 255.0

# use the first 114800 instances as training and the last 10000 as testing
x_train, x_test = x[:len(x) - 10000], x[len(x) - 10000:]
y_train, y_test = y[:len(x) - 10000], y[len(x) - 10000:]

# record number of samples in each dataset and the number of pixels in each image
x_train = x_train.reshape(len(x) - 10000, 784)
x_test = x_test.reshape(10000, 784)

# indices of some of the training images
indices = [14000, 8888, 1234]

# displaying some of the images
for index in indices:
    img = x_train[index]
    print('Image label:', chr(y_train[index]+96))
    plt.imshow(img.reshape((28, 28)))
    plt.show()

# creating the model
# creates a multi-layer perceptron with 1 hidden layer with 50 neurons
# and sets it to run through the data 20 times
model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=20, alpha=1e-4,
                      solver='sgd', verbose=10, tol=1e-4, random_state=1,
                      learning_rate_init=.1)

# fitting our model
model.fit(x_train, y_train)
print('Training set score:', model.score(x_train, y_train))
print('Test set score:', model.score(x_test, y_test))

# list with all predicted values from the training set
y_pred = model.predict(x_test)

# creating a model with 5 layers with 100 neurons each
# and sets it to run through the data 50 times
model2 = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100,), max_iter=50, alpha=1e-4,
                       solver='sgd', verbose=10, tol=1e-4, random_state=1,
                       learning_rate_init=.1)

# fitting the model
model2.fit(x_train, y_train)
print('Training set score:', model2.score(x_train, y_train))
print('Test set score:', model2.score(x_test, y_test))

# list with all predicted values from the training set
y_pred2 = model2.predict(x_test)

# visualizing the errors between predictions and actual labels
cm = confusion_matrix(y_test, y_pred2)
plt.matshow(cm)
plt.show()

# letters that may be confused
ambiguous = {'l': 'i', 'i': 'l', 'u': 'v', 'v': 'u'}

for k, v in ambiguous.items():
    # count all the mistakes for the letters above
    mistake_list = []
    for i in range(len(y_test)):
        if y_test[i] == (ord(v) - 96) and y_pred2[i] == (ord(k) - 96):
            mistake_list.append(i)
    print('There were', len(mistake_list), 'times that the letter', v, 'was predicted to be the letter', k)

    # change this to see image of a particular mistake
    mistake_to_show = 1

    # checks if mistake can be shown, if so displays it
    if len(mistake_list) > mistake_to_show:
        img = x_test[mistake_list[mistake_to_show]]
        plt.imshow(img.reshape((28, 28)))
        plt.show()
    else:
        print('Couldn\'t show mistake number', mistake_to_show + 1, 'because there were only', len(mistake_list), 'mistakes')

# saving our model
with open('./handwritten_text_classifier_model.pickle', 'wb') as f:
    pickle.dump(model2, f)
