# importing libraries
import tensorflow as tf
from tensorflow import keras

# loading data
imdb = keras.datasets.imdb

# splitting our dataset into training and testing data
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=100000)

# converting integer encoded words to string
# a dictionary mapping words to an integer index
_word_index = imdb.get_word_index()

word_index = {k: (v+3) for k, v in _word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2  # unknown
word_index['<UNUSED>'] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


# will return the decoded (human readable) reviews
def decode_review(text):
    return " ".join([reverse_word_index.get(i, '?') for i in text])


# pre-processing the data
# if review is greater than 250 words, trim off extra words
# if review is less than 250 words, add the necessary amount of 's to make it equal to 250
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index['<PAD>'],
                                                        padding='post',
                                                        maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index['<PAD>'],
                                                       padding='post',
                                                       maxlen=250)

# defining the model
model = keras.Sequential()
model.add(keras.layers.Embedding(100000, output_dim=16, input_length=250))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

# summary of the model
print(model.summary())

# example of a review
print(decode_review(train_data[0]))

# compiling the model
# compiling is picking the optimizer, loss function and metrics to keep track of
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# splitting our training data into training and validation
x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

# training the model
fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512,
                     validation_data=(x_val, y_val), verbose=1)

# testing the model
results = model.evaluate(test_data, test_labels)
print(results)

# saving the model
model.save('text_classifier_model.h5')
