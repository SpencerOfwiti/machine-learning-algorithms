# importing libraries
import tensorflow as tf
from tensorflow import keras

# loading data
imdb = keras.datasets.imdb

# a dictionary mapping words to an integer index
_word_index = imdb.get_word_index()

word_index = {k: (v+3) for k, v in _word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2  # unknown
word_index['<UNUSED>'] = 3

# loading the model
model = keras.models.load_model('text_classifier_model.h5')


# transforming our data
def review_encode(s):
    encoded = [1]

    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)

    return encoded


# open our text file and read and predict the review
with open('text.txt', encoding='utf8') as f:
    for line in f.readlines():
        nline = line.replace(',', '').replace('.', '').replace('(', '').replace(')', '')\
            .replace(':', '').replace('\'', '').strip().split(" ")
        encode = review_encode(nline)
        # making the data 250 words long
        encode = keras.preprocessing.sequence.pad_sequences([encode],
                                                            value=word_index['<PAD>'],
                                                            padding='post',
                                                            maxlen=250)
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict)
