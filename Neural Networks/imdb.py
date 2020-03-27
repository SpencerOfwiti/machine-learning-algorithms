# importing libraries
from tensorflow import keras
import pandas as pd

# loading data
imdb = keras.datasets.imdb

# splitting our dataset into training and testing data
(train_data, train_labels), (test_data, test_labels) = imdb.load_data()

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


imdb_data = {'Review': [], 'Label': []}
for i in range(len(train_data)):
    imdb_data['Review'].append(decode_review(train_data[i-1]))
    imdb_data['Label'].append(train_labels[i-1])

for i in range(len(test_data)):
    imdb_data['Review'].append(decode_review(test_data[i-1]))
    imdb_data['Label'].append(test_labels[i-1])

df = pd.DataFrame(imdb_data)
print(df)

word_index_dict = {'Word': [], 'Index': []}
for key, value in word_index.items():
    word_index_dict['Word'].append(key)
    word_index_dict['Index'].append(value)

print(max(word_index_dict['Index']))
if 'hate' in word_index_dict['Word']:
    hate_index = word_index_dict['Index'][word_index_dict['Word'].index('hate')]
    print(hate_index)

print(decode_review([hate_index]))
