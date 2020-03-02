# importing libraries
from urllib.request import urlopen
import spacy
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import wrap
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.categorical as cat
import numpy as np
import time

txt = urlopen("https://raw.githubusercontent.com/crash-course-ai/lab2-nlp/master/vlogbrothers.txt")\
    .read().decode('ascii').split("\n")

# lexical types
everything = set([w for s in txt for w in s.split()])
print('The dataset contains', len(txt), 'vlogbrothers scripts and', len(everything), 'lexical types')

# tokenizing the text
nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner', 'textcat'])
txt = [nlp(s) for s in txt]

# mark the beginning and end of each script
txt = [['<s>'] + [str(w) for w in s] + ['</s>'] for s in txt]

# separating the data into training and testing
train = txt[:-5]
test = txt[-5:]

# flatten the lists into one long string and remove extra whitespace
train = [w for s in train for w in s if not w.isspace()]
test = [w for s in test for w in s if not w.isspace()]


# cleaning up the data
def simplify(w):
    # remove extra punctuation
    w = w.replace('-', '').replace('~', '')

    # replace numbers with # sign
    w = re.sub('\d', '#', w)

    # change some endings
    if len(w) > 3 and w[-2:] in set(['ed', 'er', 'ly']):
        return [w[:-2], w[-2:]]
    elif len(w) > 4 and w[-3:] in set(['ing', "'re"]):
        return [w[:-3], w[-3:]]
    return [w]


train_clean = []
for w in train:
    for piece in simplify(w):
        train_clean.append(piece)

test_clean = []
for w in test:
    for piece in simplify(w):
        test_clean.append(piece)

print('Lexical types:', len(set(train_clean)))  # lexical types
print('Lexical tokens:', len(train_clean))  # lexical tokens

# Replace rare words with unknown
counts_clean = Counter(train_clean)
train_unk = [w if counts_clean[w] > 1 else 'unk' for w in train_clean]
test_unk = [w if w in counts_clean and counts_clean[w] > 1 else 'unk' for w in test_clean]

# count the frequencies of every word
counts = Counter(train_unk)

frequencies = [0]*8
for w in counts:
    if counts[w] >= 128:
        frequencies[0] += 1
    elif counts[w] >= 64:
        frequencies[1] += 1
    elif counts[w] >= 32:
        frequencies[2] += 1
    elif counts[w] >= 16:
        frequencies[3] += 1
    elif counts[w] >= 8:
        frequencies[4] += 1
    elif counts[w] >= 4:
        frequencies[5] += 1
    elif counts[w] >= 2:
        frequencies[6] += 1
    else:
        frequencies[7] += 1

# plot their distributions
f, a = plt.subplots(1, 1, figsize=(10, 5))
a.set(xlabel='Lexical types occurring more than n times', ylabel='Number of lexical types')

labels = [128, 64, 32, 16, 8, 4, 2, 1]
_ = sns.barplot(labels, frequencies, ax=a, order=labels)
plt.show()

# examples of rare words
rare = [w for w in counts_clean if counts_clean[w] == 1]
rare.sort()
for line in wrap(" ".join(['{:15s}'.format(w) for w in rare[-100:]]), width=70):
    print(line)

# prepare the dataset by converting words to numbers
# create a mapping from words to numbers
vocabulary = set(train_unk)
word_to_num = {}
num_to_word = {}
for num, word in enumerate(vocabulary):
    word_to_num[word] = num
    num_to_word[num] = word

# convert the dataset into numbers
train = torch.LongTensor(len(train_unk))
for i in range(len(train_unk)):
    train[i] = word_to_num[train_unk[i]]

test = torch.LongTensor(len(test_unk))
for i in range(len(test_unk)):
    test[i] = word_to_num[test_unk[i]]

# model parameters
batch_size = 20
seq_len = 35

# specifying computational GPU
device = torch.device('cpu')


# citation: https://github.com/pytorch/examples/tree/master/word_language_model
def batchify(data, bsz):
    # work out how cleanly we can divide the dataset into bsz parts
    nbatch = data.size(0) // bsz
    # trim off any extra elements that wouldn't cleanly fit
    data = data.narrow(0, 0, nbatch * bsz)
    # evenly divide the data across the bsz batches
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def get_batch(source, i, seq_len):
    seq_len = min(seq_len, len(source) - 1 - i)
    data = source[i: i + seq_len]
    target = source[i + 1: i + 1 + seq_len].view(-1)
    return data, target


def repackage_hidden(h):
    # wraps hidden states in new tensors to detach their history
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


train = batchify(train, batch_size)
test = batchify(test, batch_size)


class EncoderDecoder(nn.Module):
    def __init__(self):
        """
        Define all the parameters of the model
        """
        super(EncoderDecoder, self).__init__()
        # How tightly should we compress our language representations?
        self.embed_size = 300  # word vector size
        self.hidden_size = 600  # hidden space size

        """
        Converting words to vectors
        """
        # A lookup table for translating a word into a vector
        self.embedding = nn.Embedding(len(vocabulary), self.embed_size)
        # Initialize the word vectors with a random uniform distribution
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

        """
        An RNN (LSTM) with dropout
        """
        self.rnn = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size)
        self.shrink = nn.Linear(self.hidden_size, self.embed_size)
        self.drop = nn.Dropout(p=0.5)

        """
        Predicting words from our models
        """
        # Converting the vectors into a set of words over scores
        self.decode = nn.Linear(self.embed_size, self.embedding.weight.size(0))
        # Using the same matrix for decoding that was used for encoding
        self.decode.weight = self.embedding.weight
        self.decode.bias.data.zero_()

    def forward(self, input, hidden=None):
        """
        Running the model
        :param input:
        :param hidden:
        :return hidden, decoded:
        """
        # Mapping words to vectors
        embedded = self.embedding(input)
        # process with an RNN
        if hidden is not None:
            output, hidden = self.rnn(embedded, hidden)
        else:
            output, hidden = self.rnn(embedded)
        # apply dropout
        output = F.relu(self.shrink(self.drop(output)))
        # score the likelihood of every possible next word
        decoded = self.decode(output)
        return hidden, decoded


# training the model
def training(model, data, targets, lr, hidden):
    # reset the model
    model.zero_grad()

    # run the model to see its predictions and hidden states
    hidden, prediction_vector = model(data, hidden)
    prediction_vector = prediction_vector.view(seq_len * batch_size, -1)

    # compare the model's predictions at each timestep to original data
    loss = F.cross_entropy(prediction_vector, targets)

    # compute gradients and perform back-propagation
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
    for p in model.parameters():
        if p.grad is not None:
            p.data.add_(-lr, p.grad.data)

    # return the current model loss on this data item
    return loss.item(), repackage_hidden(hidden)


# evaluating the model
def evaluation(model):
    """
    Performs all the same logic as the training function but it does not perform back-propagation, because we dont want
    to learn from this data, just check the performance
    :param model:
    :return:
    """
    model.eval()
    hidden = None
    test_loss = 0
    for i in range(0, test.size(0) - seq_len, seq_len):
        data, targets = get_batch(test, i, seq_len)
        hidden, prediction_vector = model(data, hidden)
        hidden = repackage_hidden(hidden)

        prediction_vector = prediction_vector.view(-1, len(vocabulary))
        loss = F.cross_entropy(prediction_vector, targets)
        test_loss += loss.item()
    return test_loss / (test.size(0)/seq_len)


# creating an instance of the model
model = EncoderDecoder().float().to(device)
prev_test_loss = 1e100
# scaling the size of each step in back-propagation
learning_rate = 20
batch_size = 20  # should match batch_size used earlier for splitting up the data

num_epochs = 10
timing = time.time()
for epoch in range(num_epochs):
    print('Start')
    # set the model to training mode and iterate through the dataset
    model.train()
    hidden = None
    train_loss = 0
    start_time = time.time()
    for i in range(0, train.size(0) - 1, seq_len):
        # get the next training batch
        data, targets = get_batch(train, i, seq_len)

        # run the model and perform back-propagation
        loss, hidden = training(model, data, targets, learning_rate, hidden)
        train_loss += loss

    # Evaluate how well the model predicts unseen test data
    test_loss = evaluation(model)

    # check if the models ability to generalize has gotten worse
    # if so, slow the learning rate
    if test_loss > prev_test_loss:
        learning_rate /= 4.0

    # print the training and testing performance
    train_loss /= (train.size(0)/seq_len)
    finish_time = time.time()
    print('Epoch', epoch, 'took', finish_time - start_time, 'with train perplexity:', np.exp(train_loss),
          'and validation:', np.exp(test_loss))

    prev_test_loss = test_loss

total_time = (time.time() - timing)/60
print('Completed', num_epochs, 'epochs in', total_time, 'minutes')

# inference
# word to start sentence with
prefix = '<s> Good'

# number of words we want the model to produce
words_to_generate = 50

# number of examples at a time
batch_size = 1

# set the model to be in evaluation mode (no back-propagation)
model.eval()

argmax_sent = None
argmax_prob = 0
collection = []
for item in range(100):
    # convert our sentence start into numbers
    valid = [word_to_num[word] if word in word_to_num else word_to_num['unk'] for word in prefix.split()]
    probabilities = []

    # run the model on the same initial input and it's own generations until we reach `word_to_generate`
    for w in range(words_to_generate):
        # run the model
        input = torch.from_numpy(np.array(valid)).to(device)
        _, output = model(input.view(-1, 1))

        # get the prediction for the next word
        last_pred = output[-1, :, :].squeeze()

        # block generation of unk
        last_pred[word_to_num['unk']] = -100

        # sampling from the distribution
        if item > 0:
            # a temperature makes the distribution peakier (if < 1) or flatter if > 1
            last_pred /= 0.70

            # turn into a distribution
            dist = cat.Categorical(logits=last_pred)

            # sample
            predicted_idx = dist.sample().item()

        else:
            # if we aren't sampling, just take the most probable word
            _, predicted_idx = last_pred.max()
            predicted_idx = predicted_idx.item()

        # save the predicted word's probability
        value = F.log_softmax(last_pred, -1)[predicted_idx].item()

        # add the predicted word to the list
        valid.append(predicted_idx)

        # save the probability for sorting later
        probabilities.append(value)

    if item > 0:
        # add the sentence and it's score to a list
        generation = (np.exp(np.sum(probabilities)), ' '.join([num_to_word[w] for w in valid]))
        if generation not in collection:
            collection.append(generation)
    else:
        argmax_sent = ' '.join([num_to_word[w] for w in valid])
        argmax_prob = np.exp(np.sum(probabilities))

# get the best model predictions
collection.sort()
collection.reverse()
print('Argmax Generation:')
print('{:.2E}:  {}\n'.format(argmax_prob, '\n\t\t'.join(wrap(argmax_sent))))
print('\nSampled Generations:')
for probability, sent in collection[:10]:
    print('{:.2E}:  {}\n'.format(probability, '\n\t\t'.join(wrap(sent))))
