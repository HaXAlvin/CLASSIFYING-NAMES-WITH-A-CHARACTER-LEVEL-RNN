# Practice Pytorch @Author: Alvin Hsueh @Date: 2022-08-26
# @Reference: https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

import random
import torch
import torch.nn as nn
import glob
from string import ascii_letters
import time
import math
import os


#### Path ####

data_path = os.path.dirname(os.path.abspath(__file__)) + '/data/names/*.txt'
model_save_path = os.path.dirname(os.path.abspath(__file__)) + '/rnn.model'

#### Data Prepare ####

all_names = {}  # store all names by file
all_letters = set(ascii_letters)  # store all existed letters
for file in glob.glob(data_path):
    with open(file, 'r') as f:
        names = f.readlines()
    names = ["".join(name.split()).split('\n')[0] for name in names]  # remove \xa0 and space and \n
    all_names[file.split('/')[-1].split('.txt')[0]] = names
    all_letters.update([letter for name in names for letter in name])

all_categories = list(all_names.keys())
all_letters = "".join(all_letters)
letter_count = len(all_letters)  # count of all letters


#### Functions about data logics ####

def letter_to_index(letter):  # return -1 if letter not exist
    return all_letters.find(letter)


def letter_to_tensor(letter):  # one-hot vector, (last one is unknown letter)
    tensor = torch.zeros(1, letter_count+1)
    tensor[0][letter_to_index(letter)] = 1
    return tensor


def output_to_category(output):
    _, index = output.topk(1)
    index = index.item()
    return all_categories[index]


def random_train_data():
    category = random.choice(all_categories)
    name = random.choice(all_names[category])
    return category, name

#### Model ####


class RNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.output_linear = nn.Linear(input_size+input_size, output_size)
        self.hidden_linear = nn.Linear(input_size+input_size, input_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()
        self.train_rate = 0.005

    def forward(self, input, hidden):
        concat = torch.cat((input, hidden), 1)
        hidden = self.hidden_linear(concat)
        output = self.output_linear(concat)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.input_size)

    def train(self, category, name):
        hidden = self.init_hidden()
        self.zero_grad()
        for letter in name:
            output, hidden = self(letter_to_tensor(letter), hidden)

        loss = self.loss(output, torch.tensor(
            [all_categories.index(category)]))
        loss.backward()

        for p in self.parameters():
            p.data.add_(p.grad.data, alpha=-self.train_rate)

        return output, loss.item()

    def test(self, name, n_predictions=3):
        with torch.no_grad():
            hidden = self.init_hidden()
            for letter in name:
                output, _ = self(letter_to_tensor(letter), hidden)
            top_values, top_index = output.topk(n_predictions, 1, True)
            for i in range(n_predictions):
                value = top_values[0][i].item()
                category_index = top_index[0][i].item()
                print('(%.2f) %s' % (value, all_categories[category_index]))


#### Train ####

rnn = RNN(letter_count+1, len(all_categories))

n_iters = 100000
print_every = 5000
plot_every = 1000

# Keep track of losses for plotting
current_loss = 0
all_losses = []

def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range(1, n_iters + 1):
    category, name = random_train_data()
    output, loss = rnn.train(category, name)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess = output_to_category(output)
        correct = '✓' if guess == category else f'✗ ({category})'
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters *
              100, time_since(start), loss, name, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

torch.save(rnn, model_save_path)

#### Draw loss ####

# import matplotlib.pyplot as plt

# plt.figure()
# plt.plot(all_losses)
# plt.show()

#### Test Model ####

while True:
    name = input("Input a name: ")
    rnn.test(name)
