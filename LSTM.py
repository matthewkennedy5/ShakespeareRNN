import os
import string
import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
import load_embeddings
from progressbar_utils import init_progress_bar
import pdb
import pickle

# Hyperparameters
BATCH_SIZE = 2
HIDDEN_SIZE = 512
CHUNK_LENGTH = 1000  # How many characters to look at at a time
ITERATIONS = 10000
LEARNING_RATE = 5e-3
DECAY_RATE = 0.9    # Multiply the learning rate by this every so often
DECAY_EVERY = 10000 # Decay the learning rate after this many iterations

# Filenames
FILENAME = 'shakespeare.txt'
EMBED_FILE = 'char-embeddings.txt'
SAVE_NAME = 'test.pt'

EMBEDDINGS = load_embeddings.load(EMBED_FILE)
VOCABULARY = pickle.load(open('vocab.pkl', 'rb'))
CUDA = torch.cuda.is_available()

EMBED_SIZE = 300
PRINT_EVERY = 100
SAVE_EVERY = 1000
WARM_START = False   # Keep training the existing model instead of just using it
                    # to write out text

if CUDA:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def embed(text):
    """Perform character embedding on the given string.

    Input:
        text - String of characters which have defined embeddings

    Returns:
        Tensor of (len(text), EMBED_SIZE) containing the embeddings for each
        character.
    """
    result = torch.zeros(len(text), EMBED_SIZE, device=device)
    for i, ch in enumerate(text):
        if ch not in EMBEDDINGS:
            raise ValueError('Embeddings are not defined for the character "%s"' % (ch,))
        result[i, :] = EMBEDDINGS[ch]
    return result


def batch_generator(filename, batch_size, chunk_length, iterations):
    """Generate a single batch of training data.

    Each batch consists of multiple randomly-chosen excerpts of the text in the
    given file. Since the RNN is learning to predict characters, y is shifted to
    the right of x by one index.

    Inputs:
        filename - File containing the training text
        batch_size - Number of training examples per batch
        chunk_length - Number of characters per training example
        iterations - How many batches to train for

    Yields:
        i - Iteration number
        x - Batch of training data
        y - Vocabulary indices of the correct characters. y[n] corresponds to
                x[n-1].
    """
    with open(filename) as f:
        text = f.read()
    for i in range(iterations):
        x = torch.empty(batch_size, chunk_length, EMBED_SIZE, device=device)
        y = torch.empty(batch_size, chunk_length, device=device)
        for b in range(batch_size):
            # Randomly select a chunk of text and embed it to a tensor.
            start = np.random.randint(0, len(text) - chunk_length - 1)
            chunk = text[start : start+chunk_length+1]
            x[b, :, :] = embed(chunk[:-1])
            for c, ch in enumerate(chunk[1:]):
                y[b, c] = VOCABULARY.index(ch)
        yield i, x, y.long()



class Trainer:
    """Class to handle training of the RNN model.

    Constructor inputs:
        model - The RNN model to train
        gen - Batch generator instance to generate training data
    """
    def __init__(self, model, gen):
        self.model = model
        self.gen = gen
        self.criterion = nn.CrossEntropyLoss()

    def train(self, learning_rate, batch_size, iters):
        """Train the model for the given number of iterations.

        Inputs:
            learning_rate
            batch_size
            iters - Number of iterations to train for

        Returns:
            loss_history - Numpy array containing the loss at each iteration.
        """
        loss_history = np.zeros(iters)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, DECAY_RATE)
        bar = init_progress_bar(iters)
        bar.start()
        try:
            for i, x, y in self.gen:
                y_pred = self.model(x)
                loss_fn = 0
                for j in range(CHUNK_LENGTH):
                    loss_fn += self.criterion(y_pred[:, j, :], y[:, j])

                optimizer.zero_grad()
                loss_fn.backward()
                optimizer.step()
                loss_history[i] = loss_fn.item()
                print('\r', loss_fn.item())
                bar.update(i + 1)

                if i != 0:

                    if i % DECAY_EVERY == 0:
                        scheduler.step()
                        print('[INFO] Decayed the learning rate.')

                    if i % PRINT_EVERY == 0:
                        print('Sample:')
                        print(model.write(100))
                        print('-' * 80)

                    if i % SAVE_EVERY == 0:
                        torch.save(model.state_dict(), SAVE_NAME)
                        print('[INFO] Saved weights.')

        except KeyboardInterrupt:
            print('\n[INFO] Saving weights before quitting.\n')
            torch.save(model.state_dict(), SAVE_NAME)
            raise KeyboardInterrupt

        return loss_history


class SequenceGRU(nn.Module):
    """Two-layer LSTM model for sequence generation.

    Constructor inputs:
        input_size - Number of input dimensions of the data
        hidden_size - Desired size of hidden state
        output - Size of the output vector of the RNN
    """

    def __init__(self, input_size, hidden_size, output):
        super(SequenceGRU, self).__init__()
        self.cell = nn.LSTMCell(input_size, hidden_size)
        self.cell2 = nn.LSTMCell(hidden_size, hidden_size)
        self.cell3 = nn.LSTMCell(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output)
        self.hidden_size = hidden_size
        self.output_size = output

    def forward(self, x):
        """Compute a forward pass of SequenceGRU.

        The net takes input data at each time step, as opposed to the write()
        method in which the net uses its previous output as its next input.

        Input:
            x - Tensor of (batch_size, sequence_length, features) containing the
                input data.

        Returns:
            out - Tensor of (batch_size, sequence_length, features) containing
                the predicted sequence.
        """
        batch_size, seq_len, features = x.size()
        out = torch.zeros(batch_size, seq_len, self.output_size, device=device)
        self.cell.zero_grad()
        self.cell2.zero_grad()
        self.cell3.zero_grad()
        hidden = None
        hidden2 = None
        hidden3 = None
        for i in range(1, seq_len):
            hidden = self.cell(x[:, i, :], hidden)
            hidden2 = self.cell2(hidden[0], hidden2)
            hidden3 = self.cell3(hidden2[0], hidden3)
            out[:, i, :] = self.h2o(hidden3[0])
        return out

    def write(self, length):
        """Generate a string resembling the training data.

        After enough training, this method will return a string that should
        resemble the training data.

        Input:
            length - Number of characters in the output string

        Returns:
            result - String of generated text
        """
        with torch.no_grad():
            result = ''
            hidden1 = None
            hidden2 = None
            hidden3 = None
            x = embed(' ').to(device)
            for i in range(length):
                hidden1 = self.cell(x, hidden1)
                hidden2 = self.cell2(hidden1[0], hidden2)
                hidden3 = self.cell3(hidden2[0], hidden3)
                probs = nn.Softmax(dim=1)(self.h2o(hidden3[0])).cpu()
                probs = np.squeeze(np.array(probs))
                next_char = np.random.choice(VOCABULARY, p=probs)
                result += next_char
                x = embed(next_char)
            return result


if __name__ == '__main__':

    new_model = False
    if os.path.isfile(SAVE_NAME):
        # Load the pretrained model
        print('[INFO] Loading pretrained model.')
        model = SequenceGRU(EMBED_SIZE, HIDDEN_SIZE, len(VOCABULARY)).to(device)
        if CUDA:
            map_location = 'cuda'
        else:
            map_location = 'cpu'
        model.load_state_dict(torch.load(SAVE_NAME, map_location=map_location))
        model.train()
    else:
        # Train a new model from scratch
        model = SequenceGRU(EMBED_SIZE, HIDDEN_SIZE, len(VOCABULARY)).to(device)
        new_model = True

    if new_model or WARM_START:
        gen = batch_generator(FILENAME, BATCH_SIZE, CHUNK_LENGTH, ITERATIONS)
        trainer = Trainer(model, gen)
        loss = trainer.train(LEARNING_RATE, BATCH_SIZE, ITERATIONS)
        torch.save(model.state_dict(), SAVE_NAME)
        plt.figure()
        plt.plot(range(ITERATIONS), loss)
        plt.savefig('loss.png')

    model.eval()
    output = model.write(10000)
    print(output)
