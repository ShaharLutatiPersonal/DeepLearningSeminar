import torch.nn as nn
import torch.autograd as grad
import torch
from torch.optim.sgd import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau as plateau
import math
import ipywidgets
import traitlets
import matplotlib.pyplot as plt

"""
From wikipedia
Perplexity is a measurement of how well a probability distribution or probability model
predicts a sample.
perp = 2^(-sum(p(x)log2(q(x))))
one can easily see that the exponent is the cross-entropy
therefore in order to calculate the loss function all we need to do is
to use the cross_entropy and scale it by exp()
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 20
cross_entropy = nn.CrossEntropyLoss(reduction='mean')


def perplexity_loss(x):
    return math.exp(x)


def find_key(dic, val):
    for key in dic:
        if dic[key] == val:
            return key


def states_alloc(mode='lstm'):
    if mode == 'lstm':
        states = (torch.zeros(2, batch_size, 200).to(device),
                  torch.zeros(2, batch_size, 200).to(device))
    else:
        states = torch.zeros(2, batch_size, 200).to(device)
    return states


def state_detach(states, mode):
    if mode == 'lstm':
        (h, c) = states
        h = h.detach()
        c = c.detach()
        states = (h, c)
    else:
        states = states.detach()
    return states


sequence_length = 20  # maximal "memory" for the LSTM to remember
print('Start loading data')
train_vec, train_vocab = import_ptb_dataset(
    dataset_type='train', path='./data', batch_size=batch_size)
print('loaded the training set, using the vocabulary for other sets')
valid_vec, dontcare = import_ptb_dataset(
    dataset_type='valid', path='./data', batch_size=batch_size, vocabulary=train_vocab)
test_vec, dontcare = import_ptb_dataset(
    dataset_type='test', path='./data', batch_size=batch_size, vocabulary=train_vocab)
print('done loading')
dropout_rate = [0.35, 0]
print('run for dropouts :')
print(dropout_rate)
perpelxity_dict = {}
model_dict = {}

models = ['lstm', 'gru']
epochs_count = 20

def RunModel():
    for model_name in models:
        for dp in dropout_rate:
            print('Test models for dp = {}'.format(dp))
            if model_name == 'lstm':
                model = RNN_Zam(dict_size=len(train_vocab),
                                dp_prob=dp).to(device)
            else:
                model = GRU(dict_size=len(train_vocab), dp_prob=dp).to(device)

            if model_name == 'lstm':
                lr = 1*(1-dp*.5)
            else:
                lr = .5*(1-dp*.5)

            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer, mode='min', factor=.45, patience=2, verbose=True, threshold=2e-1)

            perpelxity_dict['{} {}'.format(model_name, dp)] = {
                'Train': [], 'Validation': [], 'Test': []}

            for epoch in range(epochs_count):
                # According to the article initialization with zero state and memory
                model.train()
                loss_term = 0
                state = states_alloc(model_name)

                for count, ix in enumerate(range(0, train_vec.size(1)-sequence_length, sequence_length)):
                    # taking overlapping vectors with seperation of one word
                    model.zero_grad()

                    x = train_vec[:, (ix): (ix + sequence_length)
                                  ].long().permute(1, 0).to(device)
                    y = train_vec[:, (ix+1):(ix+1) +
                                  sequence_length].long().permute(1, 0).to(device)

                    # Major notice ! if we don't detach pytorch will backprop the state which is big NO NO !
                    state = state_detach(state, model_name)

                    # forward is the only way we know how
                    pred, state = model(x, state)

                    # calculate loss - using cross_entropy
                    loss = cross_entropy(pred, y.reshape(-1))*y.size(1)

                    # backprop
                    loss.backward()
                    loss_term += loss.item()

                    with torch.no_grad():
                        norm = nn.utils.clip_grad_norm_(model.parameters(), 5)
                    optimizer.step()

                loss_term = (loss_term/(count+1))/batch_size
                valid_error = 0

                with torch.no_grad():
                    state = states_alloc(model_name)
                    model.eval()
                    print('Start Validation')

                    for count, ix in enumerate(range(0, valid_vec.size(1)-sequence_length, sequence_length)):
                        # taking overlapping vectors with seperation of one word
                        x = valid_vec[:, (ix): (
                            ix + sequence_length)].long().permute(1, 0).to(device)
                        y = valid_vec[:, (ix+1):(ix+1) +
                                      sequence_length].long().permute(1, 0).to(device)

                        # forward is the only way we know how
                        pred, state = model(x, state)

                        # calculate loss - using cross_entropy
                        valid_error += cross_entropy(pred,
                                                     y.reshape(-1)).item()

                    valid_error = valid_error/(count+1)
                    scheduler.step(valid_error)
                    print('Start Testing')

                    # Test Section
                    state = states_alloc(model_name)
                    test_error = 0

                    for count, ix in enumerate(range(0, test_vec.size(1)-sequence_length, sequence_length)):
                        # taking overlapping vectors with seperation of one word
                        x = test_vec[:, (ix): (ix + sequence_length)
                                     ].long().permute(1, 0).to(device)
                        y = test_vec[:, (ix+1):(ix+1) +
                                     sequence_length].long().permute(1, 0).to(device)

                        # forward is the only way we know how
                        pred, state = model(x, state)

                        # calculate loss - using cross_entropy
                        test_error += cross_entropy(pred, y.reshape(-1)).item()

                    test_error = test_error/(count+1)

                perpelxity_dict['{} {}'.format(model_name, dp)]['Train'].append(
                    perplexity_loss(loss_term))
                perpelxity_dict['{} {}'.format(model_name, dp)]['Validation'].append(
                    perplexity_loss(valid_error))
                perpelxity_dict['{} {}'.format(model_name, dp)]['Test'].append(
                    perplexity_loss(test_error))
                print('epoch {} Train Loss = {},Val Loss = {}, Test loss = {}'.format(
                    epoch, perplexity_loss(loss_term), perplexity_loss(valid_error), perplexity_loss(test_error)))
                print('Train Loss {}, Val Loss {} '.format(
                    loss_term, valid_error))

            model_dict['{} {}'.format(model_name, dp)] = model

    for mode in perpelxity_dict:
        fig, ax = plt.subplots()

        for error_type in perpelxity_dict[mode]:
            ax.plot(perpelxity_dict[mode][error_type], label=error_type)

        ax.set_title(mode)
        ax.grid()
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.legend()
        fig.show()

    fig, ax = plt.subplots()
    for mode in perpelxity_dict:
        ax.plot(perpelxity_dict[mode]['Validation'], label=mode)

    ax.grid()
    plt.xlabel('Epoch')
    ax.set_title('Accuracies different methods')
    plt.ylabel('Perplexity')
    plt.legend()
    fig.show()
    return model_dict, perpelxity_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-m', '--model', type=str, default='all',
        help='Model used for training, default <all>',
        choices=['all', 'lstm', 'gru'])
    parser.add_argument(
        '-b', '--batch_size', type=int, default=256, help='Batch size, default <256>')
    parser.add_argument(
        '-e', '--epochs', type=int, default=15, help='Number of epochs, default <15>')
    parser.add_argument(
        '-p', '--data_path', type=str, default='./data/mnist',
        help='Path to database, default <"./data/mnist">')
    parser.add_argument(
        '-v', '--verbose', dest='verbose', action='store_true', default=False, help='Enable verbose mode')
    parser.add_argument(
        '-l', '--test_mode', dest='test_mode', action='store_true', default=False, help='Load models for testing only')

    args = parser.parse_args()

    train_and_test(args.technique, args.batch_size, args.epochs,
                   args.data_path, args.verbose, args.test_mode)
