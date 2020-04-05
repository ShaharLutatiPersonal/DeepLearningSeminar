import Lenet5
import torch
from math import floor as floor
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import argparse
import time
import matplotlib.pyplot as plt
import os

models = [Lenet5.NetOriginal(), Lenet5.NetD(), Lenet5.NetBN(),
          Lenet5.NetOriginal()]

models_technique = ['none', 'dropout', 'bn', 'wd']


def test_models(model, test_loader, device):
    correct = 0
    sumv = 0
    for idx, data in enumerate(test_loader):
        test_x, test_label = data[0].to(device), data[1].to(device)
        predicted_labels = model(test_x.float()).detach()
        predict_ys = torch.argmax(predicted_labels, dim=1)
        _est = predict_ys == test_label
        correct += torch.sum(_est, dim=0).cpu().item()
        sumv += _est.shape[0]
    return correct, sumv

def train_and_test(mode, batch_size, epochs, data_path, verbose, test_mode=''):
    # Start training network
    print('*'*80)
    print('Executing technique:' + mode +
          ', batch size:{} '.format(batch_size)
          + ', epochs:{} '.format(epochs) + ', data path:{}'.format(data_path))
    print('*'*80)

    # load FashionMNIST dataset from path or download it if it doesn't exist
    train_dataset = mnist.FashionMNIST(
        data_path, train=True, download=True, transform=ToTensor())
    test_dataset = mnist.FashionMNIST(
        data_path, train=False, download=True, transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    global models
    global models_technique

    if mode != 'all':
        ix = models_technique.index(mode)
        models = [models[ix]]
        models_technique = [models_technique[ix]]

    results_dict = {'none': [], 'dropout': [], 'bn': [], 'wd': []}

    train_results_dict = {'none': [], 'dropout': [], 'bn': [], 'wd': []}

    train_samples_num = train_loader.dataset.train_data.shape[0]

    # Iteration per epoch
    iterations = len(train_loader)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if test_mode != '':
        for technique in models_technique:
            tested_model = torch.load('./models/{}.pth'.format(technique))
            tested_model.eval()
            correct, sumv = test_models(tested_model, test_loader, device)
            results_dict[technique].append(correct / sumv)
            train_results_dict[technique].append(0)

    else:
        # Train selected technique
        for technique, model in zip(models_technique, models):
            wd = 0
            if technique == 'wd':
                wd = 0.001

            if torch.cuda.is_available():
                model.to(device)

            # Declare optimizer
            sgd = SGD(model.parameters(), lr=1e-1, weight_decay=wd)

            # Declare used loss
            cross_error = CrossEntropyLoss()

            print('Start training model: ' + technique)

            # Iterate database the number of epochs
            for epoch in range(epochs):
                start_time = time.time()
                epoch_total_errors = 0

                # Set network for training mode
                model.train(True)

                # Iterate mini batches to cover entire database
                for idx, data in enumerate(train_loader):

                    # Send data to device (important when using GPU)
                    train_x, train_label = data[0].to(device), data[1].to(device)

                    # Zeros gradient to prevent accumulation
                    sgd.zero_grad()

                    # Forward pass
                    predicted_labels = model(train_x.float())

                    # Calculate error
                    error = cross_error(predicted_labels, train_label.long())

                    # Sum wrong labels
                    # Error is given as percentage, need to multiply by mini batch size
                    epoch_total_errors += error.item()*train_label.shape[0]

                    # Backpropegate error
                    error.backward()

                    # Update network parameters
                    sgd.step()

                    # Print progress
                    if verbose:
                        precentile_done = round(100*(idx + 1)/iterations)
                        progress_symbols = int(floor(precentile_done*80/100))
                        print('\r['
                              + ('#')*progress_symbols
                              + (' ')*(80 - progress_symbols)
                              + ']' +
                              ' Epoch {}/{} progress {}/100%'.format(epoch + 1, epochs, precentile_done), end='')

                # Save epoch's accuracy for current technique
                # Errors count is normalized by database size and converted to percents
                train_results_dict[technique].append(
                    1 - (epoch_total_errors / train_samples_num) / 100)

                # Stop network training before evaluating performance over test data
                model.train(False)

                if verbose:
                    print('\r\nEpoch {} training done!'.format(epoch + 1))
                    print('Start testing')

                correct, sumv = test_models(model, test_loader, device)

                if verbose:
                    print('Accuracy achieved: {:.2f}%'.format(correct / sumv))
                    print('Time elapsed for epoch {:.2f} seconds'.format(
                        time.time()-start_time))
                    print('*'*80)

                # Save test accuracy for current epoch
                results_dict[technique].append(correct / sumv)

            if verbose:
                print('Accuracies for technique {}'.format(technique))
                print(results_dict[technique])

    # Print final results table
    print('*'*50)
    print('*'*16 + ' RESULTS  SUMMARY ' + '*'*16)
    print('*'*50)
    print('* technique *** train accuracy *** test accuracy *')
    print('*'*50)
    for technique in models_technique:
        print('* ' + technique + ' '*(10 - len(technique)) + '***\t{:.2f}'.format(train_results_dict[technique][-1]*100)
              + '%         *** {:.2f}'.format(results_dict[technique][-1]*100) + '%        *')
        print('*'*50)

        # When not in test mode - show graphs and save models
        if test_mode == '':
            fig, ax = plt.subplots()
            train_string = 'Train using {} technique'.format(technique)
            test_string = 'Test using {} technique'.format(technique)
            ax.plot(range(1, epochs + 1),
                    train_results_dict[technique], label=train_string)
            ax.plot(range(1, epochs + 1),
                    results_dict[technique], label=test_string)
            ax.set_xlabel('epochs')
            ax.set_ylabel('Accuracy')
            ax.set_title('Lenet5 results using {} technique'.format(technique))
            ax.legend()
            os.mkdir('models')
            torch.save(models[models_technique.index(technique)], './models/{}.pth'.format(technique))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-t', '--technique', type=str, default='all',
        help='Technique used for training, default <all>',
        choices=['all', 'none', 'dropout', 'bn', 'wd'])
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
        '-l', '--test_mode', type=str, default='', help='Load models from path for testing only')

    args = parser.parse_args()

    train_and_test(args.technique, args.batch_size, args.epochs,
                   args.data_path, args.verbose, args.test_mode)
