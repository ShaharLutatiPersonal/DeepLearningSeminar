import Lenet5
import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import numpy as np
import argparse
import time

models = [Lenet5.NetOriginal(), Lenet5.NetD(), Lenet5.NetBN(),
          Lenet5.NetOriginal()]

models_technique = ['none', 'dropout', 'bn', 'wd']


def train_and_test(mode, batch_size, epochs, data_path, verbose):
    # Start training network
    print('*'*80)
    print('Executing technique: ' + mode + ', batch size:{} '.format(batch_size) +
          ', epochs:{} '.format(epochs) + ', data path:{}'.format(data_path))
    print('*'*80)

    # load FashionMNIST dataset from path or download it if it doesn't exist
    train_dataset = mnist.FashionMNIST(
        data_path, train=True, download=True, transform=ToTensor())
    test_dataset = mnist.FashionMNIST(
        data_path, train=False, download=True, transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    if mode != 'all':
        ix = models_technique.index(mode)
        models = [models[ix]]
        models_technique = [models_technique[ix]]

    orig_res, dropout_res, bn_res, wd_res = [], [], [], []
    results_dict = {'none': orig_res, 'dropout': dropout_res,
                    'bn': bn_res, 'wd': wd_res}

    train_orig, train_drop, train_bn, train_wd = [], [], [], []
    train_results_dict = {'none': train_orig, 'dropout': train_drop,
                          'bn': train_bn, 'wd': train_wd}

    train_samples_num = train_loader.dataset.train_data.shape[0]

    # Iteration per epoch
    iterations = len(train_loader)

    # Train selected technique
    for technique, model in zip(models_technique, models):
        wd = 0
        if technique == 'wd':
            wd = 0.001

        sgd = SGD(model.parameters(), lr=1e-1, weight_decay=wd)

        cross_error = CrossEntropyLoss()

        print('Start training model: ' + technique)

        ### Iterate database the number of epochs
        for _epoch in range(epochs):
            start_time = time.time()
            epoch_total_errors = 0

            ### Set network for training mode
            model.train(True)

            ### Iterate mini batches to cover entire database
            for idx, (train_x, train_label) in enumerate(train_loader):
                # Zeros gradient to prevent accumulation
                sgd.zero_grad()

                # Forward pass
                predicted_labels = model(train_x.float())

                # Calculate error
                _error = cross_error(predicted_labels, train_label.long())

                # Sum wrong labels
                epoch_total_errors += _error.item()*train_label.shape[0]

                # Backpropegate error
                _error.backward()

                # Update network parameters
                sgd.step()

                # Print progress
                if verbose:
                    precentile_done = round(100*idx/iterations)
                    print('\r ['+'%s' % ("#")*precentile_done+']' +
                          'progress in epoch training {}%'.format(precentile_done), end='')

            # the error given as the percentile of the error from the batch, so by multiplying by the batch size and divide it by 100
            # we get the integer number of error to be summed over the 60k DB, we divide here once by 100 and not every iteration for elegance
            # we don't multiply with fixed batch size for supporting different size batches
            train_results_dict[technique].append(
                1 - (epoch_total_errors / train_samples_num) / 100)
            correct = 0
            sumv = 0

            # We take the mnist db and use the 60k for training and 10k for test
            # since we already use mini batch we assuring that we won't fall to local artifact
            # Important Change to eval mode for Dropout to clear out
            model.train(False)

            ### 
            for idx, (test_x, test_label) in enumerate(test_loader):
                predicted_labels = model(test_x.float()).detach()
                predict_ys = np.argmax(predicted_labels, axis=-1)
                label_np = test_label.numpy()
                _est = predict_ys == test_label
                correct += np.sum(_est.numpy(), axis=-1)
                sumv += _est.shape[0]

            print('accuracy: {:.2f}'.format(
                correct / sumv) + ' epoch : {}'.format(_epoch+1))

            end_time = time.time()
            print('time elapsed {}'.format(end_time-start_time))
            print('*'*80)
            results_dict[technique].append(correct / sumv)

        print(results_dict[technique])
        #torch.save(model, 'models/mnist_{:.2f}.pkl'.format(correct / _sum))

    # Print final results
    for technique in models_technique:
        print('*'*80)
        print('Results for ' + technique + ' training accuracy')
        print(train_results_dict[technique] + '%')
        print('Results for ' + technique + ' testing accuracy')
        print(results_dict[technique] + '%')
        print('*'*80)


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
        '-p', '--data_path', type=str, default='./data/mnist', help='Path to database, default <"./data/mnist">')
    parser.add_argument(
        '-v', '--verbose', dest='verbose', action='store_true', default=False, help='Enable verbose mode')

    args = parser.parse_args()

    train_and_test(args.technique, args.batch_size, args.epochs,
                   args.data_path, args.verbose)
