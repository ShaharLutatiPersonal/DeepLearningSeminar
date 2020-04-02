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


def train_and_test(mode, batch_size, epoch, data_path, verbose):
    print('**************************************************************************************************************')
    print('executing modes: ' + mode + ',batch size:{} '.format(batch_size) +
          ',epoch:{} '.format(epoch) + ',data path = ' + data_path)
    print('**************************************************************************************************************')
    train_dataset = mnist.MNIST(
        data_path, train=True, download=False, transform=ToTensor())
    test_dataset = mnist.MNIST(
        data_path, train=False, download=False, transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    models = [Lenet5.NetOriginal(), Lenet5.NetD(), Lenet5.NetBN(),
              Lenet5.NetOriginal()]
    models_name = ['original', 'dropout',
                   'batch normalization', 'weight decay']
    if mode != 'all':
        ix = models_name.index(mode)
        models = [models[ix]]
        models_name = [models_name[ix]]
    orig_res, dropout_res, bn_res, weight_decay_res = [], [], [], []
    res_vs_name = {'original': orig_res, 'dropout': dropout_res,
                   'batch normalization': bn_res, 'weight decay': weight_decay_res}
    train_orig, train_drop, train_bn, train_wd = [], [], [], []
    train_res_vs_name = {'original': train_orig, 'dropout': train_drop,
                         'batch normalization': train_bn, 'weight decay': train_wd}
    num_of_samples_train_data = train_loader.dataset.train_data.shape[0]
    num_of_calls = len(train_loader)
    for name, model in zip(models_name, models):
        if name != 'weight decay':
            sgd = SGD(model.parameters(), lr=1e-1)
        else:
            sgd = SGD(model.parameters(), lr=1e-1, weight_decay=0.001)
        cross_error = CrossEntropyLoss()
        print('start trainig model: ' + name)
        for _epoch in range(epoch):
            start_time = time.time()
            model.train(True)
            sum_of_errors_in_epoch = 0
            for idx, (train_x, train_label) in enumerate(train_loader):
                label_np = np.zeros((train_label.shape[0], 10))
                sgd.zero_grad()  # in order to calculate fresh new chain rule derviative (otherwise take the previous value)
                predict_y = model(train_x.float())
                # taking cross entropy
                _error = cross_error(predict_y, train_label.long())
                # expalnation below:
                sum_of_errors_in_epoch += _error.item()*train_label.shape[0]
                _error.backward()
                sgd.step()
                precentile_done = round(100*idx/num_of_calls)
                if verbose:
                    print('\r ['+'%s' % ("#")*precentile_done+']' +
                          'progress in epoch trainig {}%'.format(precentile_done), end='')
            # the error given as the percentile of the error from the batch , so by multiplying by the batch size and divide it by 100
            # we get the integer number of error to be summed over the 60k DB, we divide here once by 100 and not every iteration for elegance
            # we don't multiply with fixed batch size for supporting different size batches
            if verbose:
                print()
            train_res_vs_name[name].append(
                1-(sum_of_errors_in_epoch/num_of_samples_train_data)/100)
            correct = 0
            sumv = 0
            # We take the mnist db and use the 60k for training and 10k for test
            # since we already use mini batch we assuring that we won't fall to local artifact
            # Important Change to eval mode for Dropout to clear out
            model.train(False)
            for idx, (test_x, test_label) in enumerate(test_loader):
                predict_y = model(test_x.float()).detach()
                predict_ys = np.argmax(predict_y, axis=-1)
                label_np = test_label.numpy()
                _est = predict_ys == test_label
                correct += np.sum(_est.numpy(), axis=-1)
                sumv += _est.shape[0]
            print('accuracy: {:.2f}'.format(
                correct / sumv) + ' epoch : {}'.format(_epoch+1))
            end_time = time.time()
            print('time elapsed {}'.format(end_time-start_time))
            print('**************************************************************************************************************')
            res_vs_name[name].append(correct / sumv)
        print(res_vs_name[name])
        #torch.save(model, 'models/mnist_{:.2f}.pkl'.format(correct / _sum))
    for name in models_name:
        print('Results for ' + name + ' Training accuracy')
        print(train_res_vs_name[name])
        print('**************************************************************************************************************')
        print('Results for ' + name + ' Testing accuracy')
        print(res_vs_name[name])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', type=str, help='mode to train - str , supported : all/original/dropout/batch normalization/weight decay , default = all')
    parser.add_argument(
        '-b', type=int, help='batch size - integer , default = 256')
    parser.add_argument(
        '-e', type=int, help='num of epochs - integer , default = 15')
    parser.add_argument(
        '-p', type=str, help='path to database - str , default = ./data/mnist')
    parser.add_argument(
        '-v', type=int, help='verbose indicator 1 - true,default = 1')
    args = parser.parse_args()
    # parse if not default values
    mode, batch_size, epoch, path, verbose = args.m, args.b, args.e, args.p, args.v
    if verbose == None:
        verbose = 1
    if mode == None:
        mode = 'all'
    if batch_size == None:
        batch_size = 256
    if epoch == None:
        epoch = 15
    if path == None:
        data_path = './data/mnist'
    train_and_test(mode, batch_size, epoch, data_path, verbose)
