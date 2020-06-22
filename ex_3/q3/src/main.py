import models
import torch
from torchvision.datasets import mnist
from torch.nn import BCELoss
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from copy import deepcopy
from sklearn import svm
import argparse
from math import floor as floor
from joblib import dump, load
import time
import numpy as np
import matplotlib.pyplot as plt
import random
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create global model instanace
encoder = SslEncoder().to(device)
decoder = SslDecoder().to(device)
model = VAE(encoder, decoder).to(device)
svm = svm.SVC(kernel='rbf', gamma="auto")
num_of_classes = 10
torch.manual_seed(1337)


def test_model(vae, svm, test_loader, test_dataset):
    with torch.no_grad():
        testAccuracy = 0
        # Test SVM on test dataset
        for idx, test_data in enumerate(test_loader, 0):
            test_input = test_data[0].to(device)
            test_input = test_input.view(-1, 784)
            test_labels = test_data[1].numpy()

            pred_x, mu, sig = vae(test_input)
            pred_labels = svm.predict(mu.cpu())
            testBatchScore = sum((pred_labels - test_labels) == 0) / len(test_dataset)
            testAccuracy += testBatchScore

    return testAccuracy



def train_and_test(batch_size, epochs, labels, data_path, verbose, test_mode):

    # load FashionMNIST dataset from path or download it if it doesn't exist
    train_dataset = mnist.FashionMNIST(
        data_path, train=True, download=True, transform=ToTensor())
    test_dataset = mnist.FashionMNIST(
        data_path, train=False, download=True, transform=ToTensor())

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    loss = BCELoss(reduction='sum')

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, betas=(
        0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)

    if labels == 'all':
        num_of_labels = [100, 600, 1000, 3000]
    else:
        num_of_labels = [int(labels)]

    vae_losses = []

    best_loss = 1e10
    best_model = None

    if test_mode:
        testErrorVec = []
        vae = model
        vae.load_state_dict(torch.load('./models/vae.pth', device))
        vae.eval()
        for num in num_of_labels:
            tested_svm = load('./models/svm_{}.joblib'.format(num))
            testAccuracy = test_model(vae, tested_svm, test_loader, test_dataset)

            # print some logs
            testError = 1 - testAccuracy
            testErrorVec.append(round(100*100*testError)/100)
            print('SVM error for ' + str(num) +
                  ' labeled samples is: ' + str(round(100*100*testError)/100))

    else:
        # Train the VAE over all train inputs
        for epoch in range(epochs):
            start_time = time.time()

            for idx, data in enumerate(train_loader):

                # Send data to device (important when using GPU)
                train_x, train_label = data[0].to(
                    device), data[1].to(device)

                # Zeros gradient to prevent accumulation
                optimizer.zero_grad()

                train_x = train_x.view(-1, 784)

                predicted_x, mu, sig = model(train_x)

                error = loss(predicted_x, train_x)

                KLdiv = -0.5*torch.sum(sig - torch.exp(sig) - mu**2 + 1)

                total_error = error + KLdiv

                total_error.backward()

                optimizer.step()

                # Print progress
                if verbose:
                    precentile_done = round(100*(idx + 1)/len(train_loader))
                    progress_symbols = int(floor(precentile_done*80/100))
                    print('\r['
                          + ('#')*progress_symbols
                          + (' ')*(80 - progress_symbols)
                          + ']' +
                          ' Epoch {}/{} progress {}/100%'.format(epoch + 1, epochs, precentile_done), end='')

            if (total_error < best_loss):
                best_model = deepcopy(model.state_dict())
                best_loss = total_error

            if verbose:
                print('\nLoss achieved: {:.2f}'.format(
                    total_error))
                print('Time elapsed for epoch {:.2f} seconds'.format(
                    time.time()-start_time))
                print('*'*80)

            vae_losses.append(total_error)

        fig, ax = plt.subplots()
        ax.plot(range(1, epochs + 1), vae_losses)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title('VAE loss vs epochs')

        if not os.path.exists('models'):
            os.mkdir('models')

        # Save model to file
        torch.save(best_model, './models/vae.pth')

        # Gather labeled data
        testErrorVec = []
        for num in num_of_labels:
            labels_per_class = int(num/num_of_classes)
            labeledindices = []
            for i in range(num_of_classes):
                labeledindices += random.choices(np.where(train_dataset.targets == i)[
                    0], k=labels_per_class)
            sampler = torch.utils.data.SubsetRandomSampler(labeledindices)
            labeledTrainLoader = torch.utils.data.DataLoader(train_dataset, batch_size=num, shuffle=False,
                                                             sampler=sampler)

            # Train SVM using labeled data
            with torch.no_grad():
                for idx, labeled_data in enumerate(labeledTrainLoader, 0):
                    svm_input = labeled_data[0].to(device)
                    svm_input = svm_input.view(-1, 784)
                    svm_labels = labeled_data[1]
                    pred_x, mu, sig = model(svm_input)
                    svm.fit(mu.cpu(), svm_labels)

            testAccuracy = test_model(model, svm, test_loader, test_dataset)

            # print some logs
            testError = 1 - testAccuracy
            testErrorVec.append(round(100*100*testError)/100)
            print('SVM error for ' + str(num) +
                  ' labeled samples: ' + str(round(100*100*testError)/100) + '%')

            dump(svm, './models/svm_{}.joblib'.format(num))

    fig, ax = plt.subplots()
    ax.plot(num_of_labels, testErrorVec)
    ax.set_xlabel('Labeled data')
    ax.set_ylabel('Error %')
    ax.set_title('SSL error as function of number of labeled data')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-b', '--batch_size', type=int, default=64, help='Batch size, default <64>')
    parser.add_argument(
        '-e', '--epochs', type=int, default=30, help='Number of epochs, default <30>')
    parser.add_argument(
        '-l', '--labels', type=str, default='100', help='Number of epochs, default <100>',
        choices=['all', '100', '600', '1000', '3000'])
    parser.add_argument(
        '-p', '--data_path', type=str, default='./data/FashionMNIST',
        help='Path to database, default <"./data/FashionMNIST">')
    parser.add_argument(
        '-v', '--verbose', dest='verbose', action='store_true', default=False, help='Enable verbose mode')
    parser.add_argument(
        '-l', '--test_mode', dest='test_mode', action='store_true', default=False, help='Load models for testing only')

    args = parser.parse_args()

    train_and_test(args.batch_size, args.epochs, args.labels,
                   args.data_path, args.verbose, args.test_mode)
