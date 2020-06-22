import torch.nn as nn
import torch
import math

# Encoder & Decoder classed as depicted in the paper
# for M1 algorithm we use two FC layers
# input layer dimension - 784
# hidden layer dimension - 600
# latent dimension - 50

class SslEncoder(nn.Module):
    def __init__(self):
        super(SslEncoder, self).__init__()
        self.fc1 = nn.Linear(784, 600)
        self.activation = nn.Softplus()
        self.fc2 = nn.Linear(600, 50)
        self.fc3 = nn.Linear(600, 50)

    def forward(self, input):
        output = self.fc1(input)
        output = self.activation(output)
        mu = self.fc2(output)
        # mu = self.activation(mu)
        sig = self.fc3(output)
        return mu, sig


class SslDecoder(nn.Module):
    def __init__(self):
        super(SslDecoder, self).__init__()
        self.fc1 = nn.Linear(50, 600)
        self.activation = nn.Softplus()
        self.fc2 = nn.Linear(600, 784)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.fc1(input)
        output = self.activation(output)
        output = self.fc2(output)
        output = self.sigmoid(output)
        return output

class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input):
        mu, sig = self.encoder(input)
        std = torch.exp(sig/2)
        eps = torch.randn_like(std)
        z = eps.mul(std).add(mu)

        predicted_x = self.decoder(z)
        return predicted_x, mu, sig