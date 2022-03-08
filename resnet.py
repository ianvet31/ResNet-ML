import matplotlib.pyplot as plt
#import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
import random
import sklearn.datasets



class Block(nn.Module):
    """A basic block used to build ResNet."""

    def __init__(self, num_channels):
       
        super(Block, self).__init__()
        self.conv2d_layer = nn.Conv2d(num_channels, num_channels, (3, 3), 1, 1, bias=False)
        self.batchnorm_layer = nn.BatchNorm2d(num_channels)
        self.relu_layer = nn.ReLU()
        self.conv2d_layer2 = nn.Conv2d(num_channels, num_channels, (3, 3), 1, 1, bias=False)
        self.batchnorm_layer2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        
        f = self.conv2d_layer(x)
        f = self.batchnorm_layer(f)
        f = self.relu_layer(f)
        f = self.conv2d_layer2(f)
        f = self.batchnorm_layer2(f)
        return self.relu_layer(x + f)


class ResNet(nn.Module):
    """A simplified ResNet."""

    def __init__(self, num_channels, num_classes=10):
        
        super(ResNet, self).__init__()

        self.conv2d_layer1 = nn.Conv2d(1, num_channels, (3, 3), 2, 1, bias=False)
        self.batchnorm_layer1 = nn.BatchNorm2d(num_channels)
        self.relu_layer = nn.ReLU()
        self.maxpool_layer = nn.MaxPool2d((2, 2))
        self.block_layer = Block(num_channels)
        self.adaptive_avg_layer = nn.AdaptiveAvgPool2d(1)
        self.linear_layer = nn.Linear(num_channels, num_classes)


    def forward(self, x):
        """
        The input will have shape (N, 1, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have shape (N, 10).
        """
        
        f = self.conv2d_layer1(x)
        f = self.batchnorm_layer1(f)
        f = self.relu_layer(f)
        f = self.maxpool_layer(f)
        f = self.block_layer.forward(f)
        f = self.adaptive_avg_layer(f)
        f = f.view(f.size(dim=0), -1)
        f = self.linear_layer(f)
        return f
        
        
        
def plot_resnet_loss_1():
    
    sk_digits = sklearn.datasets.load_digits()
    (X, Y) = (torch.tensor(sk_digits.data).type(torch.float), torch.tensor(sk_digits.target))
    Y = Y.type(torch.LongTensor)
    print(X.shape, Y.shape, X.max(), X.min())
    X /= X.max()
    n = X.shape[0]
    perm = list(range(n))
    random.shuffle(perm)
    (X, Y) = ({'tr': X[perm[:n//2], ...], 'te': X[perm[n//2:], ...]}, {'tr':Y[perm[:n//2]], 'te':Y[perm[n//2:]]})
    mb_sz = 128
    stepsize = 0.1
    for (_, (net_s, num_channels)) in enumerate([
        ('ResNet_1', 1),
        ('ResNet_2', 2),
        ('ResNet_4', 4),
    ]):
        losses = { 'tr' : [], 'te' : [] }
        net =  ResNet(num_channels)
        optimizer = torch.optim.Adam(net.parameters())
        for i in range(4000):
            idxs = random.sample(range(X['tr'].shape[0]), mb_sz)
            (x, y) = (X['tr'][idxs, ...], Y['tr'][idxs])
            x = x.view(x.shape[0], 1, 8, 8)

            net.zero_grad()
            yhat = net(x)
            loss = torch.nn.CrossEntropyLoss()(yhat, y)
            loss.backward()
            optimizer.step()
   
            with torch.no_grad():
                if (i + 1) % 25 == 0:
                    x = X['te']
                    x = x.view(x.shape[0], 1, 8, 8)
                    yhat2 = net(x)
                    loss2 = torch.nn.CrossEntropyLoss()(yhat2, Y['te'])
                    print(f"{i} {loss:.3f} {loss2:.3f}")
                    losses['tr'].append(loss.detach())
                    losses['te'].append(loss2.detach())
        for s in ['tr', 'te']:
            plt.figure(1)
            plt.plot(range(len(losses[s])), losses[s],
                     label = f"{net_s} {s}")
                     
    plt.figure(1)
    plt.title("risk curves")
    plt.legend()
    plt.savefig('f1.pdf')
    plt.show()        



def plot_resnet_loss_2():
   
    sk_digits = sklearn.datasets.load_digits()
    (X, Y) = (torch.tensor(sk_digits.data).type(torch.float), torch.tensor(sk_digits.target))
    Y = Y.type(torch.LongTensor)
    print(X.shape, Y.shape, X.max(), X.min())
    X /= X.max()
    n = X.shape[0]
    perm = list(range(n))
    random.shuffle(perm)
    (X, Y) = ({'tr': X[perm[:n//2], ...], 'te': X[perm[n//2:], ...]}, {'tr':Y[perm[:n//2]], 'te':Y[perm[n//2:]]})
    mb_sz = 128
    stepsize = 0.1
    for (_, (net_s, num_channels)) in enumerate([
        ('ResNet_64', 64),
    ]):
        losses = { 'tr' : [], 'te' : [] }
        net =  ResNet(num_channels)
        optimizer = torch.optim.Adam(net.parameters())
        for i in range(4000):
            idxs = random.sample(range(X['tr'].shape[0]), mb_sz)
            (x, y) = (X['tr'][idxs, ...], Y['tr'][idxs])
            x = x.view(x.shape[0], 1, 8, 8)
            
            net.zero_grad()
            yhat = net(x)
            loss = torch.nn.CrossEntropyLoss()(yhat, y)
            loss.backward()
            optimizer.step()

            
            with torch.no_grad():
                if (i + 1) % 25 == 0:
                    x = X['te']
                    x = x.view(x.shape[0], 1, 8, 8)
                    yhat2 = net(x)
                    loss2 = torch.nn.CrossEntropyLoss()(yhat2, Y['te'])
                    print(f"{i} {loss:.3f} {loss2:.3f}")
                    losses['tr'].append(loss.detach())
                    losses['te'].append(loss2.detach())
        for s in ['tr', 'te']:
            plt.figure(1)
            plt.plot(range(len(losses[s])), losses[s],
                     label = f"{net_s} {s}")
                     
    plt.figure(1)
    plt.title("risk curves")
    plt.legend()
    plt.savefig('f2.pdf')
    plt.show()
