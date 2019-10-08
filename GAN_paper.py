#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Created on 06/10/2019 22:59 
@Author: XinZhi Yao 
"""

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable

torch.manual_seed(1)  # reproducible
np.random.seed(1)

# Hyper Paramters
BATCH_SIZE = 64
TRAINING_NUMBERS = 10000
D_TRAINING_STEPS = 20
LR_G = 0.0001  # learning rate for generator
LR_D = 0.0001  # learning rate for discriminator
N_IDEAS = 5  # think of this as number of ideas for generating an art work
ART_COMPONENTS = 15  # it could be total point G can draw in the canvas
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])


#show our beautiful painting range
# plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
# plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
# plt.legend(loc='upper right')
# plt.show()

def artist_works():     # painting from the famous artist (real target)
    a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
    paintings = a * np.power(PAINT_POINTS, 2) + (a-1)
    paintings = torch.from_numpy(paintings).float()
    return paintings


G_net = nn.Sequential(
    nn.Linear(N_IDEAS, 128),
    nn.ReLU(),
    nn.Linear(128, ART_COMPONENTS),
)

D_net = nn.Sequential(
    nn.Linear(ART_COMPONENTS, 128),
    nn.ReLU(),
    nn.Linear(128, 1),  # 0/1
    nn.Sigmoid(),
)

opt_G = optim.Adam(G_net.parameters(), lr=LR_G)
opt_D = optim.Adam(D_net.parameters(), lr=LR_D)

plt.ion()   # something about continuous plotting

# start to learn
for step in range(TRAINING_NUMBERS):
    for D_step in range(D_TRAINING_STEPS):

        artist_paintings = artist_works()  # real target
        ## G_net fixed
        G_ideas = torch.randn(BATCH_SIZE, N_IDEAS)  # normal distribution
        G_paintings = G_net(G_ideas)

        prob_artist0 = D_net(artist_paintings) # how many data from real target
        prob_artist1 = D_net(G_paintings)  # how many data from real target

        # Discriminator wants to maximize prob_artist0 and minimize prob_artist0
        # This means maximizing D_loss, so we need a negative to minimize D_loss
        D_loss = - torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))

        # Update D_net
        opt_D.zero_grad()
        D_loss.backward()
        opt_D.step()

    # not resamples
    # # sample minibatch of m noise samples from noise prior pg(z)
    G_ideas = torch.randn(BATCH_SIZE, N_IDEAS)  # normal distribution
    G_paintings = G_net(G_ideas)
    prob_artist2 = D_net(G_paintings)

    # The generator wants to maximize prob_artist1
    # This means to minmize G_loss
    G_loss = torch.mean(torch.log(1. - prob_artist2))

    # Update G_net
    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()

    if step % 50 == 0:  # plotting
        plt.cla()
        plt.plot(PAINT_POINTS[0], G_paintings.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting',)
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
        plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
        plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % prob_artist0.data.numpy().mean(), fontdict={'size': 13})
        plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 13})
        plt.ylim((0, 3));plt.legend(loc='upper right', fontsize=10);plt.draw();plt.pause(0.01)

plt.ioff()
plt.show()