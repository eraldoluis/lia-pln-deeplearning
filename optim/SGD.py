#!/usr/bin/env python
# -*- coding: utf-8 -*-
import theano.tensor as T

from optim.Optimizer import Optimizer


class SGD(Optimizer):
    """Stochastic gradient descent, with support for momentum,
    learning rate decay, and Nesterov momentum.

    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter updates momentum.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    """

    def __init__(self, lr=0.01, decay=0.0):
        super(SGD, self).__init__()

        self.lr = T.scalar(name="lr")
        self.lrValue = lr
        self.decay = decay

    def getInputTensors(self):
        return [self.lr]

    def getInputValues(self, nrEpochsDone):
        lrValue = self.lrValue
        if self.decay != 0:
            lrValue /= 1 + self.decay * nrEpochsDone
        return [lrValue]

    def getUpdates(self, cost, layers):
        updates = []

        for l in layers:
            # Multiply the default LR by the LR factor of this layer.
            lr = self.lr
            lrFactor = l.getLRFactor()
            if lrFactor is not None:
                lr = lr * lrFactor

            # Structured updates (embeddings, basically).
            updates += l.getUpdates(cost, lr)

            # Default gradient parameters (all the remaining).
            for param in l.getDefaultGradParameters():
                grad = T.grad(cost, param)
                updates.append((param, param - lr * grad))

        return updates
