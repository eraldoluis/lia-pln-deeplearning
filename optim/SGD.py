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
        lrValue = self.lrValue * (1 / (1 + self.decay * nrEpochsDone))

        return [lrValue]

    def getUpdates(self, cost, layers):
        updates = []
        defaultGradParams = []

        for l in layers:
            # Structured updates (embeddings, basically).
            updates += l.getUpdates(cost, self.lr)
            # Default gradient parameters (all the remaining).
            defaultGradParams += l.getDefaultGradParameters()

        # Add updates for default-gradient parameters.
        # Compute gradient of the cost function w.r.t. each parameter.
        grads = self.defaultGradParam(cost, defaultGradParams)

        updates += [(param, param - self.lr * grad)
                   for param, grad in zip(defaultGradParams, grads)]

        return updates
