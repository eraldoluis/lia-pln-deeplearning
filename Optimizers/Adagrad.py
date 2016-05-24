#!/usr/bin/env python
# -*- coding: utf-8 -*-
from Optimizers.Optimizers import Optimizer


class Adagrad(Optimizer):
    '''Adagrad optimizer.

    It is recommended to leave the parameters of this optimizer
    at their default values.

    # Arguments
        lr: float >= 0. Learning rate.
        epsilon: float >= 0.
    '''
    def __init__(self, lr=0.01, epsilon=1e-6, *args, **kwargs):
        super(Adagrad, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.lr = K.variable(lr)

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        # accumulators
        self.weights = [K.variable(np.zeros(K.get_value(p).shape)) for p in params]
        self.updates = []

        for p, g, a in zip(params, grads, self.weights):
            new_a = a + K.square(g)  # update accumulator
            self.updates.append((a, new_a))
            new_p = p - self.lr * g / K.sqrt(new_a + self.epsilon)
            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append((p, new_p))
        return self.updates

    def get_config(self):
        return {"name": self.__class__.__name__,
                "lr": float(K.get_value(self.lr)),
                "epsilon": self.epsilon}