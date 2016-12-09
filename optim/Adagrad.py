#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy

import theano
import theano.tensor as T

from optim.Optimizer import Optimizer


class Adagrad(Optimizer):
    """Adagrad optimizer
    """

    def __init__(self, lr=0.01, decay=0.0):
        super(Adagrad, self).__init__()
        self.lr = T.scalar(name="lr")
        self.lrValue = lr
        self.__sumsSqDefGrads = []
        self.__sumsSqStructGrads = []

        # Não entendi muito isto. Por quê o decay está sempre habilitado no Adagrad????
        # if decay == 0.0:
        #     decay = 1

        self.decay = decay

    def getInputTensors(self):
        return [self.lr]

    def getInputValues(self, nrEpochsDone):
        """
        :param nmEpochDone:
        :return: new value of learning rate.
        """
        lrValue = self.lrValue * (1 / (1 + self.decay * nrEpochsDone))

        return [lrValue]

    def getUpdates(self, cost, layers):
        # Lists of variables that store the sum of the squared historical
        # gradients for the parameters of all layers. Since some layers use
        # structured gradients, the historical gradients are also structured
        # and need special treatment. So, we need to store two lists of
        # historical gradients: one for default gradient parameters and another
        # for structured gradient parameters.
        sumsSqDefGrads = []
        for l in layers:
            # Default gradient parameters also follow a default AdaGrad update.
            params = l.getDefaultGradParameters()
            ssgs = []
            for param in params:
                ssgVals = numpy.zeros(param.get_value(borrow=True).shape,
                                      dtype=theano.config.floatX)
                ssg = theano.shared(value=ssgVals,
                                    name='sumSqGrads_' + param.name,
                                    borrow=True)
                ssgs.append(ssg)
            # For default gradient parameters, we do not need to store
            # parameters or historical gradients separated by layer. We
            # just store a list of parameters and historical gradients.
            sumsSqDefGrads += ssgs
        self.__sumsSqDefGrads = sumsSqDefGrads

        sumsSqStructGrads = []
        for l in layers:
            # Structured parameters also need structured updates for the
            # historical gradients. These updates are computed by each layer.
            params = l.getStructuredParameters()
            ssgs = []
            for param in params:
                ssgVals = numpy.zeros(param.get_value(borrow=True).shape,
                                      dtype=theano.config.floatX)
                ssg = theano.shared(value=ssgVals,
                                    name='sumSqGrads_' + param.name,
                                    borrow=True)
                ssgs.append(ssg)
            # For structured parameters, we need to store the historical
            # gradients separated by layer, since the updates of these
            # variables are performed by each layer.
            sumsSqStructGrads.append(ssgs)
        self.__sumsSqStructGrads = sumsSqStructGrads

        # Build list of updates.
        updates = []
        defaultGradParams = []

        # Get structured updates and default-gradient parameters from all layers.
        for (idx, l) in enumerate(layers):
            # Structured updates (embeddings, basically).
            ssgs = self.__sumsSqStructGrads[idx]
            updates += l.getUpdates(cost, self.lr, ssgs)
            # Default gradient parameters (all the remaining).
            defaultGradParams += l.getDefaultGradParameters()

        # Add updates for default-gradient parameters.
        grads = self.defaultGradParam(cost, defaultGradParams)

        # For numerical stability.
        fudgeFactor = 1e-10

        for param, grad, ssg in zip(defaultGradParams, grads, self.__sumsSqDefGrads):
            # Update of the sum of squared gradient.
            newSsg = ssg + grad * grad
            updates.append((ssg, newSsg))
            # Update of the parameter.
            newParam = param - self.lr * (grad / (fudgeFactor + T.sqrt(newSsg)))
            updates.append((param, newParam))

        return updates
