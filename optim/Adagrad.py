#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy

import theano
import theano.tensor as T

from optim.Optimizer import Optimizer


class Adagrad(Optimizer):
    """Adagrad optimizer
    """

    def __init__(self, lr=0.01, decay=1.0):
        super(Adagrad, self).__init__()
        self.lr = T.scalar(name="lr")
        self.lrValue = lr
        self.__sumsSqDefGrads = []
        self.__sumsSqStructGrads = []

        # @eraldo: não sei a razão do código abaixo.
        #          como estava, a estratégia NORMAL não funcionava.
        # if decay == 0.0:
        #     decay = 1.0

        self.decay = decay

    def getInputTensors(self):
        return [self.lr]

    def getInputValues(self, nrEpochsDone):
        """
        :param nmEpochDone:
        :return: new value of learning rate.
        """
        lrValue = self.lrValue
        if self.decay != 0:
            lrValue /= 1 + self.decay * nrEpochsDone
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
            sumsSqDefGrads.append(ssgs)
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
            sumsSqStructGrads.append(ssgs)
        self.__sumsSqStructGrads = sumsSqStructGrads

        # Build list of updates.
        updates = []
        defaultGradParams = []

        # For numerical stability.
        fudgeFactor = 1e-10

        # Get structured updates and default-gradient parameters from all layers.
        for l, defSsgs, structSsgs in zip(layers, self.__sumsSqDefGrads, self.__sumsSqStructGrads):
            # Multiply the default LR by the LR factor of this layer.
            lr = self.lr
            lrFactor = l.getLRFactor()
            if lrFactor is not None:
                lr = lr * lrFactor

            # Structured updates (embeddings, basically).
            updates += l.getUpdates(cost, lr, structSsgs)

            # Default gradient parameters (all the remaining).
            for param, ssg in zip(l.getDefaultGradParameters(), defSsgs):
                # Compute gradient for this param.
                grad = T.grad(cost, param)

                # Update of the sum of squared gradient.
                newSsg = ssg + grad * grad
                updates.append((ssg, newSsg))

                # Update of the parameter.
                newParam = param - lr * (grad / (fudgeFactor + T.sqrt(newSsg)))
                updates.append((param, newParam))

        return updates
