#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

import theano

from model.Model import Model


class BasicModel(Model):
    def __init__(self, x, y, allLayers, optimizer, prediction, loss, yExist=False, trainMetrics=None, evalMetrics=None,
                 testMetrics=None, mode=None):
        """
            :param x: list of tensors that represent the inputs.

            :param y: list of tensor that represents the corrects outputs.

            :type allLayers: [nnet.Layer]
            :param allLayers: all model layers

            :type optimizer: Optimizer.Optimizer
            :param optimizer:

            :type prediction: T.var.TensorVariable
            :param prediction: It's the function which will responsible to predict labels

            :param yExist: This parameter is true when the learner produce your own correct output
                                            or use the input as the correct output, like DA.

            :param trainMetrics: list of Metric objects to be applied on the training dataset.

            :param evalMetrics: list of Metric objects to be applied on the evaluation dataset.

            :param testMetrics: list of Metric objects to be applied on test datasets.

            :param mode: compilation mode for Theano.
            """
        self.__x = x if isinstance(x, (set, list)) else [x]
        self.__y = y if isinstance(y, (set, list)) else [y]
        inputs = self.__x[:] if yExist else self.__x + self.__y

        super(BasicModel, self).__init__(inputs, inputs, self.__x, prediction, yExist, trainMetrics, evalMetrics,
                                         testMetrics, mode)

        self.log = logging.getLogger(__name__)
        self.__optimizer = optimizer

        # List of trainable layers.
        trainableLayers = []
        for l in allLayers:
            if l.isTrainable():
                trainableLayers.append(l)

        if trainMetrics:
            # List of inputs for training function.
            trainInputs = inputs + optimizer.getInputTensors()

            # Training updates are given by the optimizer object.
            updates = optimizer.getUpdates(loss, trainableLayers)

            # Include the variables required by all training metrics in the output list of the training function.
            trainOutputs = []
            for m in trainMetrics:
                trainOutputs += m.getRequiredVariables()

            # Training function.
            self.__trainFunction = theano.function(inputs=trainInputs, outputs=trainOutputs, updates=updates,
                                                   mode=self.mode)

    def doEpoch(self, trainIterator, epoch, iteration, devIterator, evalPerIteration, callbacks):
        lr = self.__optimizer.getInputValues(epoch)

        self.log.info({
            "epoch": epoch,
            "iteration": iteration,
            "learn_rate": lr
        })

        for x, y in trainIterator:
            iteration += 1

            if y[0].ndim > 0:
                batchSize = len(y[0])
            else:
                batchSize = 1

            # List of input variables.
            inputs = []
            inputs += x
            if self.evaluateFuncUseY():
                # Theano function receives 'y' as an input.
                inputs += y
            inputs += lr

            # Callbacks.
            self.callbackBatchBegin(inputs, callbacks)

            # Call training function.
            outputs = self.__trainFunction(*inputs)

            # Update training metrics.
            for m in self.getTrainMetrics():
                # Build metric required variable list.
                numOuputs = len(m.getRequiredVariables())
                mOut = []
                for _ in xrange(numOuputs):
                    mOut.append(outputs.pop(0))

                # Update metric values.
                m.update(batchSize, *mOut)

            # Callbacks.
            self.callbackBatchEnd(inputs, callbacks)

            # Development per-iteration evaluation
            if devIterator and evalPerIteration and (iteration % evalPerIteration) == 0:
                # Perform evaluation.
                self.__evaluate(devIterator, self.getEvaluationFunction(), self.__evalMetrics, epoch, iteration)

        return iteration
