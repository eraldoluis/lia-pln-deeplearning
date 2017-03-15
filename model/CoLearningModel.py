#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

import theano
from numpy import random

from model.Model import Model


class CoLearningModel(Model):
    def __init__(self, x, y, allLayers, optimizer, prediction, ls, lu, lossUnsEpoch, supervisedTrainMetrics,
                 unsupervisedTrainMetrics, evalMetrics, testMetrics, mode=None):

        evalInput = x + [y]
        trainInputs = supervisedTrainMetrics + unsupervisedTrainMetrics

        super(CoLearningModel, self).__init__(evalInput, evalInput, x, prediction, True, trainInputs, evalMetrics,
                                              testMetrics, mode)

        self.log = logging.getLogger(__name__)
        self.__optimizer = optimizer

        self.__trainFuncs = []
        self.__trainingMetrics = []
        self.__lossUnsEpoch = lossUnsEpoch

        optimizerInput = optimizer.getInputTensors()

        trainableLayers = []

        # Set the supervised part of the trainning
        for l in allLayers:
            if l.isTrainable():
                trainableLayers.append(l)

        if supervisedTrainMetrics:
            # List of inputs for training function.
            sourceFuncInput = x + [y] + optimizerInput

            # Include the ouputs of each metrics in a list.
            supTrainOutputs = []

            for m in unsupervisedTrainMetrics:
                supTrainOutputs += m.getRequiredVariables()

            self.__trainingMetrics.append(supervisedTrainMetrics)

            # Training updates are given by the optimizer object.
            updates = optimizer.getUpdates(ls, trainableLayers)

            # Training function.
            self.__trainFuncs.append(
                theano.function(inputs=sourceFuncInput, outputs=supTrainOutputs, updates=updates, mode=self.mode))

        if unsupervisedTrainMetrics:
            # List of inputs for training function.
            targetFuncInput = x + optimizerInput

            # Include the variables required by all training metrics in the output list of the training function.
            unsTrainOutputs = []

            for m in unsupervisedTrainMetrics:
                unsTrainOutputs += m.getRequiredVariables()

            self.__trainingMetrics.append(unsupervisedTrainMetrics)

            # Training updates are given by the optimizer object.
            updates = optimizer.getUpdates(lu, trainableLayers)

            # Training function.
            self.__trainFuncs.append(theano.function(inputs=targetFuncInput, outputs=unsTrainOutputs, updates=updates))

    def doEpoch(self, trainIterator, epoch, iteration, devIterator, evalPerIteration, callbacks):
        lr = self.__optimizer.getInputValues(epoch)

        self.log.info("Lr: %f" % lr[0])

        self.log.info({
            "epoch": epoch,
            "iteration": iteration,
            "learn_rate": lr[0]
        })

        trainBatchIterators = list(trainIterator)

        # if epoch value is less than the parameter lossUnsupervisedEpoch, so it isn't the time to apply the unsupervised training
        if epoch < self.__lossUnsupervisedEpoch:
            # Discard BatchIterator with non-annotated examples.
            trainBatchIterators.pop()
            self.log.info("Não lendo exemplos não supervisionados ")
        else:
            self.log.info("Lendo exemplos não supervisionados ")

        # Keep the index of each batch iterator. I'll need this to know example origin
        indexBybatchIterator = {}

        for i, batchGen in enumerate(trainBatchIterators):
            indexBybatchIterator[batchGen] = i

        # Training iteration
        while len(trainBatchIterators) != 0:
            inputs = []

            i = random.randint(0, len(trainBatchIterators))
            batchIterator = trainBatchIterators[i]

            try:
                _input = batchIterator.next()
            except StopIteration:
                trainBatchIterators.pop(i)
                continue

            # Get the index of the batchIterator. If batchIteratorIdx == 0 so batchIterator contains annotated examples.
            # If batchIteratorIdx == 1 so batchIterator doesn't contain annotated examples.
            batchIteratorIdx = indexBybatchIterator[batchIterator]

            x, y = _input
            batchSize = len(x[0])
            inputs += x

            # If the examples is not anotated, so there isn't label
            if batchIteratorIdx == 0:
                inputs += y

            inputs += lr

            # CallbackBegin
            self.callbackBatchBegin(inputs, callbacks)

            # Train the NN. trainFuncs[0] = supervised training; trainFuncs[1] = unsupervised training
            outputs = self.__trainFuncs[batchIteratorIdx](*inputs)

            # Update the metrics. trainingMetrics[0] = supervised metrics; trainingMetrics[1] = unsupervised metrics
            for m in self.__trainingMetrics[batchIteratorIdx]:
                numOuputs = len(m.getRequiredVariables())
                mOut = []
                for _ in xrange(numOuputs):
                    mOut.append(outputs.pop(0))

                    # Update metric values.
                    m.update(batchSize, *mOut)

            # CallbackEnd
            self.callbackBatchEnd(inputs, callbacks)

            # Development per-iteration evaluation
            if devIterator and evalPerIteration and (iteration % evalPerIteration) == 0:
                # Perform evaluation.
                self.__evaluate(devIterator, self.getEvaluationFunction(), self.__evalMetrics, epoch, iteration)
