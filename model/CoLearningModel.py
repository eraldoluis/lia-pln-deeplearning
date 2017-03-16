#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

import theano
from numpy import random

from model.Model import Model


class CoLearningModel(Model):
    def __init__(self, x, y, classifierLayers, optimizers, prediction, ls, lu, lossUnsEpoch, supervisedTrainMetrics,
                 unsupervisedTrainMetrics, evalMetrics, testMetrics, mode=None):
        """

        :param x: list of tensors that represents the inputs.
        :param y: list of tensor that represents the corrects outputs.

        :type classifierLayers: [[nnet.Layer]]
        :param classifierLayers: a list with the layers of each classifier

        :type optimizer: [Optimizer.Optimizer]
        :param optimizer: optimizers for each classifier

        :type prediction: T.var.TensorVariable
        :param prediction: It's the function which will responsible to predict labels

        :param ls: tensor that represents the supervised loss
        :param lu: tensor that represents the unsupervised loss
        :param lossUnsEpoch: the variable that controls when the lu will begin to be used.
        :param supervisedTrainMetrics:  list of Metric objects to be applied on the supervised training dataset.
        :param unsupervisedTrainMetrics: list of Metric objects to be applied on the unsupervised training dataset.
        :param evalMetrics: list of Metric objects to be applied on the evaluation dataset.
        :param testMetrics: list of Metric objects to be applied on test datasets.
        :param mode: compilation mode for Theano.
        """

        evalInput = x + [y]
        trainInputs = supervisedTrainMetrics + unsupervisedTrainMetrics

        super(CoLearningModel, self).__init__(evalInput, evalInput, x, prediction, False, trainInputs, evalMetrics,
                                              testMetrics, mode)

        self.log = logging.getLogger(__name__)
        self.__optimizers = optimizers

        self.__trainFuncs = []
        self.__trainingMetrics = []
        self.__lossUnsEpoch = lossUnsEpoch

        optimizerInput = optimizers[0].getInputTensors() + optimizers[1].getInputTensors()

        # Contains the layers of each classifier
        trainableLayers = []

        # Set the supervised part of the trainning
        for layers in classifierLayers:
            aux = []

            for l in layers:
                if l.isTrainable():
                    aux.append(l)

            trainableLayers.append(aux)

        if supervisedTrainMetrics:
            # List of inputs for training function.
            sourceFuncInput = x + [y] + optimizerInput

            # Include the ouputs of each metric in a list.
            supTrainOutputs = []

            for m in supervisedTrainMetrics:
                supTrainOutputs += m.getRequiredVariables()

            self.__trainingMetrics.append(supervisedTrainMetrics)

            # Training updates are given by the optimizer object. Each classifier has its own learning rate.
            updates = optimizers[0].getUpdates(ls, trainableLayers[0]) + optimizers[1].getUpdates(ls, trainableLayers[1])

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
            updates = optimizers[0].getUpdates(lu, trainableLayers[0]) + optimizers[1].getUpdates(lu,
                                                                                                  trainableLayers[1])

            # Training function.
            self.__trainFuncs.append(theano.function(inputs=targetFuncInput, outputs=unsTrainOutputs, updates=updates))

    def doEpoch(self, trainIterator, epoch, iteration, devIterator, evalPerIteration, callbacks):
        lrs = self.__optimizers[0].getInputValues(epoch) + self.__optimizers[1].getInputValues(epoch)

        self.log.info("Lr: [%f, %f]" % (lrs[0], lrs[1]))

        self.log.info({
            "epoch": epoch,
            "iteration": iteration,
            "learn_rate": lrs
        })

        trainBatchIterators = list(trainIterator)

        # if epoch value is less than the parameter lossUnsupervisedEpoch, so it isn't the time to apply the unsupervised training
        if epoch < self.__lossUnsEpoch:
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

            inputs += lrs

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
