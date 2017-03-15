#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools
import logging

import numpy as np

import theano
import theano.tensor as T
from model.Model import Model


class DANModel(Model):
    """
    Modelo baseado no trabalho Unsupervised Domain Adaptation by Backpropagation e
        que recebe um batch composto pela mesma quantidade de exemplos do source e do tareget.
    """

    def __init__(self, input, sourceSupLabel, unsLabel, allLayers, optimizer, predictionSup, lossSup, lossUns,
                 supTrainMetrics=None, unsTrainMetrics=None, evalMetrics=None, testMetrics=None, mode=None):
        """
        :param input: list of tensors

        :param sourceSupLabel: list of tensor that represents the corrects annotated outputs from the source domain.

        :param unsLabel: list of tensor that represents th

        :param allLayersSource: all model layers from the source

        :param allLayersTarget: all model layers from the target

        :param optimizer: Optimizer.Optimizer

        :param predictionSup: It's the function which will responsible to predict labels

        :param lossSup: It's function which calculates tagger loss

        :param lossUnsSource: It's function which calculates domain classifier loss using examples from source

        :param lossUnsTarget: It's function which calculates domain classifier loss using examples from target

        :param supervisedTrainMetrics: list of Metric objects to be applied on the supervised training. Warning: metrics
            with the same will be updates as if it were a single metric

        :param unsupervisedTrainMetrics: list of Metric objects to be applied on the unsupervised training. Warning: metrics
            with the same will be updates as if it were a single metric

        :param evalMetrics: list of Metric objects to be applied on the evaluation dataset.

        :param testMetrics: list of Metric objects to be applied on test datasets.

        :param mode: compilation mode for Theano.
        """

        evalInput = input + [sourceSupLabel]
        predictionInput = evalInput

        super(DANModel, self).__init__(evalInput, evalInput, predictionInput, predictionSup, False,
                                       supTrainMetrics + unsTrainMetrics, evalMetrics, testMetrics,
                                       mode)

        self.log = logging.getLogger(__name__)
        self.__optimizer = optimizer

        self.__trainFuncs = []
        self.__trainingMetrics = []

        optimizerInput = optimizer.getInputTensors()

        trainableLayers = []

        # Set the supervised part of the trainning
        for l in allLayers:
            if l.isTrainable():
                trainableLayers.append(l)

        # List of inputs for training function.
        sourceFuncInput = input + [sourceSupLabel, unsLabel] + optimizerInput

        # Include the variables required by all training metrics in the output list of the training function.
        trainOutputs = []

        for m in supTrainMetrics + unsTrainMetrics:
            trainOutputs += m.getRequiredVariables()

        # Training updates are given by the optimizer object.
        updates = optimizer.getUpdates(lossSup + lossUns, trainableLayers)

        # Training function.
        self.__trainFunc = theano.function(inputs=sourceFuncInput, outputs=trainOutputs, updates=updates,
                                           mode=self.mode)

        self.__supTrainMetrics = supTrainMetrics
        self.__unsTrainMetrics = unsTrainMetrics

    def doEpoch(self, trainIterator, epoch, iteration, devIterator, evalPerIteration, callbacks):
        lr = self.__optimizer.getInputValues(epoch)

        self.log.info("Lr: %f" % lr[0])

        self.log.info({
            "epoch": epoch,
            "iteration": iteration,
            "learn_rate": lr[0]
        })

        supervisedIterator = trainIterator[0]
        unsupervisedIterator = trainIterator[1]

        for x, y in supervisedIterator:

            try:
                xTarget, yTarget = unsupervisedIterator.next()
            except:
                xTarget, yTarget = unsupervisedIterator.next()

            supBatchSize = len(x[0])
            unsBatchSize = len(x[0]) + len(xTarget[0])

            inputs = []

            for i in range(len(x)):
                inputs.append(np.concatenate((x[i], xTarget[i])))

            useY = self.evaluateFuncUseY()

            if useY:
                # Theano function receives 'y' as an input
                inputs.append(y[0])
                inputs.append(np.concatenate((y[1], yTarget[0])))

            inputs += lr

            self.callbackBatchBegin(inputs, callbacks)

            outputs = self.__trainFunc(*inputs)

            for m in self.__supTrainMetrics:
                numOuputs = len(m.getRequiredVariables())
                mOut = []
                for _ in xrange(numOuputs):
                    mOut.append(outputs.pop(0))

                    # Update metric values.
                    m.update(supBatchSize, *mOut)

            for m in self.__unsTrainMetrics:
                numOuputs = len(m.getRequiredVariables())
                mOut = []
                for _ in xrange(numOuputs):
                    mOut.append(outputs.pop(0))

                    # Update metric values.
                    m.update(unsBatchSize, *mOut)

            self.callbackBatchEnd(inputs, callbacks)

            # Development per-iteration evaluation
            if devIterator and evalPerIteration and (iteration % evalPerIteration) == 0:
                # Perform evaluation.
                self.__evaluate(devIterator, self.getEvaluationFunction(), self.__evalMetrics, epoch, iteration)
