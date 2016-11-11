#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

import theano

from model.Model import Model
import numpy as np


class NegativeSamplingModel(Model):
    """
    This class trains a model using negative sampling and subsampling
    """

    def __init__(self, t, noiseRate, sampler, x, y, allLayers, optimizer, loss, trainMetrics):

        super(NegativeSamplingModel, self).__init__(None, None, None, None, trainMetrics=trainMetrics)

        self.__t = t
        self.__noiseRate = noiseRate
        self.__sampler = sampler
        self.log = logging.getLogger(__name__)
        self.__optimizer = optimizer

        # List of trainable layers.
        trainableLayers = []
        for l in allLayers:
            if l.isTrainable():
                trainableLayers.append(l)

        # List of inputs for training function.
        trainInputs = x + y + optimizer.getInputTensors()

        # Training updates are given by the optimizer object.
        updates = optimizer.getUpdates(loss, trainableLayers)

        # Include the variables required by all training metrics in the output list of the training function.
        trainOutputs = []

        for m in trainMetrics:
            trainOutputs += m.getRequiredVariables()

        # Training function.
        self.__trainFunction = theano.function(inputs=trainInputs, outputs=trainOutputs, updates=updates,
                                               mode=self.mode)

        self.__numExamples = 0

    def getCentralToken(self, correctWindow):
        centralIdx = int(len(correctWindow) / 2)
        return correctWindow[centralIdx]

    def generateNoiseExamples(self, correctWindow):
        windowToReturn = []

        centralPos = int(len(correctWindow) / 2)
        centralToken = correctWindow[centralPos]

        for _ in xrange(self.__noiseRate):
            noiseWindow = list(correctWindow)

            noiseTokenId = centralToken

            while noiseTokenId == centralToken:
                #  Search by token until it's different of central Token
                noiseTokenId = self.__sampler.sample()[0]

            noiseWindow[centralPos] = noiseTokenId

            windowToReturn.append(noiseWindow)

        return windowToReturn

    def doDiscard(self, tokenId):
        probability = self.__sampler.getProbability(tokenId)
        discardProbability = 1 - np.sqrt(self.__t / probability)
        discard = np.random.uniform() < discardProbability

        return discard

    def doEpoch(self, trainIterator, epoch, iteration, devIterator, evalPerIteration, callbacks):
        lr = self.__optimizer.getInputValues(epoch)

        self.log.info({
            "epoch": epoch,
            "iteration": iteration,
            "learn_rate": lr
        })

        for x, y in trainIterator:

            windowWords = []
            labels = []

            #  We raffle the noise examples during the training
            for correctedWindow in x[0]:
                centralToken = self.getCentralToken(correctedWindow)

                #  Subsampling randomly discards words.
                # The higher the frequency, more will increase the change of the word be discarted.
                # The equation aggressively subsamples words whose frequency is greater than t.
                if self.doDiscard(centralToken):
                    continue

                # Put correct examples and noises examples in a same batch
                windowWords.append(correctedWindow)
                labels.append(1)

                windowWords += self.generateNoiseExamples(correctedWindow)
                labels += [0] * self.__noiseRate

            # The real batch is num correct examples times noise rate
            batchSize = len(windowWords)

            # If subsampling discarded all examples, so continue the training
            if batchSize == 0:
                continue

            iteration += 1

            # List of input variables.
            inputs = [
                np.asarray(windowWords),
                np.asarray(labels),
                lr[0]
            ]

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
