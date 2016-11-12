#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

import theano

from model.Model import Model
import numpy as np


class NegativeSamplingModel(Model):
    """
    This class trains a model using negative sampling and subsampling.
    During the training, alpha will linearly drop to min_alpha.
    The lr update is done for each "numExUpdLr' examples read and ,to do that,
    we use following equation: lr =  startingLr * 1 - ( numberExamplesRead / (numEpochs * totalNumExamplesInDataset + 1) )
    """

    def __init__(self, t, noiseRate, sampler, minLr, numExUpdLr, totalExamples, numEpochs, x, y, allLayers, optimizer,
                 loss, trainMetrics):
        """

        :param t: Set threshold for occurrence of words. Those that appear with higher frequency in the training data "
                  will be randomly down-sampled;
        :param noiseRate: "Number of noise examples
        :param sampler: This object will raffle noise examples using a distribution
        :param minLr: this is the miminum value which the lr can have
        :param numExUpdLr: The lr update is done for each "numExUpdLr' examples read
        :param totalExamples: number of examples in the dataset
        :param numEpochs: total number of epochs
        :param x: list of tensors that represent the inputs.
        :param y: list of tensor that represents the corrects outputs.
        :param allLayers: all model layers

        :type optimizer: Optimizer.Optimizer
        :param optimizer:

        :param loss: tensor which represents the loss of the problem
        :param trainMetrics: list of Metric objects to be applied on the training dataset.
        """

        super(NegativeSamplingModel, self).__init__(None, None, None, None, trainMetrics=trainMetrics)

        self.__t = t
        self.__noiseRate = noiseRate
        self.__sampler = sampler
        self.log = logging.getLogger(__name__)
        self.__optimizer = optimizer

        # Setting linear decay parameters
        self.__minLr = minLr
        self.__startingLr = optimizer.getInputValues(0)[0]
        self.__numExamplesRead = 0
        self.__numExUpdLr = numExUpdLr
        self.__lr = self.__startingLr

        # Total number of examples that the model will read
        self.__totalExampleToRead = float(totalExamples * numEpochs)

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
        self.log.info({
            "epoch": epoch,
            "iteration": iteration,
        })

        for x, y in trainIterator:
            windowWords = []
            labels = []

            if self.__numExamplesRead % self.__numExUpdLr == 0:
                self.__lr = self.__startingLr * (1 - (self.__numExamplesRead / (self.__totalExampleToRead + 1)))

                if self.__lr < self.__minLr:
                    self.__lr = self.__minLr

                self.log.info({
                    "lr": self.__lr,
                    "num_examples_read": self.__numExamplesRead,
                })

            # We raffle the noise examples during the training
            for correctedWindow in x[0]:
                # Word2vec code counts the examples before sub-sampling
                self.__numExamplesRead += 1


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
                self.__lr
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
