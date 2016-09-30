#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

import itertools
from __builtin__ import set

from numpy import random

import theano
import theano.tensor as T

from ModelOperation.Model import Model, Metric


class ConcatenatorDataset(object):
    '''
    Treats a set of examples from differents dataset as unique dataset
    '''

    def __init__(self, listSyncBatchList):
        """
        :type inputGenerators: list[DataOperation.InputGenerator.BatchIterator.SyncBatchList]
        :param listSyncBatchList: list that contains SyncBatchList from every dataset
        """
        self.__list = listSyncBatchList

        total = 0
        self.__ranges = []

        for d in self.__list:
            begin = total

            self.__ranges.append((begin, begin + d.size() - 1))

            total += d.size()

        self.__total = total

    def getSize(self):
        return self.__total

    def getRandomly(self):
        i = random.randint(0, self.getSize())

        for idx, range in enumerate(self.__ranges):
            if range[0] <= i <= range[1]:
                return idx, self.__list[idx].get(i - range[0])


class GradientReversalModel(Model):
    def __init__(self, sourceInput, targetInput, sourceSupLabel, sourceUnsLabel, targetLabel):
        super(GradientReversalModel, self).__init__()

        self.__trainingFuncs = []

        self.log = logging.getLogger(__name__)

        self.__isY_ProducedByNN = False
        self.trainingMetrics = [[], [], []]
        self.evaluateMetrics = []

        self.__sourceInput = sourceInput
        self.__targetInput = targetInput

        self.__sourceSupLabel = sourceSupLabel
        self.__sourceUnsLabel = sourceUnsLabel
        self.__targetLabel = targetLabel

        self.evaluateFunction = None
        self.__predictionFunction = None
        self.__optimizer = None
        self.__unsupervisedGenerator = None

    def compile(self, allLayersSource, allLayersTarget, optimizer, predictionSup,
                unsupervisedPredSource, unsupervisedPredTarget, lossSup, lossUnsSource, lossUnsTarget):
        self.__optimizer = optimizer

        _outputSourceFuncTrain = []
        _outputTargetFuncTrain = []

        # _outputSourceFuncTrain.append(loss)
        # self.trainingMetrics.append(Metric("", "loss"))

        lossUnsMetric = Metric("", "loss_unsup")
        accUnsMetric = Metric("", "acc_unsup")

        # Source Metrics
        _outputSourceFuncTrain.append(lossSup)
        self.trainingMetrics[0].append(Metric("", "loss_sup"))

        _outputSourceFuncTrain.append(T.mean(T.eq(predictionSup, self.__sourceSupLabel)))
        self.trainingMetrics[0].append(Metric("", "acc_sup"))

        _outputSourceFuncTrain.append(lossUnsSource)
        self.trainingMetrics[0].append(lossUnsMetric)

        _outputSourceFuncTrain.append(T.mean(T.eq(unsupervisedPredSource, self.__sourceUnsLabel)))
        self.trainingMetrics[0].append(accUnsMetric)

        # Target metrics
        _outputTargetFuncTrain.append(lossUnsTarget)
        self.trainingMetrics[1].append(lossUnsMetric)

        _outputTargetFuncTrain.append(T.mean(T.eq(unsupervisedPredTarget, self.__targetLabel)))
        self.trainingMetrics[1].append(accUnsMetric)

        # Eval metrics
        _outputFuncEval = []

        _outputFuncEval.append(T.mean(T.eq(predictionSup, self.__sourceSupLabel)))
        self.evaluateMetrics.append(Metric("", "acc"))

        # Removes not trainable layers from update and see if the output of the
        trainableSourceLayers = []

        for l in allLayersSource:
            if l.isTrainable():
                trainableSourceLayers.append(l)

        trainableTargetLayers = []

        for l in allLayersTarget:
            if l.isTrainable():
                trainableTargetLayers.append(l)

        optimizerInput = optimizer.getInputTensors()

        # Create theano functions
        self.__trainFuncs = []

        sourceFuncInput = self.__sourceInput + [self.__sourceSupLabel, self.__sourceUnsLabel] + optimizerInput
        self.__trainFuncs.append(theano.function(inputs=sourceFuncInput, outputs=_outputSourceFuncTrain,
                                                 updates=optimizer.getUpdates(lossSup + lossUnsSource,
                                                                              trainableSourceLayers)))

        targetFuncInput = self.__targetInput + [self.__targetLabel] + optimizerInput
        self.__trainFuncs.append(theano.function(inputs=targetFuncInput, outputs=_outputTargetFuncTrain,
                                                 updates=optimizer.getUpdates(lossUnsTarget, trainableTargetLayers)))

        self.evaluateFunction = theano.function(self.__sourceInput + [self.__sourceSupLabel], outputs=_outputFuncEval)
        self.__predictionFunction = theano.function(self.__sourceInput, outputs=predictionSup)

    def getTrainingMetrics(self):
        metrics = []

        for l in self.trainingMetrics:
            for m2 in l:
                if not m2 in metrics:
                    metrics.append(m2)

        return metrics

    def getEvaluateMetrics(self):
        return self.evaluateMetrics

    def evaluateFuncUseY(self):
        return not self.__isY_ProducedByNN

    def getEvaluateFunction(self):
        return self.evaluateFunction

    def doEpoch(self, trainBatchGenerators, epoch, callbacks):
        lr = self.__optimizer.getInputValues(epoch)

        self.log.info("Lr: %f" % lr[0])

        trainingExamples = ConcatenatorDataset(trainBatchGenerators)

        for i in xrange(trainingExamples.getSize()):
            idx, example = trainingExamples.getRandomly()
            x, y = example

            batchSize = len(x[0])

            inputs = []
            inputs += x

            useY = self.evaluateFuncUseY()

            if useY:
                # Theano function receives 'y' as an input
                inputs += y

            inputs += lr

            self.callbackBatchBegin(inputs, callbacks)

            outputs = self.__trainFuncs[idx](*inputs)

            trainingMetrics = self.trainingMetrics[idx]

            for m, _output in itertools.izip(trainingMetrics, outputs):
                m.update(_output, batchSize)

            self.callbackBatchEnd(inputs, callbacks)
