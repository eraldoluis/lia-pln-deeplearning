#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools
import logging
import time
from numpy import random

import theano
import theano.tensor as T
from ModelOperation.Model import Metric, StopWatch, resetAllMetrics, Model


class CoLearningModel(Model):
    def __init__(self):
        super(CoLearningModel, self).__init__()

        self.log = logging.getLogger(__name__)

        self.__optimizers = None

        self.__trainFunction = None
        self.__evaluateFunction = None
        self.__predictionFunction = None

        self.__modelUnitsToTraining = []
        self.__modelUnitEvaluate = None

        self.trainingMetrics = []
        self.evaluateMetrics = []

        self.__theanoTrainFunction = []

    def addTrainingModelUnit(self, modelUnit, metrics=[]):
        self.__modelUnitsToTraining.append((modelUnit, metrics))

    def setEvaluatedModelUnit(self, modelUnit, metrics=[]):
        self.__modelUnitEvaluate = (modelUnit, metrics)

    def compile(self, optimizersAndLayers):
        allLayers = []
        self.__optimizers = []

        for optimizer, layers in optimizersAndLayers:
            self.__optimizers.append(optimizer)

        for modelUnit, metrics in self.__modelUnitsToTraining:
            funInputsTrain = []  # Inputs of the training function

            # Adding all layers to a same set
            funInputsTrain += modelUnit.x

            if modelUnit.yWillBeReceived:
                funInputsTrain.append(modelUnit.y)

            m = []
            _trainingOutputFunc = []  # Functions that will be calculated by theano

            for metricName in metrics:
                m.append(Metric(modelUnit.name, metricName))

                if metricName == "acc":
                    _trainingOutputFunc.append(T.mean(T.eq(modelUnit.prediction, modelUnit.y)))
                elif metricName == "loss":
                    _trainingOutputFunc.append(modelUnit.loss)
                    loss = modelUnit.loss

            self.trainingMetrics.append(m)
            updates = []

            for optimizer, d in optimizersAndLayers:
                layers = d[modelUnit]
                trainableLayers = []

                for l in layers:
                    if l.isTrainable():
                        trainableLayers.append(l)

                updates += optimizer.getUpdates(loss, trainableLayers)
                funInputsTrain += optimizer.getInputTensors()

            self.__theanoTrainFunction.append(
                theano.function(inputs=funInputsTrain, outputs=_trainingOutputFunc, updates=updates))

        funInputsEvaluate = []  # Inputs of the evaluation function
        _testOutputFunc = []  # Functions that will be calculated by theano

        if self.__modelUnitEvaluate is not None:
            modelUnit, metrics = self.__modelUnitEvaluate

            funInputsEvaluate += modelUnit.x

            funInputsPrediction = modelUnit.x
            _prediction = modelUnit.prediction

            if modelUnit.yWillBeReceived:
                funInputsEvaluate.append(modelUnit.y)

            for metricName in metrics:
                self.evaluateMetrics.append(Metric(modelUnit.name, metricName))

                if metricName == "acc":
                    _testOutputFunc.append(T.mean(T.eq(modelUnit.prediction, modelUnit.y)))
                elif metricName == "loss":
                    _testOutputFunc.append(modelUnit.loss)

        if self.__modelUnitEvaluate is not None:
            self.__evaluateFunction = theano.function(inputs=funInputsEvaluate, outputs=_testOutputFunc)
            self.__predictionFunction = theano.function(inputs=funInputsPrediction, outputs=_prediction)

    def getTrainingMetrics(self):
        l = []

        for m1 in self.trainingMetrics:
            for m2 in m1:
                l.append(m2)

        return l

    def getEvaluateMetrics(self):
        return self.evaluateMetrics

    def evaluateFuncUseY(self):
        modelUnit = self.__modelUnitEvaluate[0]

        return modelUnit.yWillBeReceived

    def getEvaluateFunction(self):
        return self.__evaluateFunction

    def doEpoch(self,trainBatchGenerators, epoch, callbacks):
        lr = []

        for optimizer in self.__optimizers:
            lr += optimizer.getInputValues(epoch)

        self.log.info("Lr: %s" % str(lr))

        trainBatchGeneratorsCp = list(trainBatchGenerators)
        t = {}

        for i, batchGen in enumerate(trainBatchGeneratorsCp):
            t[batchGen] = i

        while len(trainBatchGeneratorsCp) != 0:
            inputs = []
            batchSizes = {}

            i = random.randint(0, len(trainBatchGeneratorsCp))
            inputGenerator = trainBatchGeneratorsCp[i]

            try:
                _input = inputGenerator.next()
            except StopIteration:
                trainBatchGeneratorsCp.pop(i)
                continue

            idx = t[inputGenerator]

            modelUnitPlusMetrics = self.__modelUnitsToTraining[idx]
            modelUnit = modelUnitPlusMetrics[0]

            x, y = _input
            inputs += x

            batchSize = len(x[0])
            batchSizes[modelUnit.name] = batchSize

            if modelUnit.yWillBeReceived:
                # Theano function receives 'y' as an input
                inputs += y

            inputs += lr

            self.callbackBatchBegin(inputs, callbacks)

            outputs = self.__theanoTrainFunction[idx](*inputs)
            trainingMetrics = self.trainingMetrics[idx]

            for m, _output in itertools.izip(trainingMetrics, outputs):
                m.update(_output, batchSizes[m.modelUnitName])

            self.callbackBatchEnd(inputs, callbacks)


    # def evaluate(self, testBatchInterator, verbose=True):
    #     stopWatch = StopWatch()
    #     stopWatch.start()
    #
    #     resetAllMetrics(self.evaluateMetrics)
    #
    #     for x, y in testBatchInterator:
    #         batchSize = len(x[0])
    #
    #         inputs = []
    #         inputs += x
    #
    #         if self.__modelUnitEvaluate[0].yWillBeReceived:
    #             # Theano function receives 'y' as an input
    #             inputs += y
    #
    #         outputs = self.__evaluateFunction(*inputs)
    #
    #         for m, _output in itertools.izip(self.evaluateMetrics, outputs):
    #             m.update(_output, batchSize)
    #
    #     duration = stopWatch.lap()
    #
    #     logs = {}
    #
    #     for metric in self.evaluateMetrics:
    #         logs[metric.metricName] = metric.calculate()
    #
    #     if verbose:
    #         info = ""
    #         # Print information
    #         info += " [test: %ds]" % duration
    #
    #         for k, v in logs.iteritems():
    #             info += ' - %s:' % k
    #             info += ' %.6f' % v
    #
    #         self.log.info(info)
    #
    #     return logs
