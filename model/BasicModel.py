#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools
import logging

import theano
import theano.tensor as T

from model.Model import Metric, Model, StopWatch


class BasicModel(Model):
    def __init__(self, x, y, yExist=False, evalPerIteration=0, devBatchIterator=None, mode=None):
        """
        :param x: list of tensors that represent the inputs.

        :param y: list of tensor that represents the corrects outputs.

        :param yExist: This parameter is true when the learner produce your own correct output
                                        or use the input as the correct output, like DA.

        :param outputLayer: a list of outputs

        :param evalPerIteration: number of iterations before each evaluation. If it is equal to zero, do not perform
                                 more than one evaluation per epoch.

        :param devBatchIterator: iterator over development examples for per-iteration evaluation.
        """
        super(BasicModel, self).__init__(mode)

        self.__theanoFunction = None
        self.log = logging.getLogger(__name__)
        self.__calculateAcc = False
        self.__metrics = []
        self.__isY_ProducedByNN = yExist
        self.trainingMetrics = []
        self.evaluateMetrics = []
        self.__evalPerIteration = evalPerIteration
        self.__devBatchIterator = devBatchIterator

        if not isinstance(x, (set, list)):
            self.__x = [x]
        else:
            self.__x = x

        if not isinstance(y, (set, list)):
            self.__y = [y]
        else:
            self.__y = y

        self.__loss = None
        self.__inputs = None
        self.__prediction = None
        self.__layers = None
        self.__trainFunction = None
        self.evaluateFunction = None
        self.__predictionFunction = None
        self.__optimizer = None

    def getTrainingMetrics(self):
        return self.trainingMetrics

    def getEvaluateMetrics(self):
        return self.evaluateMetrics

    def evaluateFuncUseY(self):
        return not self.__isY_ProducedByNN

    def getEvaluateFunction(self):
        return self.evaluateFunction

    def getPredictionFunction(self):
        return self.__predictionFunction

    def compile(self, allLayers, optimizer, predictionFunction, lossFunction, metrics=[]):
        """
        :type allLayers: [ NNet.Layer.Layer]
        :param allLayers: all model layers

        :type optimizer: Optimizer.Optimizer
        :param optimizer:

        :type predictionFunction: T.var.TensorVariable
        :param predictionFunction: It's the function which will responsible to predict labels

        :type lossFunction: T.var.TensorVariable
        :param lossFunction: It's the function which will calculate the loss


        :param metrics: the names of the metrics to be measured. Nowadays, this classe just accepted "loss" and "acc"(accuracy)
        """
        self.__optimizer = optimizer
        self.__prediction = predictionFunction

        self.__metrics += metrics

        _outputFunc = []
        for metricName in self.__metrics:
            if metricName == "acc":
                self.__calculateAcc = True
                _outputFunc.append(T.mean(T.eq(self.__prediction, self.__y[0])))
            elif metricName == "loss":
                self.__loss = lossFunction
                _outputFunc.append(self.__loss)

            self.trainingMetrics.append(Metric("", metricName))
            self.evaluateMetrics.append(Metric("", metricName))

        # Removes not trainable layers from update and see if the output of the
        trainableLayers = []

        for l in allLayers:
            if l.isTrainable():
                trainableLayers.append(l)

        # Create the inputs of which theano function
        inputsOutputs = []
        inputsOutputs += self.__x

        if not self.__isY_ProducedByNN:
            inputsOutputs += self.__y

        funInputs = inputsOutputs + optimizer.getInputTensors()

        # Create the theano functions
        self.__trainFunction = theano.function(inputs=funInputs, outputs=_outputFunc,
                                               updates=optimizer.getUpdates(self.__loss, trainableLayers),
                                               mode=self.mode)
        self.evaluateFunction = theano.function(inputs=inputsOutputs, outputs=_outputFunc, mode=self.mode)
        self.__predictionFunction = theano.function(inputs=self.__x, outputs=self.__prediction, mode=self.mode)

    def doEpoch(self, trainBatchGenerators, epoch, iteration, callbacks):
        lr = self.__optimizer.getInputValues(epoch)

        self.log.info("Lr: %f" % lr[0])

        # For per-iteration evaluation.
        devBatchIterator = self.__devBatchIterator

        for x, y in trainBatchGenerators:
            iteration += 1

            # TODO: debug
            batchSize = 1
            # batchSize = len(x[0])

            inputs = []
            inputs += x

            if self.evaluateFuncUseY():
                # Theano function receives 'y' as an input.
                inputs += y

            inputs += lr

            self.callbackBatchBegin(inputs, callbacks)

            outputs = self.__trainFunction(*inputs)

            for m, _output in itertools.izip(self.trainingMetrics, outputs):
                m.update(_output, batchSize)

            self.callbackBatchEnd(inputs, callbacks)

            # Development per-iteration evaluation
            if devBatchIterator and (iteration % self.__evalPerIteration) == 0:
                evaluationStopWatch = StopWatch()
                evaluationStopWatch.start()

                results = []
                for metricName, value in self.evaluate(devBatchIterator, verbose=False).iteritems():
                    key = "eval_" + metricName
                    results.append((key, value))

                evalDuration = evaluationStopWatch.lap()

                # Print information
                info = "iteration %d" % iteration
                info += " [eval: %ds]" % evalDuration
                for k, v in results:
                    info += ' - %s:' % k
                    info += ' %.6f' % v
                self.log.info(info)

        return iteration