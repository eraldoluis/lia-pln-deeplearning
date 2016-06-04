#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

import itertools
import theano.tensor as T
import theano

from ModelOperation.Callback import BaseLogger


class Model:
    def __init__(self, x, y, outputLayer):
        '''
        :param x: list of tensors that represent the inputs.

        :param y: tensor that represents the correct output.

        :type outputLayer: NNet.Layer.Layer
        :param outputLayer: the output
        '''

        self.__outputLayer = outputLayer
        self.__output = outputLayer.getOutput()
        self.__theanoFunction = None
        self.log = logging.getLogger(__name__)
        self.__calculateAcc = False
        self.__metrics = ["loss"]
        self.__y = y

        if not isinstance(x, (set, list)):
            self.__x = [x]
        else:
            self.__x = x

        self.__isY_ProducedByNN = False

        self.__loss = None
        self.__inputs = None
        self.__prediction = None
        self.__layers = None
        self.__trainFunction = None
        self.__evaluateFunction = None
        self.__predictionFunction = None
        self.__optimizer = None

    def compile(self, optimizer, loss, prediction=None, metrics=[]):
        '''
        :type optimizer: Optimizer.Optimizer
        :param optimizer:

        :type loss: ModelOperation.Objective.Objective
        :param loss:

        :type prediction: ModelOperation.Prediction.Prediction
        :param prediction:

        :param metrics: the names of the metrics to be measured. Nowadays, this classe just accepted "loss" and "acc"(accuracy)
        '''
        self.__optimizer = optimizer

        if prediction:
            self.__prediction = prediction.predict(self.__output)
        else:
            self.__prediction = self.__output

        self.__metrics += metrics

        _outputFunc = []
        for m in self.__metrics:
            if m == "acc":
                self.__calculateAcc = True
                _outputFunc.append(T.mean(T.eq(self.__prediction, self.__y)))
            elif m == "loss":
                self.__loss = loss.calculateError(self.__output, self.__prediction, self.__y)
                _outputFunc.append(self.__loss)

        # Removes not trainable layers from update and see if the output of the
        trainableLayers = []

        for l in self.__outputLayer.getLayerSet():
            if l.isTrainable():
                trainableLayers.append(l)
            if l.getOutput() == self.__y:
                self.__isY_ProducedByNN = True

        # Create the inputs of which theano function
        inputsOutputs = []
        inputsOutputs += self.__x

        if not self.__isY_ProducedByNN:
            inputsOutputs += [self.__y]

        funInputs = inputsOutputs + optimizer.getInputTensors()

        # Create the theano functions
        self.__trainFunction = theano.function(inputs=funInputs, outputs=_outputFunc,
                                               updates=optimizer.getUpdates(self.__loss, trainableLayers))
        self.__evaluateFunction = theano.function(inputs=inputsOutputs, outputs=_outputFunc)
        self.__predictionFunction = theano.function(inputs=self.__x, outputs=self.__prediction)

    def prediction(self):
        pass

    def train(self, trainBatchGenerator, numEpochs, devBatchGenerator=None, callbacks=[]):
        for cb in callbacks:
            cb.onTrainBegin({})

        # We insert the BaseLogger in front of the list,
        #   because in this way every Callback can get the metric values like 'loss'.
        callbacks.insert(0, BaseLogger(self.__metrics))

        for epoch in range(numEpochs):
            for cb in callbacks:
                cb.onEpochBegin(epoch)

            lr = self.__optimizer.getInputValues(epoch)

            for x, y in trainBatchGenerator:
                for cb in callbacks:
                    cb.onBatchBegin([x, y], {"batchSize": len(x[0])})

                logs = {"batchSize": len(x[0])}
                inputs = []
                inputs += x

                if not self.__isY_ProducedByNN:
                    # Theano function receives 'y' as an input
                    inputs += [y]

                inputs += lr

                outputs = self.__trainFunction(*inputs)

                for m, _output in itertools.izip(self.__metrics, outputs):
                    logs[m] = _output

                for cb in callbacks:
                    cb.onBatchEnd([x, y], logs)

            logs = {}

            self.log.info("Lr: %f" % lr[0])

            if devBatchGenerator:
                for metricName, value in self.evaluate(devBatchGenerator, verbose=False).iteritems():
                    logs["val_" + metricName] = value

            for cb in callbacks:
                cb.onEpochEnd(epoch, logs)

        for cb in callbacks:
            cb.onTrainEnd(_output)

    def evaluate(self, testBatchInterator, verbose=True):
        base = BaseLogger(self.__metrics, verbose)

        for x, y in testBatchInterator:
            logs = {"batchSize": len(x[0])}

            inputs = []
            inputs += x

            if not self.__isY_ProducedByNN:
                # Theano function receives 'y' as an input
                inputs += [y]
            outputs = self.__evaluateFunction(*inputs)

            for m, _output in itertools.izip(self.__metrics, outputs):
                logs[m] = _output

            base.onBatchEnd([x, y], logs)

        logs = {}

        base.onEpochEnd(0, logs)

        return logs
