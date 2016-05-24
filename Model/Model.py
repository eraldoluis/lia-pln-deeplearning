#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

import itertools
import theano.tensor as T
import theano
from keras import callbacks

from util.util import getTheanoTypeByDimension


class Callback(object):
    '''Abstract base class used to build new callbacks.

    # Properties
        params: dict. Training parameters
            (eg. verbosity, batch size, number of epochs...).
        model: instance of `keras.models.Model`.
            Reference of the model being trained.

    The `logs` dictionary that callback methods
    take as argument will contain keys for quantities relevant to
    the current batch or epoch.

    Currently, the `.fit()` method of the `Sequential` model class
    will include the following quantities in the `logs` that
    it passes to its callbacks:

        on_epoch_end: logs include `acc` and `loss`, and
            optionally include `val_loss`
            (if validation is enabled in `fit`), and `val_acc`
            (if validation and accuracy monitoring are enabled).
        on_batch_begin: logs include `size`,
            the number of samples in the current batch.
        on_batch_end: logs include `loss`, and optionally `acc`
            (if accuracy monitoring is enabled).
    '''

    def __init__(self):
        pass

    def onEpochBegin(self, epoch, logs={}):
        pass

    def onEpochEnd(self, epoch, logs={}):
        pass

    def onBatchBegin(self, batch, logs={}):
        pass

    def onBatchEnd(self, batch, logs={}):
        pass

    def onTrainBegin(self, logs={}):
        pass

    def onTrainEnd(self, logs={}):
        pass


class BaseLogger(Callback):
    '''Callback that accumulates epoch averages of
    the metrics being monitored.

    This callback is automatically applied to
    every Keras model.
    '''

    def __init__(self, metrics, verbose=True):
        self.__metrics = metrics
        self.__seen = 0
        self.__totals = {}
        self.__verbose = verbose
        self.log = logging.getLogger(__name__)

    def onEpochBegin(self, epoch, logs={}):
        self.__seen = 0
        self.__totals = {}

    def onBatchEnd(self, batch, logs={}):
        batch_size = logs.get('batchSize', 0)
        self.__seen += batch_size

        for k, v in logs.items():
            if k in self.__totals:
                self.__totals[k] += v * batch_size
            else:
                self.__totals[k] = v * batch_size

    def onEpochEnd(self, epoch, logs={}):
        for k in self.__metrics:
            if k in self.__totals:
                # make value available to next callbacks
                logs[k] = self.__totals[k] / self.__seen

        if self.__verbose:
            info = ""
            for k, v in logs.itervalues():
                info += ' - %s:' % k
                info += ' %.6f' % v

            self.log.info(info)


class Model:
    def __init__(self, outputLayer, dimOutput=0, name=None, dtype=None):
        """
        :type mainContainer: NNet.Containers.Container
        :param mainContainer: container which contains the whole neural network
        """

        self.__outputLayer = outputLayer
        self.__output = outputLayer.getOutput()
        self.__theanoFunction = None
        self.log = logging.getLogger(__name__)
        self.__calculateAcc = False
        self.__metrics = ["loss"]
        self.__y = getTheanoTypeByDimension(dimOutput + 1, name, dtype)

        self.__loss = None
        self.__inputs = None
        self.__prediction = None
        self.__layers = None
        self.__trainFunction = None
        self.__evaluateFunction = None
        self.__predictionFunction = None
        self.__optimizer = None

    def compile(self, optimizer, loss, prediction=None, metrics=[]):
        self.__optimizer = optimizer

        if prediction:
            self.__prediction = prediction.predict(self.__output)
        else:
            self.__prediction = self.__output

        # Finding the neural network inputs
        self.__inputs = []
        queue = [self.__outputLayer]
        self.__layers = [self.__outputLayer]

        while len(queue):
            layer = queue.pop()
            previousLayer = layer.getPreviousLayer()

            if len(previousLayer) == 0:
                for _input in layer.getInputs():
                    self.__inputs.append(_input)
            else:
                for previous in previousLayer:
                    queue.append(previous)
                    self.__layers.append(previous)

        self.__metrics += metrics

        _outputFunc = []
        for m in self.__metrics:
            if m == "acc":
                self.__calculateAcc = True
                _outputFunc.append(T.mean(T.eq(self.__prediction, self.__y)))
            elif m == "loss":
                self.__loss = loss.calculateError(self.__output, prediction, self.__y)
                _outputFunc.append(self.__loss)

        inputsOutputs = self.__inputs + [self.__y]
        funInputs = inputsOutputs + optimizer.getInputTensors()
        self.__trainFunction = theano.function(inputs=funInputs, outputs=_outputFunc,
                                               updates=optimizer.getUpdates(self.__loss, self.__layers))

        self.__evaluateFunction = theano.function(inputs=inputsOutputs, outputs=_outputFunc)
        self.__predictionFunction = theano.function(inputs=self.__inputs, outputs=self.__prediction)

    def prediction(self):
        pass

    def evaluate(self):
        pass

    def train(self, trainInputGenerator, numEpochs, devInputGenerator=None, callbacks=[]):
        for cb in callbacks:
            cb.onTrainBegin({})

        callbacks.append(BaseLogger(self.__metrics))

        for epoch in range(numEpochs):
            for x, y in trainInputGenerator:
                for cb in callbacks:
                    cb.onBatchBegin([x, y], {"batchSize": len(x)})

                logs = {"batchSize": len(y)}

                inputs = x + [y] + self.__optimizer.getInputValues(epoch)

                outputs = self.__trainFunction(*inputs)

                for m, _output in itertools.izip(self.__metrics, outputs):
                    logs[m] = _output

                for cb in callbacks:
                    cb.onBatchEnd([x, y], logs)

            logs = {}

            if devInputGenerator:
                for metricName, value in self.evaluate(devInputGenerator).itervalues():
                    logs["val_" + metricName] = value

            for cb in callbacks:
                cb.onEpochEnd(epoch, logs)

        for cb in callbacks:
            cb.onTrainEnd(epoch, _output)

    def evaluate(self, testInputGenerator):
        base = BaseLogger(self.__metrics)

        for x, y in testInputGenerator:
            logs = {"batchSize": len(x)}

            inputs = [x, y]
            outputs = self.__predictionFunction(inputs)

            for m, _output in itertools.izip(outputs, self.metrics):
                logs[m] = _output

            base.onBatchEnd([x, y], logs)

        base.onEpochEnd(0, logs)

        return logs
