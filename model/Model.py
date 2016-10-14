#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import time
import theano
import json


def resetAllMetrics(metrics):
    """
    Reset all metrics in the given list.

    :param metrics: list of metrics to be reset.
    """
    for metric in metrics:
        metric.reset()


class StopWatch(object):
    """
    Stopwatch to assess any procedure running time.
    """

    def __init__(self):
        self.__start = None

    def start(self):
        self.__start = int(time.time())

    def lap(self):
        return int(time.time()) - self.__start


class Model(object):
    """

    """

    def __init__(self, x, y, allLayers, optimizer, prediction, loss, yExist=False, trainMetrics=[], evalMetrics=[],
                 mode=None):
        """
        :param x: list of tensors that represent the inputs.

        :param y: list of tensor that represents the corrects outputs.

        :type allLayers: [nnet.Layer]
        :param allLayers: all model layers

        :type optimizer: Optimizer.Optimizer
        :param optimizer:

        :type prediction: T.var.TensorVariable
        :param prediction: It's the function which will responsible to predict labels

        :param yExist: This parameter is true when the learner produce your own correct output
                                        or use the input as the correct output, like DA.

        :param trainMetrics: list of Metric objects to be applied on the training dataset.

        :param evalMetrics: list of Metric objects to be applied on the evaluation dataset.

        :param mode: compilation mode for Theano.
        """
        self.mode = mode
        self.log = logging.getLogger(__name__)
        self.__isY_ProducedByNN = yExist

        # Lists of metrics.
        self.__trainMetrics = trainMetrics
        self.__evalMetrics = evalMetrics

        if not isinstance(x, (set, list)):
            self.__x = [x]
        else:
            self.__x = x

        if not isinstance(y, (set, list)):
            self.__y = [y]
        else:
            self.__y = y

        self.__optimizer = optimizer
        self.__prediction = prediction

        # List of trainable layers.
        trainableLayers = []
        for l in allLayers:
            if l.isTrainable():
                trainableLayers.append(l)

        # Inputs for the evaluation function.
        evalInputs = self.__x[:]
        if not self.__isY_ProducedByNN:
            evalInputs += self.__y

        # List of inputs for training function.
        trainInputs = evalInputs + optimizer.getInputTensors()

        # Training updates are given by the optimizer object.
        updates = optimizer.getUpdates(loss, trainableLayers)

        # Include the variables required by all training metrics in the output list of the training function.
        trainOutputs = []
        for m in self.__trainMetrics:
            trainOutputs += m.getRequiredVariables()

        # Training function.
        self.__trainFunction = theano.function(inputs=trainInputs, outputs=trainOutputs, updates=updates,
                                               mode=self.mode)

        # Include the variables required by all evaluation metrics in the output list of the evaluation function.
        evalOutputs = []
        for m in self.__evalMetrics:
            evalOutputs += m.getRequiredVariables()

        # Evaluation function.
        self.__evaluateFunction = theano.function(inputs=evalInputs, outputs=evalOutputs, mode=self.mode)

        # Prediction function.
        self.__predictionFunction = theano.function(inputs=self.__x, outputs=self.__prediction, mode=self.mode)

        self.callBatchBegin = False
        self.callBatchEnd = False

    def prediction(self, inputs):
        return self.getPredictionFunction()(*inputs)

    def callbackBatchBegin(self, inputs, callbacks):
        for cb in callbacks:
            cb.onBatchBegin(inputs, {})
        self.callBatchBegin = True

    def callbackBatchEnd(self, inputs, callbacks):
        for cb in callbacks:
            cb.onBatchEnd(inputs, {})
        self.callBatchEnd = True

    def evaluateFuncUseY(self):
        return not self.__isY_ProducedByNN

    def getPredictionFunction(self):
        return self.__predictionFunction

    def train(self, trainBatchIterator, numEpochs, devBatchIterator=None, evalPerIteration=None, callbacks=[]):
        """
        Runs one epoch of training (one pass over the whole training dataset).

        :param trainBatchIterator: list of batch generators for the inputs
            (training instances)

        :param numEpochs: number of passes over the training dataset.

        :param devBatchIterator: batch generator for the development dataset.

        :param callbacks: list of callbacks.

        :param evalPerIteration: indicates whether evaluation on the development
            dataset will be performed on a per-iteration basis.
        """
        # Aliases.
        log = self.log

        stopWatch = StopWatch()

        for cb in callbacks:
            cb.onTrainBegin({})

        # One iteration corresponds to one call to the training procedure.
        # Thus, it corresponds to the process of one mini-batch of examples.
        iteration = 0

        for epoch in xrange(numEpochs):
            for cb in callbacks:
                cb.onEpochBegin(epoch)

            resetAllMetrics(self.__trainMetrics)
            stopWatch.start()
            self.callBatchBegin = False
            self.callBatchEnd = False

            iteration = self.doEpoch(trainBatchIterator, epoch, iteration, devBatchIterator, evalPerIteration,
                                     callbacks)

            if not self.callBatchBegin:
                self.log.warning("You didn't call the callbackBatchBegin function in doEpoch function")

            if not self.callBatchEnd:
                self.log.warning("You didn't call the callbackBatchEnd function in doEpoch function")

            trainingDuration = stopWatch.lap()

            # Dump training metrics results.
            for m in self.__trainMetrics:
                log.info({
                    "type": "metric",
                    "subtype": "evaluation",
                    "epoch": epoch,
                    "iteration": iteration,
                    "name": m.getName(),
                    "values": m.getValues()
                })

            # Evaluate model after each epoch.
            if devBatchIterator and not evalPerIteration:
                self.evaluate(devBatchIterator, epoch, iteration)

            # Dump training duration.
            log.info({
                "type": "duration",
                "subtype": "evaluation",
                "epoch": epoch,
                "iteration": iteration,
                "duration": trainingDuration
            })

            # Callbacks.
            for cb in callbacks:
                cb.onEpochEnd(epoch, {})

        for cb in callbacks:
            cb.onTrainEnd()

    def doEpoch(self, trainBatchGenerators, epoch, iteration, devBatchIterator, evalPerIteration, callbacks):
        lr = self.__optimizer.getInputValues(epoch)

        self.log.info({
            "epoch": epoch,
            "iteration": iteration,
            "learn_rate": lr
        })

        for x, y in trainBatchGenerators:
            iteration += 1

            if y[0].ndim > 0:
                batchSize = len(y[0])
            else:
                batchSize = 1

            # List of input variables.
            inputs = []
            inputs += x
            if self.evaluateFuncUseY():
                # Theano function receives 'y' as an input.
                inputs += y
            inputs += lr

            # Callbacks.
            self.callbackBatchBegin(inputs, callbacks)

            # Call training function.
            outputs = self.__trainFunction(*inputs)

            # Update training metrics.
            for m in self.__trainMetrics:
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
            if devBatchIterator and evalPerIteration and (iteration % evalPerIteration) == 0:
                # Perform evaluation.
                self.evaluate(devBatchIterator, epoch, iteration)

        return iteration

    def evaluate(self, testBatchInterator, epoch, iteration):
        # Aliases.
        evalMetrics = self.__evalMetrics
        evalFunction = self.__evaluateFunction
        log = self.log

        # Reset all evaluation metrics.
        resetAllMetrics(evalMetrics)

        # Record time elapsed during evaluation.
        stopWatch = StopWatch()
        stopWatch.start()

        for x, y in testBatchInterator:
            # TODO: acho perigoso calcular acurácia da validação (ou do teste) desta forma. Acho que deveria ser feito
            # de uma maneira mais clara e simples. Por exemplo, construir dois arrays y e ŷ para todos os exemplos e
            # daí calcular a acurácia (ou qualquer outra métrica).

            # Alterei o cálculo do batchSize para ser feito pelo tamanho do y, ao invés do tamanho do x. Acho isto mais
            # geral pois funciona, por exemplo, para classificação de documentos também, onde o x de um exemplo é maior
            # do que 1 (várias palavras).
            if y[0].ndim > 0:
                batchSize = len(y[0])
            else:
                batchSize = 1

            # List of input variables.
            inputs = []
            inputs += x
            if self.evaluateFuncUseY():
                # Theano function receives 'y' as an input
                inputs += y

            # Execute the evaluation function (compute the output for all metrics).
            outputs = evalFunction(*inputs)

            # Update each metric with the computed outputs.
            for m in evalMetrics:
                # Build list of required variables.
                numOuputs = len(m.getRequiredVariables())
                mOut = []
                for _ in xrange(numOuputs):
                    mOut.append(outputs.pop(0))

                # Update metric values.
                m.update(batchSize, *mOut)

        duration = stopWatch.lap()

        # Dump metrics results.
        for m in evalMetrics:
            log.info({
                "type": "metric",
                "subtype": "evaluation",
                "epoch": epoch,
                "iteration": iteration,
                "name": m.getName(),
                "values": m.getValues()
            })

        # Dump evaluation duration.
        log.info({
            "type": "duration",
            "subtype": "evaluation",
            "epoch": epoch,
            "iteration": iteration,
            "duration": duration
        })
