#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools
import logging
import time


def resetAllMetrics(metrics):
    for metric in metrics:
        metric.reset()


class Metric(object):
    def __init__(self, modelUnitName, metricName):
        self.modelUnitName = modelUnitName
        self.metricName = metricName
        self.value = 0.0
        self.seen = 0

    def update(self, value, batchSize):
        self.value += value * batchSize
        self.seen += batchSize

    def reset(self):
        self.value = 0.0
        self.seen = 0.0

    def calculate(self):
        if self.seen == 0.0:
            return -1

        return self.value / self.seen


class StopWatch(object):
    def __init__(self):
        self.__start = None

    def start(self):
        self.__start = int(time.time())

    def lap(self):
        return int(time.time()) - self.__start


class Model(object):
    def __init__(self):
        self.log = logging.getLogger(__name__)
        self.callBatchBegin = False
        self.callBatchEnd = False

    def compile(self):
        raise NotImplementedError()

    def getTrainingMetrics(self):
        raise NotImplementedError()

    def getEvaluateMetrics(self):
        raise NotImplementedError()

    def getPredictionFunction(self):
        raise NotImplementedError()

    def doEpoch(self, epoch, trainBatchGenerators, callbacks):
        raise NotImplementedError()

    def prediction(self,inputs):
        return self.getPredictionFunction()(*inputs)

    def evaluateFuncUseY(self):
        raise NotImplementedError()

    def getEvaluateFunction(self):
        raise NotImplementedError()

    def callbackBatchBegin(self, inputs, callbacks):
        for cb in callbacks:
            cb.onBatchBegin(inputs, {})

        self.callBatchBegin = True

    def callbackBatchEnd(self, inputs, callbacks):
        for cb in callbacks:
            cb.onBatchEnd(inputs, {})

        self.callBatchEnd = True

    def train(self,
              trainBatchGenerators,
              numEpochs,
              devBatchIterator=None,
              callbacks=[]):
        """
        Runs one epoch of training (one pass over the whole training dataset).

        :param trainBatchGenerators: list of batch generators for the inputs
            (training instances)

        :param numEpochs: number of passes over the training dataset.

        :param devBatchIterator: batch generator for the development dataset.

        :param callbacks: list of callbacks.

        :param evalPerIteration: indicates whether evaluation on the development
            dataset will be performed on a per-iteration basis.
        """
        stopWatch = StopWatch()
        trainingMetrics = self.getTrainingMetrics()

        for cb in callbacks:
            cb.onTrainBegin({})

        # One iteration corresponds to one call to the training procedure.
        # Thus, it corresponds to the process of one mini-batch of examples.
        iter = 0

        for epoch in xrange(numEpochs):
            for cb in callbacks:
                cb.onEpochBegin(epoch)

            resetAllMetrics(trainingMetrics)
            stopWatch.start()
            self.callBatchBegin = False
            self.callBatchEnd = False

            self.doEpoch(trainBatchGenerators, epoch, callbacks)

            if not self.callBatchBegin:
                self.log.warning("You didn't call the callbackBatchBegin function in doEpoch function")

            if not self.callBatchEnd:
                self.log.warning("You didn't call the callbackBatchEnd function in doEpoch function")

            trainingDuration = stopWatch.lap()

            logs = {}
            results = []

            for metric in trainingMetrics:
                key = metric.modelUnitName + "_" + metric.metricName
                value = metric.calculate()
                logs[key] = value
                results.append((key, value))

            if devBatchIterator:
                evaluationStopWatch = StopWatch()
                evaluationStopWatch.start()

                for metricName, value in self.evaluate(devBatchIterator, verbose=False).iteritems():
                    key = "eval_" + metricName
                    logs[key] = value
                    results.append((key, value))

                evalDuration = evaluationStopWatch.lap()

            # Print information
            info = "epoch %d" % epoch
            info += " [train: %ds]" % trainingDuration

            if devBatchIterator:
                info += " [test: %ds]" % evalDuration

            for k, v in results:
                info += ' - %s:' % k
                info += ' %.6f' % v

            self.log.info(info)

            for cb in callbacks:
                cb.onEpochEnd(epoch, logs)

        for cb in callbacks:
            cb.onTrainEnd()

    def evaluate(self, testBatchInterator, verbose=True, name="test"):
        stopWatch = StopWatch()
        evaluateMetrics = self.getEvaluateMetrics()
        evaluateFunction = self.getEvaluateFunction()

        stopWatch.start()

        resetAllMetrics(evaluateMetrics)

        for x, y in testBatchInterator:
            batchSize = len(x[0])

            inputs = []
            inputs += x

            if self.evaluateFuncUseY():
                # Theano function receives 'y' as an input
                inputs += y

            outputs = evaluateFunction(*inputs)

            for m, _output in itertools.izip(evaluateMetrics, outputs):
                m.update(_output, batchSize)

        duration = stopWatch.lap()

        logs = {}

        for metric in evaluateMetrics:
            logs[metric.metricName] = metric.calculate()

        if verbose:
            info = ""
            # Print information
            info += " [%s: %ds]" % (name, duration)

            for k, v in logs.iteritems():
                info += ' - %s:' % k
                info += ' %.6f' % v

            self.log.info(info)

        return logs
