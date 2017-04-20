#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import time

import theano


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
    def __init__(self, evalInputs, testInput, predictionInput, prediction, isYProduceByNN=False, trainMetrics=None,
                 evalMetrics=None, testMetrics=None, mode=None):
        """
        :param evalInputs: list of tensors that represent the inputs of evaluation function.

        :param testInput: list of tensors that represent the inputs of test function.

        :param predictionInput: list of tensors that represent the inputs of prediction function.

        :type prediction: T.var.TensorVariable
        :param prediction: It's the function which will responsible to predict labels

        :param isYProduceByNN: This parameter is true when the learner produce your own correct output
                                        or use the input as the correct output, like DA.
        :param trainMetrics: list of Metric objects to be applied on the training dataset.

        :param evalMetrics: list of Metric objects to be applied on the evaluation dataset.

        :param testMetrics: list of Metric objects to be applied on test datasets.

        :param mode: compilation mode for Theano.
        """
        self.log = logging.getLogger(__name__)
        self.callBatchBegin = False
        self.callBatchEnd = False
        self.__trainMetrics = trainMetrics
        self.__evalMetrics = evalMetrics
        self.__testMetrics = testMetrics
        self.__isYProducedByNN = isYProduceByNN
        self.mode = mode

        if evalMetrics:
            # Include the variables required by all evaluation metrics in the output list of the evaluation function.
            evalOutputs = []
            for m in evalMetrics:
                evalOutputs += m.getRequiredVariables()

            # Evaluation function.
            self.__evaluateFunction = theano.function(inputs=evalInputs, outputs=evalOutputs, mode=self.mode)

        if testMetrics:
            # Include the variables required by all test metrics in the output list of the test function.
            testOutputs = []
            for m in testMetrics:
                testOutputs += m.getRequiredVariables()

            # Evaluation function.
            self.__testFunction = theano.function(inputs=testInput, outputs=testOutputs, mode=self.mode)

        # Prediction function.
        if prediction:
            self.__predictionFunction = theano.function(inputs=predictionInput, outputs=prediction, mode=self.mode)


    def getEvaluationFunction(self):
        return self.__evaluateFunction

    def getTestFunction(self):
        return self.__testFunction

    def getPredictionFunction(self):
        return self.__predictionFunction

    def getTrainMetrics(self):
        return self.__trainMetrics

    def prediction(self, inputs):
        return self.__predictionFunction(*inputs)

    def callbackBatchBegin(self, inputs, callbacks):
        for cb in callbacks:
            cb.onBatchBegin(inputs, {})
        self.callBatchBegin = True

    def callbackBatchEnd(self, inputs, callbacks):
        for cb in callbacks:
            cb.onBatchEnd(inputs, {})
        self.callBatchEnd = True

    def isYProducedByNN(self):
        return self.__isYProducedByNN

    def evaluateFuncUseY(self):
        return not self.__isYProducedByNN

    def getPredictionFunction(self):
        return self.__predictionFunction

    def train(self, trainIterator, numEpochs, devIterator=None, evalPerIteration=None, callbacks=[]):
        """
        Train this model during the given number of epochs on the examples returned by the given iterator.

        :param trainIterator: list of batch generators for the inputs
            (training instances)

        :param numEpochs: number of passes over the training dataset.

        :param devIterator: batch generator for the development dataset.

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

            iteration = self.doEpoch(trainIterator, epoch, iteration, devIterator, evalPerIteration, callbacks)

            if not self.callBatchBegin:
                self.log.warning("You didn't call the callbackBatchBegin function in doEpoch function")

            if not self.callBatchEnd:
                self.log.warning("You didn't call the callbackBatchEnd function in doEpoch function")

            trainingDuration = stopWatch.lap()

            # Dump training metrics results.
            for m in self.__trainMetrics:
                log.info({
                    "type": "metric",
                    "subtype": "train",
                    "epoch": epoch,
                    "iteration": iteration,
                    "name": m.getName(),
                    "values": m.getValues()
                })

            # Evaluate model after each epoch.
            if devIterator and not evalPerIteration:
                self._evaluate(devIterator, self.__evaluateFunction, self.__evalMetrics, epoch, iteration)

            # Dump training duration.
            log.info({
                "type": "duration",
                "subtype": "training",
                "epoch": epoch,
                "iteration": iteration,
                "duration": trainingDuration
            })

            # Callbacks.
            for cb in callbacks:
                cb.onEpochEnd(epoch, {})

        for cb in callbacks:
            cb.onTrainEnd()

    def doEpoch(self, trainIterator, epoch, iteration, devIterator, evalPerIteration, callbacks):
        """
        Train this model during a one epoch.

        :param trainIterator: list of batch generators for the inputs (training instances)

        :param epochs: current epoch.

        :param iteration: number of iterations done so far

        :param devIterator: batch generator for the development dataset.

        :param callbacks: list of callbacks.

        :param evalPerIteration: indicates whether evaluation on the development
            dataset will be performed on a per-iteration basis.

        :return number of iterations done so far
        """

        raise NotImplementedError()

    def test(self, examplesIterator):
        """
        Apply this model to the given examples and compute the test metrics given in the constructor. If no test metric
        has been given in the constructor, this method will fail.

        :param examplesIterator: iterator on test examples.
        """
        self._evaluate(examplesIterator, self.__testFunction, self.__testMetrics, None, None)

    def _evaluate(self, examplesIterator, evalFunction, evalMetrics, epoch, iteration):
        """
        Apply the given evaluation function on the examples returned by the given iterator. The given metrics are
        computed. These metrics must be the same used when compiling the given evaluation function. For instance, if you
        give the test function, then the given metrics must be the test metrics (given in the constructor). This is
        necessary because the function outputs are defined based on the related metrics.

        :param examplesIterator: iterator over the examples to be evaluated.
        :param evalFunction: compiled Theano function to evaluate the metrics.
        :param evalMetrics: list of metrics to be computed.
        :param epoch: training epoch (used during training).
        :param iteration: training iteration (used during training).
        """
        # Aliases.
        log = self.log

        # Reset all evaluation metrics.
        resetAllMetrics(evalMetrics)

        # Record time elapsed during evaluation.
        stopWatch = StopWatch()
        stopWatch.start()

        for x, y in examplesIterator:
            # TODO: acho perigoso calcular acurácia da validação (ou do teste) desta forma. Acho que deveria ser feito
            # de uma maneira mais clara e simples. Por exemplo, construir dois arrays y e ŷ para todos os exemplos e
            # daí calcular a acurácia (ou qualquer outra métrica).

            # Alterei o cálculo do batchSize para ser feito pelo tamanho do y, ao invés do tamanho do x. Acho isto mais
            # geral pois funciona, por exemplo, para classificação de documentos também, onde o x de um exemplo é maior
            # do que 1 (várias palavras).

            if self.isYProducedByNN():
               batchSize = len(x[0])
            elif y[0].ndim > 0:
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
                r = m.update(batchSize, *mOut)

                if r is not None:
                    with open('./epoch{0}.txt'.format(epoch), 'a') as f:
                        for v in r:
                            f.write('{0},'.format(v))
        
                        f.write("\n")

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
