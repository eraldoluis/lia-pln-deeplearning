#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools
import logging
import time

import theano
import theano.tensor as T


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
        return self.value / self.seen


class StopWatch(object):
    def __init__(self):
        self.__start = None

    def start(self):
        self.__start = int(time.time())

    def lap(self):
        return int(time.time()) - self.__start


class ModelUnit:
    def __init__(self, name, x, y, loss, prediction=None, yWillBeReceived=True):
        '''
            :param x: list of tensors that represent the inputs.

            :param y: tensor that represents the correct output.

            :param yExist: This parameter is true when the learner produce your own correct output
                                            or use the input as the correct output, like DA.

        '''
        self.name = name
        self.x = x
        self.y = y
        self.loss = loss
        self.prediction = prediction
        self.yWillBeReceived = yWillBeReceived


class Model:
    def __init__(self, mode = None):
        self.log = logging.getLogger(__name__)

        self.__optimizers = None

        self.__trainFunction = None
        self.__evaluateFunction = None
        self.__predictionFunction = None

        self.__modelUnitsToTraining = []
        self.__modelUnitEvaluate = None

        self.__trainingMetrics = []
        self.__evaluateMetrics = []
        
        self.__mode = mode

    def addTrainingModelUnit(self, modelUnit, metrics=[]):
        self.__modelUnitsToTraining.append((modelUnit, metrics))

    def setEvaluatedModelUnit(self, modelUnit, metrics=[]):
        self.__modelUnitEvaluate = (modelUnit, metrics)

    def compile(self, optimizersAndLayers):
        losses = []
        _trainingOutputFunc = []  # Functions that will be calculated by theano
        funInputsTrain = []  # Inputs of the training function
        allLayers = []

        for modelUnit, metrics in self.__modelUnitsToTraining:
            # Adding loss to an array
            losses.append(modelUnit.loss)

            # Adding all layers to a same set
            funInputsTrain += modelUnit.x

            if modelUnit.yWillBeReceived:
                funInputsTrain.append(modelUnit.y)

            for metricName in metrics:
                self.__trainingMetrics.append(Metric(modelUnit.name, metricName))

                if metricName == "acc":
                    _trainingOutputFunc.append(T.mean(T.eq(modelUnit.prediction, modelUnit.y)))
                elif metricName == "loss":
                    _trainingOutputFunc.append(modelUnit.loss)

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
                self.__evaluateMetrics.append(Metric(modelUnit.name, metricName))

                if metricName == "acc":
                    _testOutputFunc.append(T.mean(T.eq(modelUnit.prediction, modelUnit.y)))
                elif metricName == "loss":
                    _testOutputFunc.append(modelUnit.loss)

        if len(losses) > 1:
            # Creates a main loss that is equal to all Sum losses
            mainLoss = T.sum(T.stack(losses), 0)
            # self.__outputFunctionName.append(0, "main")
            _trainingOutputFunc.append(mainLoss)
        else:
            mainLoss = losses[0]

        # Removes not trainable layers from update
        updates = []
        self.__optimizers = []

        for optimizer, layers in optimizersAndLayers:
            trainableLayers = []

            for l in layers:
                if l.isTrainable():
                    trainableLayers.append(l)

            updates += optimizer.getUpdates(mainLoss, trainableLayers)
            funInputsTrain += optimizer.getInputTensors()

            self.__optimizers.append(optimizer)

        # Create the theano functions
        self.__trainFunction = theano.function(inputs=funInputsTrain, 
                                               outputs=_trainingOutputFunc,
                                               updates=updates,
                                               mode=self.__mode)

        if self.__modelUnitEvaluate is not None:
            self.__evaluateFunction = theano.function(inputs=funInputsEvaluate, 
                                                      outputs=_testOutputFunc, 
                                                      mode=self.__mode)
            self.__predictionFunction = theano.function(inputs=funInputsPrediction, 
                                                        outputs=_prediction, 
                                                        mode=self.__mode)

    def prediction(self):
        pass

    def train(self, 
              trainBatchGenerators, 
              numEpochs, 
              devBatchGenerator=None, 
              callbacks=[], 
              evalPerIteration=0):
        '''
        Runs one epoch of training (one pass over the whole training dataset).
        
        :param trainBatchGenerators: list of batch generators for the inputs
            (training instances)
        
        :param numEpochs: number of passes over the training dataset. 
        
        :param devBatchGenerator: batch generator for the development dataset. 
        
        :param callbacks: list of callbacks.
        
        :param evalPerIteration: indicates whether evaluation on the development
            dataset will be performed on a per-iteration basis.
        '''
        stopWatch = StopWatch()

        for cb in callbacks:
            cb.onTrainBegin({})

        # One iteration corresponds to one call to the training procedure.
        # Thus, it corresponds to the process of one mini-batch of examples.
        iter = 0

        for epoch in range(numEpochs):
            for cb in callbacks:
                cb.onEpochBegin(epoch)

            lr = []

            for optimizer in self.__optimizers:
                lr += optimizer.getInputValues(epoch)

            resetAllMetrics(self.__trainingMetrics)

            stopWatch.start()

            for _inputs in itertools.izip_longest(*trainBatchGenerators):
                inputs = []
                batchSizes = {}
                
                iter += 1

                for modelUnitPlusMetrics, in_ in itertools.izip(self.__modelUnitsToTraining, _inputs):
                    modelUnit = modelUnitPlusMetrics[0]

                    if in_ is None:
                        raise Exception(
                            "The epoch from a model unit finished, but the others units have more examples to train.")

                    x, y = in_
                    # TODO: Irving, a mudança abaixo (inclusão dos colchetes) tem
                    # a ver com a mudança que fiz na linha 100 do arquivo 
                    # BatchIterator.py.
                    inputs += [x]

                    batchSize = len(x[0])
                    batchSizes[modelUnit.name] = batchSize

                    if modelUnit.yWillBeReceived:
                        # Theano function receives 'y' as an input
                        inputs += [y]

                for cb in callbacks:
                    cb.onBatchBegin(inputs, {})

                inputs += lr

                outputs = self.__trainFunction(*inputs)

                for m, _output in itertools.izip(self.__trainingMetrics, outputs):
                    m.update(_output, batchSizes[m.modelUnitName])

                for cb in callbacks:
                    cb.onBatchEnd(inputs, {})

                if devBatchGenerator and evalPerIteration > 0 and (iter % evalPerIteration) == 0:
                    evaluationStopWatch = StopWatch()
                    evaluationStopWatch.start()

                    results = []
                    for metricName, value in self.evaluate(devBatchGenerator, verbose=False).iteritems():
                        key = "eval_" + metricName
                        results.append((key, value))

                    evalDuration = evaluationStopWatch.lap()

                    # Print information
                    info = "iteration %d" % iter
                    info += " [eval: %ds]" % evalDuration
                    for k, v in results:
                        info += ' - %s:' % k
                        info += ' %.6f' % v
                    self.log.info(info)

            trainingDuration = stopWatch.lap()

            logs = {}
            results = []

            self.log.info("Lr: %s" % str(lr))

            for metric in self.__trainingMetrics:
                key = metric.modelUnitName + "_" + metric.metricName
                value = metric.calculate()
                logs[key] = value
                results.append((key, value))

            # Print information
            info = "epoch %d" % epoch
            info += " [train: %ds]" % trainingDuration

            if devBatchGenerator and evalPerIteration == 0:
                evaluationStopWatch = StopWatch()
                evaluationStopWatch.start()

                for metricName, value in self.evaluate(devBatchGenerator, verbose=False).iteritems():
                    key = "eval_" + metricName
                    logs[key] = value
                    results.append((key, value))

                info += " [eval: %ds]" % evaluationStopWatch.lap()

            for k, v in results:
                info += ' - %s:' % k
                info += ' %.6f' % v

            self.log.info(info)

            for cb in callbacks:
                cb.onEpochEnd(epoch, logs)

        for cb in callbacks:
            cb.onTrainEnd()

    def evaluate(self, testBatchInterator, verbose=True):
        stopWatch = StopWatch()
        stopWatch.start()

        resetAllMetrics(self.__evaluateMetrics)

        for x, y in testBatchInterator:
            batchSize = len(x[0])

            inputs = []
            # TODO: Irving, a mudança abaixo (inclusão dos colchetes) tem
            # a ver com a mudança que fiz na linha 100 do arquivo 
            # BatchIterator.py.
            inputs += [x]

            if self.__modelUnitEvaluate[0].yWillBeReceived:
                # Theano function receives 'y' as an input
                inputs += [y]

            outputs = self.__evaluateFunction(*inputs)

            for m, _output in itertools.izip(self.__evaluateMetrics, outputs):
                m.update(_output, batchSize)

        duration = stopWatch.lap()

        logs = {}

        for metric in self.__evaluateMetrics:
            logs[metric.metricName] = metric.calculate()

        if verbose:
            info = ""
            # Print information
            info += " [test: %ds]" % duration

            for k, v in logs.iteritems():
                info += ' - %s:' % k
                info += ' %.6f' % v

            self.log.info(info)

        return logs
