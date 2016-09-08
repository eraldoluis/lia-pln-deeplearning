#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

import itertools
import theano
import theano.tensor as T

from ModelOperation.Model import Model, Metric


class ReverseGradientModel(Model):

    def __init__(self, x, y ):
        super(ReverseGradientModel, self).__init__()

        self.__theanoFunction = None
        self.log = logging.getLogger(__name__)
        self.__calculateAcc = False
        self.__metrics = []
        self.__isY_ProducedByNN = False
        self.trainingMetrics = []
        self.evaluateMetrics = []

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
        self.__unsupervisedGenerator = None
        self.__lossEval = None

    def compile(self,allLayers, optimizer, predictionSup, predictionUns, lossFunction, loss, lossSup, lossUns):
        self.__optimizer = optimizer
        self.__loss =lossFunction

        _outputFuncTrain = []

        _outputFuncTrain.append(loss)
        self.trainingMetrics.append(Metric("", "loss"))

        _outputFuncTrain.append(lossSup)
        self.trainingMetrics.append(Metric("", "loss_sup"))

        _outputFuncTrain.append(lossUns)
        self.trainingMetrics.append(Metric("", "loss_uns"))

        _outputFuncTrain.append(T.mean(T.eq(predictionSup, self.__y[0])))
        self.trainingMetrics.append(Metric("", "acc_sup"))

        _outputFuncTrain.append(T.mean(T.eq(predictionUns, T.concatenate(self.__y[1:]))))
        self.trainingMetrics.append(Metric("", "acc_unsup"))


        _outputFuncEval = []

        _outputFuncEval.append(T.mean(T.eq(predictionSup, self.__y[0])))
        self.evaluateMetrics.append(Metric("", "acc"))


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
        self.__trainFunction = theano.function(inputs=funInputs, outputs=_outputFuncTrain,
                                               updates=optimizer.getUpdates(self.__loss, trainableLayers))
        self.evaluateFunction = theano.function(inputs=[self.__x[0],self.__y[0]], outputs=_outputFuncEval)
        self.__predictionFunction = theano.function(inputs=self.__x[:1], outputs=predictionSup)


    def getTrainingMetrics(self):
        return self.trainingMetrics

    def getEvaluateMetrics(self):
        return self.evaluateMetrics

    def evaluateFuncUseY(self):
        return not self.__isY_ProducedByNN

    def getEvaluateFunction(self):
        return self.evaluateFunction

    def doEpoch(self, trainBatchGenerators, epoch, callbacks):
        lr = self.__optimizer.getInputValues(epoch)

        self.log.info("Lr: %f" % lr[0])

        while True:
            try:
                xSource, ySource  = trainBatchGenerators[0].next()
            except StopIteration:
                break

            # Treats the target dataset as infinity dataset.
            # Thus the end epoch happens when we arrived at the end of file.
            try:
                xTarget, yTarget = trainBatchGenerators[1].next()
            except StopIteration:
                xTarget, yTarget = trainBatchGenerators[1].next()


            batchSize = len(xSource[0])

            inputs = []
            inputs += xSource
            inputs += xTarget

            useY = self.evaluateFuncUseY()

            if useY:
                # Theano function receives 'y' as an input
                inputs += ySource
                inputs += yTarget

            inputs += lr

            self.callbackBatchBegin(inputs, callbacks)

            outputs = self.__trainFunction(*inputs)

            for m, _output in itertools.izip(self.trainingMetrics, outputs):
                m.update(_output, batchSize)

            self.callbackBatchEnd(inputs, callbacks)