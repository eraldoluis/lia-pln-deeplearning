#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools
import logging
from numpy import random

import theano
import theano.tensor as T
from model.Model import Model


class ConcatenatorDataset(object):
    """
    Treats a set of examples from differents dataset as unique dataset
    """

    def __init__(self, listSyncBatchList):
        """
        :type inputGenerators: list[DataOperation.InputGenerator.BatchIterator.SyncBatchIterator]
        :param listSyncBatchList: list that contains SyncBatchIterator from every dataset
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
    """
    Modelo baseado no trabalho Domain-Adversarial Training of Neural Networks.
    Diferente do paper que treina o modelo usando mini-batch, nós treinamos a rede de forma estocástica.
    Para isso, foi necessário criar duas entradas para a rede, sendo que a primeira recebe os exemplos do source e
        a segunda recebe os exemplos do target.
    Quando a rede recebe os exemplos do source, nós treinamos o classificador de domínios e o tagger ao mesmo tempo e
        quando nós recebemos os exemplos do target, nós treinamos somente o classificador de domínios.
    Devido a esta arquitetura e múltiplas saídas, foi necessário criar 3 saídas da rede: uma relacionada a saída do tagger
        e duas relacionadas a saída do classificador de domínios quando recebem, respectivamente, os exemplos
        do source e do target.
    """

    def __init__(self, input, sourceSupLabel, unsLabel, allLayersSource,
                 allLayersTarget, optimizer, predictionSup, lossSup, lossUnsSource, lossUnsTarget,
                 supervisedTrainMetrics=None, unsupervisedTrainMetrics=None, evalMetrics=None, testMetrics=None,
                 mode=None):
        """
        :param input: list of tensors

        :param sourceSupLabel: list of tensor that represents the corrects annotated outputs from the source domain.

        :param unsLabel: list of tensor that represents th

        :param allLayersSource: all model layers from the source

        :param allLayersTarget: all model layers from the target

        :param optimizer: Optimizer.Optimizer

        :param predictionSup: It's the function which will responsible to predict labels

        :param lossSup: It's function which calculates tagger loss

        :param lossUnsSource: It's function which calculates domain classifier loss using examples from source

        :param lossUnsTarget: It's function which calculates domain classifier loss using examples from target

        :param supervisedTrainMetrics: list of Metric objects to be applied on the supervised training. Warning: metrics
            with the same will be updates as if it were a single metric

        :param unsupervisedTrainMetrics: list of Metric objects to be applied on the unsupervised training. Warning: metrics
            with the same will be updates as if it were a single metric

        :param evalMetrics: list of Metric objects to be applied on the evaluation dataset.

        :param testMetrics: list of Metric objects to be applied on test datasets.

        :param mode: compilation mode for Theano.
        """

        # Acurácia e a loss do classificador de domínios são calculadas tanto quando recebem os exemplos do source
        # e os exemplos do target. Porém, devido a forma como é realizada o treinamento estas métricaS são calculadas
        # usando variáveis diferentes para cada um dos tipos de exemplos. Atualmente, cada métrica somente esta ligada
        # a um conjunto de variáveis e, se eu utilizasse esta implementação atual, o classificador de domínios teria uma
        # loss e acurácia para os exemplos do source e outra para os exemplos do target.
        # Para evitar isso, eu considerei que as métricas com nomes iguais calculam a mesma coisa e assim somente uma delas
        # deve ser atualizada. Assim, a loss e acurácia do classificador de domínios são calculados usando uma mesma métrica
        # e permite que cada métrica seja calculada usando variáveis do theano diferentes.
        trainMetricsByName = {}

        trainMetrics = []

        if unsupervisedTrainMetrics is not None:
            trainMetrics += unsupervisedTrainMetrics

        if supervisedTrainMetrics is not None:
            trainMetrics += set(supervisedTrainMetrics)

        for m in trainMetrics:
            trainMetricsByName[m.getName()] = m

        evalInput = input + [sourceSupLabel]

        super(GradientReversalModel, self).__init__(evalInput, evalInput, input, predictionSup, False,
                                                    trainMetricsByName.values(), evalMetrics, testMetrics, mode)

        self.log = logging.getLogger(__name__)
        self.__optimizer = optimizer

        self.__trainFuncs = []
        self.__trainingMetrics = []

        optimizerInput = optimizer.getInputTensors()

        trainableSourceLayers = []

        # Set the supervised part of the trainning
        for l in allLayersSource:
            if l.isTrainable():
                trainableSourceLayers.append(l)

        if supervisedTrainMetrics:

            # List of inputs for training function.
            sourceFuncInput = input + [sourceSupLabel, unsLabel] + optimizerInput

            # Include the variables required by all training metrics in the output list of the training function.
            trainOutputs = []
            metrics = []

            for m in supervisedTrainMetrics:
                metrics.append(trainMetricsByName[m.getName()])
                trainOutputs += m.getRequiredVariables()

            self.__trainingMetrics.append(metrics)

            # Training updates are given by the optimizer object.
            updates = optimizer.getUpdates(lossSup + lossUnsSource, trainableSourceLayers)

            # Training function.
            self.__trainFuncs.append(
                theano.function(inputs=sourceFuncInput, outputs=trainOutputs, updates=updates, mode=self.mode))

        # Set the unsupervised part of the trainning
        trainableTargetLayers = []

        for l in allLayersTarget:
            if l.isTrainable():
                trainableTargetLayers.append(l)

        if unsupervisedTrainMetrics:
            # List of inputs for training function.
            targetFuncInput = input + [unsLabel] + optimizerInput

            # Include the variables required by all training metrics in the output list of the training function.
            trainOutputs = []
            metrics = []

            for m in unsupervisedTrainMetrics:
                metrics.append(trainMetricsByName[m.getName()])
                trainOutputs += m.getRequiredVariables()

            self.__trainingMetrics.append(metrics)

            # Training updates are given by the optimizer object.
            updates = optimizer.getUpdates(lossUnsTarget, trainableTargetLayers)

            # Training function.
            self.__trainFuncs.append(theano.function(inputs=targetFuncInput, outputs=trainOutputs, updates=updates))


    def doEpoch(self, trainIterator, epoch, iteration, devIterator, evalPerIteration, callbacks):
        lr = self.__optimizer.getInputValues(epoch)

        self.log.info("Lr: %f" % lr[0])

        self.log.info({
            "epoch": epoch,
            "iteration": iteration,
            "learn_rate": lr[0]
        })

        trainingExamples = ConcatenatorDataset(trainIterator)

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

            for m in  self.__trainingMetrics[idx]:
                numOuputs = len(m.getRequiredVariables())
                mOut = []
                for _ in xrange(numOuputs):
                    mOut.append(outputs.pop(0))

                    # Update metric values.
                    m.update(batchSize, *mOut)

            self.callbackBatchEnd(inputs, callbacks)

            # Development per-iteration evaluation
            if devIterator and evalPerIteration and (iteration % evalPerIteration) == 0:
                # Perform evaluation.
                self.__evaluate(devIterator, self.getEvaluationFunction(), self.__evalMetrics, epoch, iteration)
