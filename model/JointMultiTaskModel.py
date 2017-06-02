#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from itertools import izip

import theano

from model.Model import Model


class JointMultiTaskModel(Model):
    """
    We create a neural network with N output layers. These layers are joined by a shared layer and
    each of them is linked with a specific task. Thus, NN has exclusive layers for each task and
    layers that are shared along all tasks.

    To training this NN, we join all task data in one dataset in a way that we are able identify the example origin.
    We select an example of this dataset and we only update layers of a task and the shared layers
    using the task loss function; therefore, our learning is online.

    We assume that all task receive the same type of input.
    """

    def __init__(self, x, y, taskLayers, optimizer, prediction, taskLosses, taskTrainingMetrics=None, evalMetrics=None,
                 testMetrics=None, mode=None):
        """
        :param x: list of tensors that represents the inputs.

        :param y: list of tensor that represents the corrects outputs.

        :param taskLayers: Each dimension of this list has the shared layers and  the exclusive layers of a task.

        :param optimizer: Optimizer.Optimizer

        :param prediction: It's the function which will responsible to predict labels

        :param taskLosses: A list with loss functions of each task.

        :param taskTrainingMetrics:  Each dimension of this list has the metrics of a task to be applied on training dataset.

        :param evalMetrics: list of Metric objects to be applied on the evaluation dataset.

        :param testMetrics: list of Metric objects to be applied on test datasets.

        :param mode: compilation mode for Theano.
        """
        # Flat taskTrainingMetrics list
        trainingMetrics = [metric for trainingMetrics in taskTrainingMetrics for metric in trainingMetrics]

        # Theano function inputs
        inputs = x + y

        # Constructor
        super(JointMultiTaskModel, self).__init__(inputs, inputs, x, prediction, False,
                                                  trainingMetrics, evalMetrics, testMetrics,
                                                  mode)
        # Inicialize attributes
        self.__optimizer = optimizer
        self.__taskTrainingMetrics = taskTrainingMetrics
        self.__trainFunctions = []
        self.log = logging.getLogger(__name__)

        if trainingMetrics:
            # List of inputs for training function. All task receive the same input.
            trainingInputs = inputs + optimizer.getInputTensors()

            # We create a theano function for each task. This function is used to calculate the metrics and update the parameters
            for layers, trainingMetrics, loss in izip(taskLayers, taskTrainingMetrics, taskLosses):
                # List of trainable layers.
                trainableLayers = []
                for l in layers:
                    if l.isTrainable():
                        trainableLayers.append(l)

                # For each task, we update shared and exclusive layers by meand of the task loss function and the optimizer object.
                updates = optimizer.getUpdates(loss, trainableLayers)

                # Include the variables required by all training metrics in the output list of the training function.
                trainingOutputs = []
                for m in trainingMetrics:
                    trainingOutputs += m.getRequiredVariables()

                # Training function.
                self.__trainFunctions.append(
                    theano.function(inputs=trainingInputs, outputs=trainingOutputs, updates=updates, mode=self.mode))

    def doEpoch(self, trainIterator, epoch, iteration, devIterator, evalPerIteration, callbacks):
        """
        Train this model during a one epoch.

        :param trainIterator: a BatchIteratorUnion object (training instances)
        :type trainIterator: data.BatchIteratorUnion.BatchIteratorUnion

        :param epochs: current epoch.

        :param iteration: number of iterations done so far

        :param devIterator: batch generator for the development dataset.

        :param callbacks: list of callbacks.

        :param evalPerIteration: indicates whether evaluation on the development
            dataset will be performed on a per-iteration basis.

        :return number of iterations done so far
        """
        lr = self.__optimizer.getInputValues(epoch)

        self.log.info({
            "epoch": epoch,
            "iteration": iteration,
            "learn_rate": lr
        })

        for taskIdx, examples in trainIterator:
            # Features and label
            x, y = examples
            iteration += 1

            # Batch Size
            batchSize = len(x[0])

            # List of input variables of the loss function.
            inputs = []
            inputs += x
            inputs += y
            inputs += lr

            # Callbacks.
            self.callbackBatchBegin(inputs, callbacks)

            # Call training function of a task.
            outputs = self.__trainFunctions[taskIdx](*inputs)

            # Update training metrics.
            for m in self.__taskTrainingMetrics[taskIdx]:
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
            if devIterator and evalPerIteration and (iteration % evalPerIteration) == 0:
                # Perform evaluation.
                self._evaluate(devIterator, self.getEvaluationFunction(), self.__evalMetrics, epoch, iteration)

        return iteration
