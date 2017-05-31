#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools
import logging

import numpy as np

import theano
import theano.tensor as T
from model.Model import Model


class JointMultiTaskModel(Model):
    """

    """
    #TODO: Comentar código

    def __init__(self, x, y, allLayers, optimizer, prediction, loss,
                 trainingMetrics=None, evalMetrics=None, testMetrics=None, mode=None):
        """
        :param x: list of tensors that represents the inputs.

        :param y: list of tensor that represents the corrects outputs.

        :param allLayers: all model layers

        :param optimizer: Optimizer.Optimizer

        :param prediction: It's the function which will responsible to predict labels

        :param loss: It's function which calculates global loss

        :param trainingMetrics:  list of Metric objects to be applied on the training dataset.

        :param evalMetrics: list of Metric objects to be applied on the evaluation dataset.

        :param testMetrics: list of Metric objects to be applied on test datasets.

        :param mode: compilation mode for Theano.
        """
        super(JointMultiTaskModel, self).__init__([x,y], [x,y], x, prediction, False,
                                                  trainingMetrics, evalMetrics, testMetrics,
                                                  mode)
        #TODO: Terminar