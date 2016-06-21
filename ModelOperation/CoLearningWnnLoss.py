#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ModelOperation.Objective import Objective
import theano.tensor as T
from theano.ifelse import ifelse


class CoLearningWnnLoss(Objective):

    def calculateError(self, output, ypred, ytrue):
        supervisedLoss = -T.mean(T.log(output[0])[T.arange(ytrue.shape[0]), ytrue]) + -T.mean(
            T.log(output[1])[T.arange(ytrue.shape[0]), ytrue])

        unsupervisedLoss = -T.mean(T.log(output[0])[T.arange(ypred[1].shape[0]), ypred[1]]) + -T.mean(
            T.log(output[1])[T.arange(ypred[0].shape[0]), ypred[0]])

        return ifelse(T.lt(ytrue, 0), supervisedLoss, unsupervisedLoss)
