#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from theano import printing

from ModelOperation.Objective import Objective
import theano.tensor as T
from theano.ifelse import ifelse


class CoLearningWnnLoss(Objective):
    def __init__(self, lambda_):
        self.__lambda = lambda_

    def calculateError(self, output, ypred, ytrue):
        ytrueAux = printing.Print("y")(ytrue.clip(0, sys.maxint))
        yClipped = printing.Print("y_cli")(ytrue.clip(-1, 0))
        z = printing.Print("z")(yClipped + 1)

        nmUnsurpervisedEx = printing.Print("unsupervised")(-T.sum(yClipped))
        nmSurpervisedEx = (ytrueAux.shape[0] - nmUnsurpervisedEx).clip(1, sys.maxint)

        supervisedLoss = -T.sum(z * T.log(output[0])[T.arange(ytrueAux.shape[0]), ytrue]) / nmSurpervisedEx + \
                         - T.sum(z * T.log(output[1])[T.arange(ytrueAux.shape[0]), ytrue]) / nmSurpervisedEx

        z_ = yClipped * -1
        e = T.eq(ypred[0], ypred[1])
        nmUnsurpervisedEx = nmUnsurpervisedEx.clip(1, sys.maxint)

        unsupervisedLoss = - T.sum(
            e * z_ * T.log(output[0])[T.arange(ypred[1].shape[0]), ypred[1]]) / nmUnsurpervisedEx + \
                           - T.sum(e * z_ * T.log(output[1])[T.arange(ypred[0].shape[0]), ypred[0]]) / nmUnsurpervisedEx

        return supervisedLoss + self.__lambda * unsupervisedLoss
