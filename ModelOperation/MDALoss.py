#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Model import Objective
from Model.Objective import MeanSquaredError
import theano.tensor as T


class MDALoss(Objective):
    '''
    Marginalized Denoising Auto-Encoder Loss.
    '''

    def __init__(self, objective, useDropout, noiseRate, encoderW, encoderOutput, input):
        '''
        :type objective: Model.Objective.Objective
        :param objective: can be mean squared error or cross entropy

        :param useDropout: This parameter must be true, if MDA uses a Unbiased Mask-out/drop-out.
                        If MDA uses Additive Gaussian, than this parameter must be false.

        :param noiseRate: The drop-out noise rate.

        :type encoderW: NNet.LinearLayer.LinearLayer
        :param encoderW: weights of encoder

        :param input: theano variable
        '''
        self.__obj = objective
        self.__useDropout = useDropout
        self.__noiseRate = noiseRate
        self.__encoderW = encoderW
        self.__decoderOutput = encoderOutput
        self.__x = input

    def calculateError(self, output, ypred, ytrue):
        noiseRate = self.__noiseRate
        W = self.__encoderW
        WT = self.__encoderW.T
        z = self.__decoderOutput
        dz = z * (1 - z)

        L = self.__obj.calculateError(output, ypred, ytrue)

        if isinstance(self.__obj, MeanSquaredError):
            # Mean Square
            df_x_2 = T.dot(T.sum(W * W, 1) * T.sqr(dz) , T.sqr(WT))
        else:
            # Cross Entropy
            dy = ypred * (1 - ypred)
            df_x_2 = T.dot(T.dot(dy, T.sqr(self.W)) * T.sqr(dz), T.sqr(WT))

        if self.__useDropout:
            # Droup-out
            x_2 = self.x * self.x
            L2 = noiseRate / (1 - noiseRate) * T.mean(T.sum(df_x_2 * x_2, axis=1))
        else:
            # Additive Gaussian
            L2 = noiseRate * noiseRate * T.mean(T.sum(df_x_2, axis=1))

        return T.mean(L) + 0.5 * L2

