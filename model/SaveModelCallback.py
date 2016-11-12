#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

from model.Callback import Callback


class SaveModelCallback(Callback):
    """
    This class gets the name of the metric to be monitored and if the model will be saved when this metric value has
    the biggest or lowest value until that time.
    """

    def __init__(self, modelWritter, metricName, biggest=True):
        """
        :type modelWritter: Model.ModelWriter.ModelWriter
        :param modelWritter: this class is responsible to save the model

        :param metricName: metric name which the class will monitor

        :param biggest: if this value is true, so the model will be saved when metric value of one epoch is bigger than the
            others epochs.
            if this value is true, so the model will be saved when metric value of one epoch is lower than the
            others epochs.

        """

        self.__modelWritter = modelWritter
        self.__metricName = metricName
        self.__biggest = biggest

        if self.__biggest:
            self.__bestValue = -sys.maxint
        else:
            self.__bestValue = sys.maxint

    def onEpochEnd(self, epoch, logs={}):
        metricValue = logs[self.__metricName]


        if self.__biggest:
            if metricValue > self.__bestValue:
                self.__modelWritter.save()
                self.__bestValue = metricValue
        else:
            if metricValue < self.__bestValue:
                self.__modelWritter.save()
                self.__bestValue = metricValue
