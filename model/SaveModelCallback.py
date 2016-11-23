#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

from model.Callback import Callback


class SaveModelCallback(Callback):
    """
    This class gets the name of the metric to be monitored and if the model will be saved when this metric value has
    the biggest or lowest value until that time.
    """

    def __init__(self, modelWritter, metric, attrDict, biggest=True):
        """
        :type modelWritter: Model.ModelWriter.ModelWriter
        :param modelWritter: this class is responsible to save the model

        :type metric: model.Metric.Metric
        :param metric: a metric object

        :param attrDict: the method getValues of Metric returns a json object in a dictionary format. To get right
            value of this json, you need to pass throw this parameter a string with the follow format "attr1.attr2.attr3',
            which each attr1, attr2 and attr3 are attributes of a object. For instance, imagine you want to get value of
            the attribute 'v2' of following json {'v1':{ 'v2': 1 , "v3": 2 }. To do that, you need to set this parameter
            as 'v1.v2'.

        :param biggest: if this value is true, so the model will be saved when metric value of one epoch is bigger than the
            others epochs.
            if this value is true, so the model will be saved when metric value of one epoch is lower than the
            others epochs.

        """

        self.__modelWritter = modelWritter
        self.__metric = metric
        self.__pathMetricValue =  attrDict.split(".")
        self.__biggest = biggest

        if self.__biggest:
            self.__bestValue = -sys.maxint
        else:
            self.__bestValue = sys.maxint

    def onEpochEnd(self, epoch, logs={}):
        jsonObj = self.__metric.getValues()

        metricValue = jsonObj
        for attrName in self.__pathMetricValue:
            metricValue = metricValue[attrName]

        if self.__biggest:
            if metricValue > self.__bestValue:
                self.__modelWritter.save()
                self.__bestValue = metricValue
        else:
            if metricValue < self.__bestValue:
                self.__modelWritter.save()
                self.__bestValue = metricValue
