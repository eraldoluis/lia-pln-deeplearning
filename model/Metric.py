#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

from collections import OrderedDict
from itertools import izip

import math
import theano.tensor as T
import numpy as np


class Metric(object):
    """
    A metric specifies some tensors (Theano's variables) to be watched. The update method is called for every batch
    seen during training or for every validation batch. The metric must keep track of how many examples have been seen
    in order to compute necessary averages or any more complex value.
    """

    def __init__(self, name):
        self.name = name

    def getName(self):
        return self.name

    def getRequiredVariables(self):
        raise NotImplementedError()

    def update(self, numExamples, *values):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def getValues(self):
        raise NotImplementedError()


class LossMetric(Metric):
    """
    Compute mean loss over a sequence of mini-batches.
    """

    def __init__(self, name, loss, isAvgLoss=False):
        # Super-class constructor.
        super(LossMetric, self).__init__(name)

        # Required variable.
        self.__loss = loss

        # Number of examples seen so far.
        self.numExamples = 0

        # Accumulated loss over all examples seen so far.
        self.accumLoss = 0.0

        # If the loss is the average of all errors
        # Normally, methods use the average instead of the sum
        self.__isAvgLoss = isAvgLoss

    def getRequiredVariables(self):
        return [self.__loss]

    def update(self, numExamples, *values):
        self.numExamples += numExamples

        value = values[0][0] if isinstance(values[0], (set, list)) else values[0]

        if self.__isAvgLoss:
            self.accumLoss += value * numExamples
        else:
            self.accumLoss += value

    def reset(self):
        self.numExamples = 0
        self.accumLoss = 0.0

    def getValues(self):
        return {
            "loss": self.accumLoss / self.numExamples,
            "accumLoss": self.accumLoss,
            "numExamples": self.numExamples
        }


class AccuracyMetric(Metric):
    """
    Compute mean accuracy over a sequence of mini-batches.
    """

    def __init__(self, name, correct, prediction):
        # Super-class constructor.
        super(AccuracyMetric, self).__init__(name)

        # Output produced.
        self.__output = T.eq(correct, prediction)
        # Number of examples seen so far.
        self.numExamples = 0
        # Accumulated accuracy over all examples seen so far.
        self.accumAccuracy = 0.0

    def getRequiredVariables(self):
        return [self.__output]

    def update(self, numExamples, *values):
        self.numExamples += numExamples
        self.accumAccuracy += values[0].sum()

    def reset(self):
        self.numExamples = 0
        self.accumAccuracy = 0.0

    def getValues(self):
        return {
            "accuracy": self.accumAccuracy / self.numExamples,
            "accumAccuracy": self.accumAccuracy,
            "numExamples": self.numExamples
        }


class FMetric(Metric):
    """
    Compute precision and recall values for each class and summarize them using (macro and micro) F-measure.
    """

    def __init__(self, name, correct, prediction, labels=None, beta=1.0):
        """
        Create a F-measure metric object. It needs two variables: the correct and the predicted tensors. These can be
        1-D arrays (in case of (mini-) batch processing) or scalars (in case of online training).

        :param correct: tensor variable representing the correct outputs.
        :param prediction: tensor variable representing the predicted outputs.
        :param labels: list of labels. When not given, it is assumed to be the list of seen labels.
        :param beta: F-measure parameter that balances between precision (values lower than one) and recall (values
            higher than one).
        """
        # Super-class constructor.
        super(FMetric, self).__init__(name)

        # Required values to compute this metric (correct and predicted labels).
        self.__correct = correct
        self.__prediction = prediction

        # List of labels.
        self.__labels = labels

        # F-measure beta argument.
        self.beta = beta

        # Initialize counters.
        self.__reset()

    def __reset(self):
        # Number of examples seen so far.
        self.numExamples = 0
        # True positive counts (per label).
        self.tp = {}
        # False positive counts (per label).
        self.fp = {}
        # False negative counts (per label).
        self.fn = {}

        if self.__labels:
            for l in self.__labels:
                self.tp[l] = 0
                self.fp[l] = 0
                self.fn[l] = 0

    def getRequiredVariables(self):
        return [self.__correct, self.__prediction]

    def update(self, numExamples, *values):
        # Aliases.
        tp = self.tp
        fp = self.fp
        fn = self.fn

        (correct, prediction) = values

        # In some networks, the output is a scalar, not an array of scalars.
        if correct.ndim == 0:
            correct = [correct.item()]
            prediction = [prediction.item()]

        count = 0
        for y, yHat in izip(correct, prediction):
            count += 1
            if y == yHat:
                tp[y] = tp.get(y, 0) + 1
            else:
                fn[y] = fn.get(y, 0) + 1
                fp[yHat] = fp.get(yHat, 0) + 1

        if numExamples != count:
            raise (
                "Given number of examples (%d) is different from the number of given outputs (%d)" % (
                    numExamples, count))

        self.numExamples += count

    def reset(self):
        self.__reset()

    def getValues(self):
        # Aliases.
        tp = self.tp
        fp = self.fp
        fn = self.fn
        beta = self.beta

        # Per-label metrics.
        p = {}
        r = {}
        f = {}
        tpAccum = 0
        fpAccum = 0
        fnAccum = 0

        # Macro-averaged metrics.
        macroP = 0.0
        macroR = 0.0

        labels = set(tp.keys() + fp.keys() + fn.keys())
        for label in labels:
            # Obtain values from dicts.
            tpv = tp.get(label, 0)
            fpv = fp.get(label, 0)
            fnv = fn.get(label, 0)

            # Compute precision, recall and f-measure for the label.
            pv = tpv / (tpv + fpv) if (tpv + fpv) > 0 else 0
            rv = tpv / (tpv + fnv) if (tpv + fnv) > 0 else 0
            fv = (1 + beta) * pv * rv / (beta * pv + rv) if (pv + rv) > 0 else 0

            # Set precision and recall dicts.
            p[label] = pv
            r[label] = rv
            f[label] = fv

            # Update accumulated values (fro macro-averaged metrics).
            macroP += pv
            macroR += rv

            # Update accumulated counts (for micro-averaged metrics).
            tpAccum += tpv
            fpAccum += fpv
            fnAccum += fnv

        # Macro-averaged metrics.
        numLabels = len(labels)
        macroP /= numLabels
        macroR /= numLabels
        macroF = (1 + beta) * macroP * macroR / (beta * macroP + macroR)

        # Micro-averaged metrics.
        microP = tpAccum / (tpAccum + fpAccum)
        microR = tpAccum / (tpAccum + fnAccum)
        microF = (1 + beta) * microP * microR / (beta * microP + microR)

        return {
            "numExamples": self.numExamples,
            "numLabels": numLabels,
            "beta": beta,
            "truePositives": tpAccum,
            "falsePositives": tpAccum,
            "falseNegatives": fnAccum,
            "macro": {
                "precision": macroP,
                "recall": macroR,
                "f": macroF
            },
            "micro": {
                "precision": microP,
                "recall": microR,
                "f": microF
            },
            "perLabel": {
                "truePositives": tp,
                "falsePositives": fp,
                "falseNegatives": fn,
                "precision": p,
                "recall": r,
                "f": f
            }
        }


class ActivationMetric(Metric):
    """
    Get the activations of one layer.
    For epoch or number of iteration,
        we calculate the average and standard deviation and create a histogram of layer activation .
    """

    def __init__(self, name, activation, intervals, absolutValueOption):

        """
        Create a Activation metric object.
        You need to pass the theano variable
            which represents layer activation output.


        :param name: name of the metric
        :param activation: the theano variable which represents layer activation
        :param intervals: a list of real numbers. Imagine that parameter has following value: [-1, -0.5, 0, 0.5, 1].
            So in this case, we have a histogram with 5 intervals: [-1, 0.5), [0.5,0.0), [0.0,0.5), (0.5,1].
        :param absolutValueOption: There are the following options to this parameter:
            "none": It doesn't use absolute value to calculate de average and to create the histogram;
            "avg": It only use absolute value to calculate the average;
            "all": It use absolute value to calculate the average to create the histogram;

        """
        # Super-class constructor.
        super(ActivationMetric, self).__init__(name)

        self.__activation = activation
        self.__intervals = intervals
        self.__isToAbsoluteValueAvg = absolutValueOption in ["all", "avg"]
        self.__isToAbsoluteValueHist = absolutValueOption in ["all"]

        # This parameter is really initialized in the reset method
        self.__statistics = None
        self.reset()

    def getRequiredVariables(self):
        return [self.__activation]

    def update(self, numExamples, *values):
        for batchActivationValues in values:
            avg = self.__statistics["average"]
            variance = self.__statistics["variance"]
            histogram = self.__statistics["histogram"]
            totalExample = self.__statistics["total_example"]
            # all = self.__statistics["all_examples"]

            if isinstance(batchActivationValues[0], (int, float)):
                # It's not using batch or mini-batch
                batchActivationValues = [batchActivationValues]

            for activationValues in batchActivationValues:
                for actValue in activationValues:
                    totalExample += 1
                    oldAvg = avg

                    # all.append(actValue)

                    if self.__isToAbsoluteValueAvg:
                        actValueAvg = math.fabs(actValue)
                    else:
                        actValueAvg = actValue

                    avg += (actValueAvg - avg) / totalExample
                    variance += (actValueAvg - avg) * (actValueAvg - oldAvg)

                    if self.__isToAbsoluteValueHist:
                        actValueHist = math.fabs(actValue)
                    else:
                        actValueHist = actValue

                    if actValueHist < self.__intervals[0]:
                        raise Exception("One value(%.4f) of a activation is smaller than the minimum value %.4f" % (
                            actValueHist, self.__intervals))

                    if actValueHist > self.__intervals[-1]:
                        raise Exception("One value(%.4f) of a activation is greater than the maximum value %.4f" % (
                            actValueHist, self.__intervals[-1]))

                    minInterval = self.__intervals[0]

                    for idx, interval in enumerate(self.__intervals[1:]):
                        if interval == self.__intervals[-1]:
                            if minInterval <= actValueHist <= interval:
                                histogram[idx] += 1
                                break
                        else:
                            if minInterval <= actValueHist < interval:
                                histogram[idx] += 1
                                break

                        minInterval = interval

            self.__statistics["average"] = avg
            self.__statistics["variance"] = variance
            # DEBUG
            # allEx = np.asarray(self.__statistics["all_examples"])
            # assert np.isclose(avg, np.fabs(allEx).mean() if self.__isToAbsoluteValueAvg else allEx.mean())
            # assert np.isclose(variance / totalExample, np.fabs(allEx).var() if self.__isToAbsoluteValueAvg else allEx.var())

            self.__statistics["histogram"] = histogram
            self.__statistics["total_example"] = totalExample

    def reset(self):
        self.__statistics = {
            # We calculate de avg and standard deviation using the Welford’s method
            #   (http://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/)
            "average": 0.0,
            "variance": 0.0,
            "total_example": 0,

            # the number of classes = len(intervals) - 1
            "histogram": [0 for _ in self.__intervals[:-1]],

            # "all_examples": []
        }

    def getValues(self):
        totalExample = self.__statistics["total_example"]
        variance = self.__statistics["variance"] / totalExample

        histogram = OrderedDict()

        for idx, interval in enumerate(self.__intervals[:-1]):
            intervalStr = "[%.2f, %.2f]" % (self.__intervals[idx], self.__intervals[idx + 1])
            histogram[intervalStr] = self.__statistics["histogram"][idx]

        # DEBUG
        # all_examples = np.fabs(np.asarray(self.__statistics["all_examples"])) if self.__isToAbsoluteValueHist else np.asarray(self.__statistics["all_examples"])
        # for a, b  in izip(np.histogram(all_examples, self.__intervals)[0], self.__statistics["histogram"]):
        #     assert np.isclose(a, b)
        # print self.__statistics["histogram"]
        # print np.histogram(all_examples, self.__intervals)
        #
        # print np.fabs(np.asarray(self.__statistics["all_examples"])).mean() if self.__isToAbsoluteValueAvg else np.asarray(self.__statistics["all_examples"]).mean()
        # print np.fabs(np.asarray(self.__statistics["all_examples"])).var() if self.__isToAbsoluteValueAvg else np.asarray(self.__statistics["all_examples"]).var()


        return {
            "average": self.__statistics["average"],
            "variance": variance,
            "std_deviation": math.sqrt(variance),
            "total_example": totalExample,
            "histogram": histogram
        }


class DerivativeMetric(Metric):
    """
    This object gets a derivatives of a layer.
    When the method getValues is called,
        it returns the average, standard deviation and a histogram of derivative.
    """

    def __init__(self, name, lossFunction, variable, intervals, absolutValueOption):
        """
        Create a Derivate metric object.
        You need to pass the theano variable
            which represents layer activation output.
        :param name: name of the metric

        :type lossFunction: T.var.TensorVariable
        :param lossFunction: It's the function which represents the loss function

        :param variable: theano variable of the model

        :param activation: the theano variable which represents layer activation
        :param intervals: a list of real numbers. Imagine that parameter has following value: [-1, -0.5, 0, 0.5, 1].
            So in this case, we have a histogram with 5 intervals: [-1, 0.5), [0.5,0.0), [0.0,0.5), [0.5,1].

        :param absolutValueOption: There are the following options to this parameter:
            "none": It doesn't use absolute value to calculate de average and to create the histogram;
            "avg": It only use absolute value to calculate de average;
            "all": It use absolute value to calculate de average to create the histogram;

        """
        # Super-class constructor.
        super(DerivativeMetric, self).__init__(name)

        # Calculate the
        self.__grads = T.grad(lossFunction, variable)
        self.__intervals = intervals
        self.__isToAbsoluteValueAvg = absolutValueOption in ["all", "avg"]
        self.__isToAbsoluteValueHist = absolutValueOption in ["all"]

        # This parameter is really initialized in the reset method
        self.__statistics = None
        self.reset()

    def getRequiredVariables(self):
        return [self.__grads]

    def update(self, numExamples, *values):
        for batchDerivativeValues in values:
            avg = self.__statistics["average"]
            variance = self.__statistics["variance"]
            histogram = self.__statistics["histogram"]
            totalExample = self.__statistics["total_example"]
            # all = self.__statistics["all_examples"]

            if isinstance(batchDerivativeValues[0], (int, float)):
                # It's not using batch or mini-batch
                batchDerivativeValues = [batchDerivativeValues]

            for derivativeValues in batchDerivativeValues:
                for derivativeValue in derivativeValues:
                    totalExample += 1
                    oldAvg = avg

                    if self.__isToAbsoluteValueAvg:
                        derivativeValueAvg = math.fabs(derivativeValue)
                    else:
                        derivativeValueAvg = derivativeValue

                    avg += (derivativeValueAvg - avg) / totalExample
                    variance += (derivativeValueAvg - avg) * (derivativeValueAvg - oldAvg)
                    # all.append(derivativeValue)

                    if self.__isToAbsoluteValueHist:
                        derivativeValueHist = math.fabs(derivativeValue)
                    else:
                        derivativeValueHist = derivativeValue

                    if derivativeValueHist < self.__intervals[0]:
                        raise Exception("One value(%.4f) of a activation is smaller than the minimum value %.4f" % (
                            derivativeValueHist, self.__intervals[0]))

                    if derivativeValueHist > self.__intervals[-1]:
                        raise Exception("One value(%.4f) of a activation is greater than the maximum value %.4f" % (
                            derivativeValueHist, self.__intervals[-1]))


                    minInterval = self.__intervals[0]

                    for idx, interval in enumerate(self.__intervals[1:]):
                        if interval == self.__intervals[-1]:
                            if minInterval <= derivativeValueHist <= interval:
                                histogram[idx] += 1
                                break
                        else:
                            if minInterval <= derivativeValueHist < interval:
                                histogram[idx] += 1
                                break

                        minInterval = interval


            self.__statistics["average"] = avg
            self.__statistics["variance"] = variance
            # DEBUG
            # allEx = np.asarray(self.__statistics["all_examples"])
            # assert np.isclose(avg, np.fabs(allEx).mean() if self.__isToAbsoluteValueAvg else allEx.mean())
            # assert np.isclose(variance / totalExample,
            #                   np.fabs(allEx).var() if self.__isToAbsoluteValueAvg else allEx.var())

            self.__statistics["histogram"] = histogram
            self.__statistics["total_example"] = totalExample

    def reset(self):
        self.__statistics = {
            # We calculate de avg and standard deviation using the Welford’s method
            #   (http://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/)
            "average": 0.0,
            "variance": 0.0,
            "total_example": 0,

            # the number of classes = len(intervals) - 1
            "histogram": [0 for _ in self.__intervals[:-1]],

            # "all_examples": []
        }

    def getValues(self):
        totalExample = self.__statistics["total_example"]
        variance = self.__statistics["variance"] / totalExample

        histogram = OrderedDict()

        for idx, interval in enumerate(self.__intervals[:-1]):
            intervalStr = "[%.2f, %.2f]" % (self.__intervals[idx], self.__intervals[idx + 1])
            histogram[intervalStr] = self.__statistics["histogram"][idx]

        # DEBUG
        # all_examples = np.fabs(np.asarray(
        #     self.__statistics["all_examples"])) if self.__isToAbsoluteValueHist else np.asarray(
        #     self.__statistics["all_examples"])
        #
        # print self.__statistics["histogram"]
        # print np.histogram(all_examples, self.__intervals)
        # for a, b in izip(np.histogram(all_examples, self.__intervals)[0], self.__statistics["histogram"]):
        #     assert np.isclose(a, b)
        #
        # print np.fabs(np.asarray(
        #     self.__statistics["all_examples"])).mean() if self.__isToAbsoluteValueAvg else np.asarray(
        #     self.__statistics["all_examples"]).mean()
        # print np.fabs(np.asarray(self.__statistics["all_examples"])).var() if self.__isToAbsoluteValueAvg else np.asarray(
        #     self.__statistics["all_examples"]).var()

        return {
            "average": self.__statistics["average"],
            "variance": variance,
            "std_deviation": math.sqrt(variance),
            "total_example": totalExample,
            "histogram": histogram
        }
