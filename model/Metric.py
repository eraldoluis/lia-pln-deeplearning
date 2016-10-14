#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from itertools import izip
import theano.tensor as T


class Metric(object):
    """
    A metric specifies some tensors (Theano's variables) to be watched. The update method is called for every batch
    seen during training or for every validation batch (usually the whole dataset, in this case). The metric must keep
    track of how many examples have been seen in order to compute necessary averages or any more complex value.
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

    def __init__(self, name, loss):
        # Super-class constructor.
        super(LossMetric, self).__init__(name)

        # Required variable.
        self.__loss = loss

        # Number of examples seen so far.
        self.numExamples = 0

        # Accumulated loss over all examples seen so far.
        self.accumLoss = 0.0

    def getRequiredVariables(self):
        return [self.__loss]

    def update(self, numExamples, *values):
        self.numExamples += numExamples
        self.accumLoss += values[0]

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
        self.accumAccuracy += values[0]

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
            "Given number of examples (%d) is different from the number of given outputs (%d)" % (numExamples, count))

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
