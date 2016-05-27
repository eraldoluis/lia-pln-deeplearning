#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging


class Callback(object):
    '''Abstract base class used to build new callbacks.

    # Properties
        params: dict. Training parameters
            (eg. verbosity, batch size, number of epochs...).
        model: instance of `keras.models.Model`.
            Reference of the model being trained.

    The `logs` dictionary that callback methods
    take as argument will contain keys for quantities relevant to
    the current batch or epoch.

    Currently, the `.fit()` method of the `Sequential` model class
    will include the following quantities in the `logs` that
    it passes to its callbacks:

        on_epoch_end: logs include `acc` and `loss`, and
            optionally include `val_loss`
            (if validation is enabled in `fit`), and `val_acc`
            (if validation and accuracy monitoring are enabled).
        on_batch_begin: logs include `size`,
            the number of samples in the current batch.
        on_batch_end: logs include `loss`, and optionally `acc`
            (if accuracy monitoring is enabled).
    '''

    def __init__(self):
        pass

    def onEpochBegin(self, epoch, logs={}):
        pass

    def onEpochEnd(self, epoch, logs={}):
        pass

    def onBatchBegin(self, batch, logs={}):
        pass

    def onBatchEnd(self, batch, logs={}):
        pass

    def onTrainBegin(self, logs={}):
        pass

    def onTrainEnd(self, logs={}):
        pass


class BaseLogger(Callback):
    '''Callback that accumulates epoch averages of
    the metrics being monitored.

    This callback is automatically applied to
    every model.
    '''

    def __init__(self, metrics, verbose=True):
        self.__metrics = metrics
        self.__seen = 0
        self.__verbose = verbose
        self.log = logging.getLogger(__name__)

        self.__totals = {}
        for metric in self.__metrics:
            self.__totals[metric] = 0.0

    def onEpochBegin(self, epoch, logs={}):
        self.__seen = 0
        self.__totals = {}

        for metric in self.__metrics:
            self.__totals[metric] = 0.0

    def onBatchEnd(self, batch, logs={}):
        batch_size = logs.get('batchSize', 0)
        self.__seen += batch_size

        for k, v in logs.items():
            if k in self.__totals:
                self.__totals[k] += v * batch_size

    def onEpochEnd(self, epoch, logs={}):
        for k in self.__metrics:
            if k in self.__totals:
                # make value available to next callbacks
                logs[k] = self.__totals[k] / self.__seen

        if self.__verbose:
            info = ""
            for k, v in logs.iteritems():
                info += ' - %s:' % k
                info += ' %.6f' % v

            self.log.info(info)

class SaveModel:

    def save(self):
        not In



class SaveBestLossCallback(Callback):
    pass

