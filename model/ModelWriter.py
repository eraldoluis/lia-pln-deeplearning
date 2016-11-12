#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import logging

from time import time

from persistence import H5py
from persistence.H5py import H5py


class ModelWriter(object):
    '''
    This class saves model and its parameters in a hDF5 file
    '''

    def __init__(self, savePath, listOfPersistentObjs, args, parametersToSave):
        """
        :param savePath: path where the model will be save

        :param listOfPersistentObjs: list of PersistentObject. These objects represents the  necessary data to be saved.

        :param args: object with all parameters

        :param parametersToSave: a list with the name of parameters which need to be saved.
        """
        self.__h5py = H5py(savePath)
        self.__listOfPersistentObjs = listOfPersistentObjs
        self.__log = logging.getLogger(__name__)

        parameters = {}

        for parameterName in parametersToSave:
            parameters[parameterName] = getattr(args, parameterName)

        self.__h5py.addAttribute("parameters", json.dumps(parameters))

    def save(self):
        begin = int(time())

        for obj in self.__listOfPersistentObjs:
            if obj.getName():
                self.__h5py.save(obj)

        self.__log.info("Model Saved in %d", int(time()) - begin)