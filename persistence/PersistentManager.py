#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
This class responsible to save and load the attributes of object in a certain database.
Each object will have a unique name, because this manager will use that to search for attributes of the object.
You can create and retrieve metadata for data in a dataset. This metadata are called attributes.
As the objects, the metadata name needs to be unique.
"""
from abc import ABCMeta, abstractmethod


class PersistentManager(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def getObjAttributesByObjName(self, objName):
        """
        Return the attributes of object with a certain name

        :param objName: name of the object
        :return: a dictionary with keys and values are, respectively, the names and values of attributes.
        """
        raise NotImplementedError()

    @abstractmethod
    def save(self,persistentObject):
        """
        Save the attributes of an object

        :type persistence.PersistentObject.PersistentObject
        :param persistentObject: Objects that will be saved
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def load(self, persistentObject):
        """
        Load the attributes of an object

        :type persistence.PersistentObject.PersistentObject
        :param persistentObject: Object that will be loaded
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def close(self):
        """
        Close the manager
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def addAttribute(self,attrName, attrValue):
        """
        Store metadata to the data.

        :param attrName: the name of the metadata. This name need to be unique.
        :param attrValue: the value of the metadata
        :return: boolean. If the attribute was saved with success.
        """
        raise NotImplementedError()

    @abstractmethod
    def getAttribute(self, attrName):
        """
        Retrieve the metadata

        :param attrName: the name of the metadata. This name need to be unique.
        :return: the value of the metadata
        """
        raise NotImplementedError()
