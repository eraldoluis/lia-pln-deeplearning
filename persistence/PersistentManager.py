#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
This class saves and load the persistent objects.
"""
from abc import ABCMeta, abstractmethod


class PersistentManager(object):
    __metaclass__ = ABCMeta

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
