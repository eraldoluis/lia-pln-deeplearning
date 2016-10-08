#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
This class represents a object that can be persisted.
The objects of this class has a name that needs to be unique in the persistence environment.

This class has the responsibility to return a dictionary with the attribute names and values which it wants to save and
load the attributes by itself.
"""
from abc import ABCMeta, abstractmethod


class PersistentObject(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def getName(self):
        """
        Return the object name

        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def getAttributes(self):
        """
        Return a dictionary with keys and values are, respectively, the names and values of attributes.

        :return: {}
        """
        raise NotImplementedError()

    @abstractmethod
    def load(self, attributes):
        """
        Load persisted attributes in object
        :type {}
        :param attributes: a dictionary with keys and values are, respectively, the names and values of attributes.

        :return: None
        """
        raise NotImplementedError()