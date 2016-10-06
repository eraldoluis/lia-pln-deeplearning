#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
This class saves and load the  persistent objects in h5py file.
"""
import h5py

from persistence.PersistentManager import PersistentManager


class H5py(PersistentManager):
    def __init__(self, filePath):
        self.__f = h5py.File(filePath, "a")

    def save(self, persistentObject):
        objName = persistentObject.getName()
        objAttributes = persistentObject.getAttributes()

        if not objName:
            raise Exception("Object Name is None")

        # Create a group with the object name
        attrGrp = self.__f.require_group(objName)

        # Create datasets for each object attribute
        for attrName, attrValue in objAttributes.iteritems():
            if not attrName in attrGrp:
                self.__f.create_dataset(attrName, data=attrValue)
            else:
                self.__f[attrName] = attrValue

    def load(self, persistentObject):
        objName = persistentObject.getName()

        if not objName:
            raise Exception("Object Name is None")

        persistentObject.load(self.__f[objName])

    def close(self):
        self.__f.flush()
        self.__f.close()

    def addAttribute(self, attrName, attrValue):
        self.__f.attrs[attrName] = attrValue

    def getAttribute(self, attrName):
        return self.__f.attrs[attrName]

    def __setitem__(self, key, value):
        self.addAttribute(key, value)

    def __getitem__(self, item):
        return self.getAttribute(item)
