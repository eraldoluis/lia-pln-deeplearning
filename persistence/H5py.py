#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
This class saves and load the  persistent objects in h5py file.
"""
import h5py

from persistence.PersistentManager import PersistentManager

11
class H5py(PersistentManager):
    def __init__(self, filePath):
        self.__f = h5py.File(filePath, "r")

    def getObjAttributesByObjName(self, objName):
        if not objName:
            raise Exception("Object Name is None")

        return self.__f[objName]

    def save(self, persistentObject):
        objName = persistentObject.getName()
        objAttributes = persistentObject.getAttributes()

        if not objName:
            raise Exception("Object Name is None")

        # Create a group with the object name
        attrGrp = self.__f.require_group(objName)

        # Create datasets for each object attribute
        for attrName, attrValue in objAttributes.iteritems():
            if  attrName not in attrGrp:
                attrGrp.create_dataset(attrName, data=attrValue)
            else:
                attrGrp[attrName][...] = attrValue

    def objectExist(self,objName):
        return objName in self.__f

    def load(self, persistentObject):
        objName = persistentObject.getName()

        persistentObject.load(self.getObjAttributesByObjName(objName))

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
