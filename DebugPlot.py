#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy
#import matplotlib
#matplotlib.use('Agg')

import matplotlib.pyplot as plt

class DebugPlot:
    
    def saveHist(self, values, b, xlabel, ylabel, title, filename):
        
        plt.figure()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        
        if numpy.amin(values) != numpy.amax(values):
            plt.xlim((numpy.amin(values), numpy.amax(values)))
        else:
            values = numpy.append(values, values[0] - 1)
            plt.xlim((numpy.amin(values), numpy.amax(values)))
            
        plt.hist(values, bins=b)
        plt.grid(True)
        plt.savefig(filename)
        
        plt.clf()
        plt.close()
