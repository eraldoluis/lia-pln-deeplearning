#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

class DebugPlot:
  def saveHist(values, bin, xlabel, ylabel, title, filename):
	  plt.figure()
	  
	  #l = plt.plot(bins, 'r--', linewidth=1)
	  
	  plt.xlabel(xlabel)
	  plt.ylabel(ylabel)
	  plt.title(title)
	  if numpy.amin(values)!= numpy.amax(values):
	      plt.xlim((numpy.amin(values), numpy.amax(values)))
	  else:
	      values = numpy.append(values,values[0]-1)
	      plt.xlim((numpy.amin(values), numpy.amax(values)))
	      
	  n, bins, patches = plt.hist(values)        
	  plt.grid(True)
	  
	  plt.savefig(filename)