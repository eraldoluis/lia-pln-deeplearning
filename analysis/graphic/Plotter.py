#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from ..extractor import extract

class Plotter:

	def __init__(self, graphic, size = (16,9)):
		"""
		Initiates the plotter.
		:param graphic: Graphic to be plotted.
		"""
		self.clean()
		
		plt.figure(figsize = size)
		self.graphic = graphic
		
	def clean(self):
		"""
		Cleans the plotter up.
		:return void:
		"""
		plt.clf()
		plt.cla()
		plt.close()

	def extract(self, filter, property):
		"""
		Extracts the data to be plotted.
		:param filter: The filter to use within files' data.
		:param property: Property to be plotted.
		:return list: Data to be plotted.
		"""
		return [type('file', (object,), {
			'file': file,
			'data': extract(file.path, filter, property)
		}) for file in self.graphic.source]
		
	def labels(self):
		"""
		Sets the label for graphic's axis.
		:return void:
		"""
		if self.graphic.labelX is not None:
			plt.xlabel(self.graphic.labelX)
		
		if self.graphic.labelY is not None:
			plt.ylabel(self.graphic.labelY)
	
	def plot(self, filter, property, axis=(0, 1)):
		"""
		Plots and shows the graphic.
		:param filter: The filter to use within files' data.
		:param property: Property to be plotted.
		:param axis: Value interval for the y-axis.
		"""
		data = self.extract(filter, property)
		xlen = min([len(d.data) for d in data])
		xval = range(*self.graphic.range.indices(xlen))

		for d in data:
			plt.plot(xval, d.data, color=d.file.color.raw(), lw=2, label=d.file.label)
			
		plt.axis((0, xlen) + axis)
		self.labels()
		
		plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
		plt.show()