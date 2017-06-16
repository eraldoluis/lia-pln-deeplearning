#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os.path
from Color import Color
from Plotter import Plotter

class Graphic:
	red = Color(255, 0, 0)
	green = Color(0, 255, 0)
	blue = Color(0, 0, 255)
	magenta = Color(255, 0, 255)
	cyan = Color(0, 255, 255)
	yellow = Color(255, 255, 0)
	black = Color(0, 0, 0)
	grey = Color(128, 128, 128)
	white = Color(255, 255, 255)
	
	colors = [red, green, blue, magenta, cyan, yellow, black, grey]
	
	def __init__(self, root=str()):
		"""
		Initiates graphical configuration.
		:type root: str
		:param root: Root directory from which files should be fetched from.
		"""
		self.root = root.strip().rstrip('/\\') + '/'
		self.range = slice(None, None, None)
		self.source = []
		
		self.labelX = None
		self.labelY = None
	
	def addfile(self, filename, label, color = None):
		"""
		Adds a file to the fetching list.
		:param filename: File name to be added to list.
		:param label: Label given to data presented from this file.
		:param color: Color to be given to data presented from this file.
		:return bool: Could the file be added?
		"""
		filename = self.root + filename.lstrip('/\\')
		
		if not os.path.isfile(filename):
			return False
		
		if color is None:
			color = Graphic.colors[len(self.source) % 8]
		elif not isinstance(color, Color):
			color = Color(*color)
		
		self.source.append(type('obj', (object,), {
			'path': filename, 'label': label, 'color': color
		}))
		
		return True
	
	def interval(self, start=None, limit=None, step=None):
		"""
		Sets an interval of data to be displayed in graphic.
		:param start: Starting value of interval.
		:param limit: Limiting value of interval.
		:param step: Size of steps to be jumped.
		:return self: Allows method chaining.
		"""
		self.range = slice(start, limit, step)
		return self
	
	def xlabel(self, label):
		"""
		Sets a label for the x-axis.
		:param label: Label for the x-axis.
		:return self: Allows method chaining.
		"""
		self.labelX = label
		return self
	
	def ylabel(self, label):
		"""
		Sets a label for the y-axis.
		:param label: Label for the y-axis.
		:return self: Allows method chaining.
		"""
		self.labelY = label
		return self
	
	def plot(self, filter, property, axis = (0,1)):
		"""
		Plots the graphic and exhibits it.
		:param filter: The filter to use within files' data.
		:param property: Property to be plotted.
		:param axis: Value interval for the y-axis.
		"""
		plotter = Plotter(self)
		plotter.plot(filter, property, axis)
		plotter.clean()