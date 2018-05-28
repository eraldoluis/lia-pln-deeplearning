#!/usr/bin/env python
# -*- coding: utf-8 -*-

class Color:

	def __init__(self, red, green, blue, scale = 255):
		"""
		Initiates a new color.
		
		:param red: Amount of red in color to be created.
		:param green: Amount of green in color to be created.
		:param blue: Amount of blue in color to be created.
		:param scale: The maximum value the amounts of color can reach.
		"""
		self.r = red
		self.g = green
		self.b = blue
		self.scale  = scale
		
		(self.r, self.g, self.b) = self.normalize(scale)
		
	def normalize(self, scale):
		"""
		Normalizes color properties so they can be used in graphic.
		:param scale: Scale proportion to be normalized.
		:return list: Normalized color properties.
		"""
		r = self.r / float(scale)
		b = self.b / float(scale)
		g = self.g / float(scale)
		
		return [r, g, b]

	def raw(self):
		"""
		Returns color parameters as a list.
		:return list: Color properties.
		"""
		return (self.r, self.g, self.b)