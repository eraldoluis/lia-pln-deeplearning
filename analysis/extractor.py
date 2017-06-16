#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import codecs
import logging

log = logging.Logger(__name__)

def rawfilter(raw, property):
	"""
	Return the value of a qualified name property.
	:param raw: Dictionary from which property is fetched.
	:param property: Qualified name property.
	:return:
	"""
	if type(property) in [list, tuple]:
		return [rawfilter(raw, p) for p in property]
	
	for name in property.split('.'):
		
		if not isinstance(raw, dict):
			return None
		
		if name not in raw:
			return None
		
		raw = raw[name]
		
	return raw

def extract(file, filter, properties):
	"""
	Extracts data from file, given the parameters.
	:param file: File to be extracted.
	:param filter: Filter to be applied to lines.
	:param properties: Property to be extracted from filtered lines.
	:return list: Data gathered from file.
	"""
	data = []
	
	with codecs.open(file, mode = "r", encoding = "utf8") as hfile:
		for line in hfile:
			try:
				raw = json.loads(line)
				
				if rawfilter(raw, filter[0]) == filter[1]:
					data.append(rawfilter(raw, properties))
			
			except ValueError as e:
				log.error("Error loading JSON", e)
				
	return data