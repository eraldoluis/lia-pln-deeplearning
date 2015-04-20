#!/usr/bin/env python
# -*- coding: utf-8 -*-

class Datum:
  def __init__(self, word, label, lexiconIndex):
    self.word = word
    self.label = label
    self.lexIndex = lexiconIndex
    self.features = []
    self.guessLabel = []
#    self.previousLabel = []
