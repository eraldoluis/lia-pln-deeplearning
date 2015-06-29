#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
'''
import re
from Data.WordVector import WordVector
from Data.Lexicon import Lexicon
import numpy as np

class FeatureFactory:

    def __init__(self,wordVectorSize,charVectorSize):
        self.__lexicon = Lexicon(); 
        self.__lexiconOfLabel = Lexicon();
        self.__wordVecs = WordVector(wordVectorSize);
        if charVectorSize:
            self.__charcon = Lexicon(); 
            self.__charVecs = WordVector(charVectorSize);
             
        
    def readWordVectors(self, vecfilename, vocabfilename):
        '''
        Load a dictionary from vocabfilename and the vector (embedding)
        for each word from vecfilename.
        '''
        # Read the vocabulary file (one word per line).
        fVoc = open(vocabfilename, 'r')
        for line in fVoc:
            word = line.strip()
            # Ignore empty lines.
            if len(word) == 0:
                continue
            self.__lexicon.put(word)
        fVoc.close()

        # Read the vector file (one word vector per line).
        # Each vector represents a word from the vocabulary.
        # The vectors must be in the same order as in the vocabulary.
        fVec = open(vecfilename, 'r')
        for line in fVec:
            self.__wordVecs.append([float(num) for num in line.split()])
        fVec.close()
        
        self.__wordVecs.setLenWordVectorAuto()

        assert (self.__wordVecs.getLength() == self.__lexicon.getLen())

    def readCharVectors(self, vecfilename, vocabfilename):
        '''
        Load a dictionary from vocabfilename and the vector (embedding)
        for each char from vecfilename.
        '''
        # Read the vocabulary file (one word per line).
        fVoc = open(vocabfilename, 'r')
        for line in fVoc:
            char = line.strip()
            # Ignore empty lines.
            if len(char) == 0:
                continue
            self.__charcon.put(char)
        fVoc.close()

        # Read the vector file (one word vector per line).
        # Each vector represents a char from the vocabulary.
        # The vectors must be in the same order as in the vocabulary.
        fVec = open(vecfilename, 'r')
        for line in fVec:
            self.__charVecs.append([float(num) for num in line.split()])
        fVec.close()
        
        self.__charVecs.setLenWordVectorAuto()

        assert (self.__charVecs.getLength() == self.__charcon.getLen())
        
    def readData(self, filename,addWordUnknown=False):
        '''
        Read the data from a file and return a matrix which the first row is the words indexes  and second row is the labels values
        '''
        data = [[],[]]
        indexes = data[0]
        labels = data[1]
        
        if addWordUnknown:
            self.__lexicon.put("UNNKN")
            self.__lexicon.put("<s>")
            self.__lexicon.put("</s>")
           

        f = open(filename, 'r')
        for line in f:
            line_split = line.split()
            # Ignore empty lines.
            if len(line_split) < 2:
                continue
            i = 0
            for i in range(0,len(line_split)):  
                if(line_split[i].find("word=")!=-1):
                    word = line_split[i][5:]
                    lexiconIndex = self.__lexicon.getLexiconIndex(word)
            
                    if addWordUnknown and lexiconIndex == 0:
                        lexiconIndex = self.__lexicon.put(word)
                        #self.__wordVecs.append(None)
                      
                    indexes.append(lexiconIndex)

                    while (line_split[i].find("word")!=-1):
                        i+=1
                    labels.append(self.__lexiconOfLabel.put(line_split[i]))

        f.close()
        
        if addWordUnknown:
            self.__wordVecs.startRandomAllVecs(self.__lexicon.getLen())

        return data

    def readDataWithChar(self, filename,addWordUnknown=False,addCharUnknown=False):
        '''
        Read the data from a file and return a matrix which the first row is the words indexes, second row is the labels values
                 the third row is the char indexes of each word 
        '''
        data = [[],[],{},[]]
        wordIndexes = data[0]
        wordLabels = data[1]
        charIndexes = data[2]
        numCharsOfWord = data[3]
        
        
        if addCharUnknown:
            self.__charcon.put("UNNKN")
            self.__charcon.put("<s>")
            self.__charcon.put("</s>")
            numCharsOfWord.append(1)
            numCharsOfWord.append(1)
            numCharsOfWord.append(1)
            
            
        if addWordUnknown:
            charIndexes[0] = [self.__lexicon.put("UNNKN")]
            charIndexes[1] = [self.__lexicon.put("<s>")]
            charIndexes[2] = [self.__lexicon.put("</s>")]
             
             
        f = open(filename, 'r')
        for line in f:
            line_split = line.split()
            # Ignore empty lines.
            if len(line_split) < 2:
                continue
            
            for i in range(0,len(line_split)):  
                if(line_split[i].find("word=")!=-1):
                    word = line_split[i][5:]
                    word = re.sub(r'[1-9]',"0",word)
                    
                    
                    
                    lexiconIndex = self.__lexicon.getLexiconIndex(word)
                                   
                    if lexiconIndex == 0 and addWordUnknown:
                            lexiconIndex = self.__lexicon.put(word)
                            
                    if charIndexes.get(lexiconIndex) is None:  
                    
                        wordCharIndex = []
                        for char in word:
                            charconIndex = self.__charcon.getLexiconIndex(char)
                            if addCharUnknown and charconIndex == 0:
                                charconIndex = self.__charcon.put(char)
                            wordCharIndex.append(charconIndex)    
            
                        charIndexes[lexiconIndex] = wordCharIndex
                        numCharsOfWord.append(len(word))
                        

                    wordIndexes.append(lexiconIndex)
                    
                                         
                    while (line_split[i].find("word")!=-1):
                        i+=1
                    wordLabels.append(self.__lexiconOfLabel.put(line_split[i]))
    
        f.close()
       
        if addWordUnknown:
            self.__wordVecs.startRandomAllVecs(self.__lexicon.getLen())
   
        if addCharUnknown:
            self.__charVecs.startRandomAllVecs(self.__charcon.getLen())
       
        return data
    
    def readTestData(self, filename):
        '''
        Read the data from a file and return a matrix which the first row is the words indexes, second row is the labels values
                 the third row is the char indexes of each word 
        '''
        data = [[],[]]
        wordIndexes = data[0]
        wordLabels = data[1]
        numTotal = 0
        num = 0     
        f = open(filename, 'r')
        for line in f:
            line_split = line.split()
            # Ignore empty lines.
            if len(line_split) < 2:
                continue
            
            for i in range(0,len(line_split)):  
                if(line_split[i].find("word=")!=-1):
                    word = line_split[i][5:]
                    word = re.sub(r'[1-9]',"0",word)
                    
                    lexiconIndex = self.__lexicon.getLexiconIndex(word)
                    wordIndexes.append(lexiconIndex)
                    
                    numTotal +=1               
                    if lexiconIndex:
                        num +=1
                                                                         
                    while (line_split[i].find("word")!=-1):
                        i+=1
                    wordLabels.append(self.__lexiconOfLabel.put(line_split[i]))
        f.close()
        
        return data  
    
    def getLexiconLen(self):
        '''
        Return the number of words in the lexicon.
        '''
        return self.__lexicon.getLen()

    def getLexicon(self):
        return self.__lexicon
    
    def getNumberOfLabel(self):
        return self.__lexiconOfLabel.getLen()
   
    def getLexiconOfLabel(self):
        return self.__lexiconOfLabel

    def getWordVector(self):
        return self.__wordVecs
      
    def getCharcon(self):
        return self.__charcon
      
    def getCharVector(self):
        return self.__charVecs
    
    def getDict(self):
        return self.__lexicon.getDict()
    
    def getDict2(self):
        return self.__charcon.getDict()