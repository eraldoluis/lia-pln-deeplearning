#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
'''
import re
import codecs

class MacMorphoReader:
    
    def __init__(self,fileWithFeatures):
        self.fileWithFeatures = fileWithFeatures 
     
    def readTestData(self, filename,lexicon,lexiconOfLabel,lexiconRaw,separateSentences=True,addWordUnknown=False,withCharwnn=False,charVars=[None,None,{},[]],addCharUnknown=False,filters=[],unknownDataTest=None,unknownDataTestCharIdxs=None):
        return self.readData(filename, lexicon, lexiconOfLabel,lexiconRaw,None,separateSentences,False,withCharwnn,charVars,False,filters,None,unknownDataTest,unknownDataTestCharIdxs)
    
    def addToken(self, lexicon, wordVecs, addWordUnknown, filters, indexesBySentence, rawWord,setWordsInDataSet,unknownData,lexiconRaw,indexesOfRawBySentence,numCharsOfRawBySentence,withCharwnn,charVars,addCharUnknown,unknownDataCharIdxs):
        
        charcon = charVars[0]
        charVector = charVars[1]
        charIndexesOfLexiconRaw = charVars[2]
        numCharsOfLexiconRaw = charVars[3]
        
        for f in filters[:-1]:
            rawWord = f.filter(rawWord)
            
        word = filters[-1].filter(rawWord)        
        lexiconIndex = lexicon.getLexiconIndex(word)
        

        if addWordUnknown and lexicon.isUnknownIndex(lexiconIndex):
            lexiconIndex = lexicon.put(word)
            wordVecs.append(None)
        
        if setWordsInDataSet is not None:
            setWordsInDataSet.add(word)
            
        if lexicon.isUnknownIndex(lexiconIndex) and unknownData is not None:
            unknownData.append(word)
        
        indexesBySentence.append(lexiconIndex)
        
        if withCharwnn:
            
            lexiconIndex = lexiconRaw.getLexiconIndex(rawWord)
            if lexiconRaw.isUnknownIndex(lexiconIndex):
                lexiconIndex = lexiconRaw.put(rawWord)
                numCharsOfLexiconRaw.append(len(rawWord))
                            
                charIndexes = []
                for char in rawWord:
                    idx = charcon.getLexiconIndex(char)
                    if addCharUnknown and charcon.isUnknownIndex(idx):
                        idx = charcon.put(char)
                        charVector.append(None)
                    charIndexes.append(idx)
                charIndexesOfLexiconRaw[lexiconIndex] = charIndexes
                if unknownDataCharIdxs is not None:
                    unknownDataCharIdxs.append(charIndexes) 
                    
            indexesOfRawBySentence.append(lexiconIndex)
            numCharsOfRawBySentence.append(len(rawWord))        
            
        

    def addLabel(self, lexiconOfLabel, labelsBySentence, token):
        labelsBySentence.append(lexiconOfLabel.put(token))
                                        
    def readTokenAndLabelOfFileWithFeature(self, lexicon, lexiconOfLabel, wordVecs, addWordUnknown, filters, indexesBySentence, labelsBySentence,lexiconRaw, indexesOfRawBySentence, numCharsOfRawBySentence , token,setWordsInDataSet,unknownData,withCharwnn,charVars,addCharUnknown,unknownDataCharIdxs):
                
        prefWord = 'word='
        
        if prefWord in token:
            word = token[len(prefWord):]
            self.addToken(lexicon, wordVecs, addWordUnknown, filters, indexesBySentence, word,setWordsInDataSet,unknownData,
                          lexiconRaw,indexesOfRawBySentence,numCharsOfRawBySentence,withCharwnn,charVars,addCharUnknown,unknownDataCharIdxs)
        elif re.search(r'^([A-Z]|\W)', token) is not None:
            self.addLabel(lexiconOfLabel, labelsBySentence, token)
                                
    def readTokenAndLabelOfRawFile(self, lexicon, lexiconOfLabel, wordVecs, addWordUnknown, filters, indexesBySentence, labelsBySentence,lexiconRaw, indexesOfRawBySentence, numCharsOfRawBySentence,token, setWordsInDataSet,unknownData,withCharwnn,charVars,addCharUnknown,unknownDataCharIdxs):
        s = token.split('_')
        
        #assert (len(s[0])>0)
        #assert (len(s[1])>0)
        
        if (len(s[0]) and len(s[1])):
            self.addToken(lexicon, wordVecs, addWordUnknown, filters, indexesBySentence, s[0],setWordsInDataSet,unknownData,
                          lexiconRaw,indexesOfRawBySentence,numCharsOfRawBySentence,withCharwnn,charVars,addCharUnknown,unknownDataCharIdxs)
        
            self.addLabel(lexiconOfLabel, labelsBySentence, s[1])
        else:
	    print 'the','[',token,']',s[0],s[1]
    

    def readData(self, filename,lexicon,lexiconOfLabel,lexiconRaw, wordVecs=None, separateSentences=True, addWordUnknown=False,withCharwnn=False,charVars=[None,None,{},[]],addCharUnknown=False,filters=[], setWordsInDataSet=None,unknownDataTest=[],unknownDataTestCharIdxs=None):
        '''
        Read the data from a file and return a matrix which the first row is the words indexes, second row is the labels values and 
        the third row has the indexes of the raw words, and the fourth the number of chars each word in the training set has.
        '''
        data = [[],[],[],[]]
        indexes = data[0]
        labels = data[1]
        indexesOfRaw = data[2]
        numCharsOfRaw = data[3]
        
        print 'hi',len(charVars[2]),len(charVars[3])
        
        func = self.readTokenAndLabelOfFileWithFeature if self.fileWithFeatures else self.readTokenAndLabelOfRawFile
        
        f = codecs.open(filename, 'r', 'utf-8')
        
        
        for line in f:
            
            line_split = line.split()
            # Ignore empty lines.
            if len(line_split) < 2:
                continue
            
            if separateSentences:
                indexesBySentence = []
                labelsBySentence = []
                indexesOfRawBySentence = []
                numCharsOfRawBySentence = []
                unknownDataBySentence = [] if unknownDataTest is not None else None
            else:
                indexesBySentence = indexes
                indexesOfRawBySentence = indexesOfRaw
                numCharsOfRawBySentence = numCharsOfRaw
                labelsBySentence = labels
                unknownDataBySentence = unknownDataTest
            
            for token in line_split:
                func(lexicon, lexiconOfLabel, wordVecs, addWordUnknown, filters, indexesBySentence, labelsBySentence,
                      lexiconRaw,indexesOfRawBySentence, numCharsOfRawBySentence,token,setWordsInDataSet,unknownDataBySentence, withCharwnn,charVars,addCharUnknown,unknownDataTestCharIdxs)
            
            if len(indexesBySentence) != len(labelsBySentence):
                raise Exception('Número de tokens e labels não são iguais na linha: ' + line)
                         
            if separateSentences:
                indexes.append(indexesBySentence)
                labels.append(labelsBySentence)
                indexesOfRaw.append(indexesOfRawBySentence)
                numCharsOfRaw.append(numCharsOfRawBySentence)
                
                if unknownDataTest is not None:
                    unknownDataTest.append(unknownDataBySentence)
                
        f.close()
        print len(charVars[2]),len(charVars[3])
        #assert(charVars[2]==charVars[3])
        
        return data
    
    
    