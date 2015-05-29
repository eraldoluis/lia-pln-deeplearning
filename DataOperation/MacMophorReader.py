#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
'''
import re
import codecs

class MacMorphoReader:
    
    def __init__(self,fileWithFeatures):
        self.fileWithFeatures = fileWithFeatures 
     
    def readTestData(self, filename,lexicon,lexiconOfLabel,separateSentences=True,filters=[],unkownDataTest=None):
        return self.readData(filename, lexicon, lexiconOfLabel,None,separateSentences,False,filters,None,unkownDataTest)
    
    def addToken(self, lexicon, wordVecs, addWordUnkown, filters, indexesBySentence, word,setWordsInDataSet,unkownData):
        for f in filters:
            word = f.filter(word)
                
        lexiconIndex = lexicon.getLexiconIndex(word)

        if addWordUnkown and lexicon.isUnknownIndex(lexiconIndex):
            lexiconIndex = lexicon.put(word)
            wordVecs.append(None)
        
        if setWordsInDataSet is not None:
            setWordsInDataSet.add(word)
            
        if lexicon.isUnknownIndex(lexiconIndex) and unkownData is not None:
            unkownData.append(word)
        
        indexesBySentence.append(lexiconIndex)

    def addLabel(self, lexiconOfLabel, labelsBySentence, token):
        labelsBySentence.append(lexiconOfLabel.put(token))

    def readTokenAndLabelOfFileWithFeature(self, lexicon, lexiconOfLabel, wordVecs, addWordUnkown, filters, indexesBySentence, labelsBySentence, token,setWordsInDataSet,unkownData):
        prefWord = 'word='
        
        if prefWord in token:
            word = token[len(prefWord):]
            self.addToken(lexicon, wordVecs, addWordUnkown, filters, indexesBySentence, word,setWordsInDataSet,unkownData)
        elif re.search(r'^([A-Z]|\W)', token) is not None:
            self.addLabel(lexiconOfLabel, labelsBySentence, token)
    
    def readTokenAndLabelOfRawFile(self, lexicon, lexiconOfLabel, wordVecs, addWordUnkown, filters, indexesBySentence, labelsBySentence, token,setWordsInDataSet,unkownData):
        s = token.split('_')
        
        self.addToken(lexicon, wordVecs, addWordUnkown, filters, indexesBySentence, s[0],setWordsInDataSet,unkownData)
        self.addLabel(lexiconOfLabel, labelsBySentence, s[1])
    

    def readData(self, filename,lexicon,lexiconOfLabel, wordVecs=None, separateSentences= True, addWordUnkown=False,filters=[], setWordsInDataSet=None,unkownDataTest=[]):
        '''
        Read the data from a file and return a matrix which the first row is the words indexes  and second row is the labels values
        '''
        data = [[],[]]
        indexes = data[0]
        labels = data[1]
        
        func = self.readTokenAndLabelOfFileWithFeature if self.fileWithFeatures else self.readTokenAndLabelOfRawFile
        
        f = codecs.open(filename, 'r', 'utf-8')
        a = 0
        
        for line in f:
            
#             a +=1
#                   
#             if a ==10:
#                 break;
            
            line_split = line.split()
            # Ignore empty lines.
            if len(line_split) < 2:
                continue
            
            if separateSentences:
                indexesBySentence = []
                labelsBySentence = []
                unkownDataBySentence = [] if unkownDataTest is not None else None
            else:
                indexesBySentence = indexes
                labelsBySentence = labels
                unkownDataBySentence = unkownDataTest
            
            for token in line_split:
                func(lexicon, lexiconOfLabel, wordVecs, addWordUnkown, filters, indexesBySentence, labelsBySentence, token,setWordsInDataSet,unkownDataBySentence)
            
            if len(indexesBySentence) != len(labelsBySentence):
                raise Exception('Número de tokens e labels não são iguais na linha: ' + line)
                         
            if separateSentences:
                indexes.append(indexesBySentence)
                labels.append(labelsBySentence)
                
                if unkownDataTest is not None:
                    unkownDataTest.append(unkownDataBySentence)
                
        f.close()
        
#         list = []
#         i = 0
#         for l in data[0]:
#             if len(l) < 5:
#                 list.append((i,len(l)))
#             i+=1
# 
#         print list
        return data
    
    
    
 