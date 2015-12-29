#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod


import codecs
import os
import random
import util
import re
import sys
import NNet
import logging


class UnknownGenerateStrategy:
    __metaclass__ = ABCMeta
    
    unknownNameDefault = u'UUUNKKK'
    
    def getUnknownStr(self):
        return UnknownGenerateStrategy.unknownNameDefault
    
    @abstractmethod
    def generateUnkown(self,wv):
        pass
    
class RandomUnknownStrategy(UnknownGenerateStrategy):
    
    def generateUnkown(self,wv,wvFile):
        dim =  len(wv.itervalues().next())
        
        return NNet.Util.FeatureVectorsGenerator().generateVector(dim)

class MeanLessShowedWordsUnknownStrategy(UnknownGenerateStrategy):
     
    def __init__(self,meanSize):
        self.meanSize = meanSize
     
    def generateUnkown(self,wv,wvFile):
        wordVectorFile = codecs.open(wvFile , "r", encoding='utf8');
        words = []
        dim =  len(wv.itervalues().next())
        
        wordVectorFile.readline()
        
        for line in wordVectorFile:
            
            words.append(line.rstrip().split(' ', 1 )[0])
        
        
        if self.meanSize <1 and self.meanSize>0:
            numberWordToUse = int(len(words) * self.meanSize)                    
        else:     
            numberWordToUse = int(self.meanSize)
        
        if numberWordToUse > len(words):
            raise NameError('O n�mero de palavras para serem usadas na constru��o  � maior que o n�mero de palavras no dicion�rio')
        
        
        averageWordVector = [0.0] * dim
        numberWordToUseAux = numberWordToUse
        
        while numberWordToUseAux > 0:
            w  = wv[words.pop()]

            for i in range(len(w)):
                averageWordVector[i] += w[i]
            
            numberWordToUseAux -= 1
            
        for i in range(len(averageWordVector)):
            averageWordVector[i] = averageWordVector[i]/numberWordToUse
            
            
        return averageWordVector
         
     
class ChosenUnknownStrategy(UnknownGenerateStrategy):
    
    def __init__(self,unknownName):
        self.__unknownName= unknownName
        self.__randomUnknownStrategy = RandomUnknownStrategy()
    
    def getUnknownStr(self):
        return self.__unknownName
    
    def generateUnkown(self,wv,wvFile):
        if self.__unknownName in wv:
            return wv[self.__unknownName]
        
        return self.__randomUnknownStrategy.generateUnkown(wv,wvFile)


class WordFeatureGenerator:
    
    __metaclass__ = ABCMeta

    @staticmethod
    def getParserArgument(self):
        pass
    
    @abstractmethod
    def generate(self, args):
        pass
    
class Word2VecGenerate(WordFeatureGenerator):
    
    def __init__(self, w2vPath, unknownGenerateStrategy,logger):
        self.__w2vPath = w2vPath
        self.__filters = []
        self.__unknownGenerateStrategy = unknownGenerateStrategy
        self.__logger = logger
    
    def addFilter(self, filter):
        self.__filters.append(filter)
    
    @staticmethod
    def readW2VFile(filename):
        file = codecs.open(filename, 'r', 'utf-8')
        a = 0

        
        dict = {}
        
        for line in file:
            if re.search('^[0-9]+ [0-9]+$', line):
                if a > 0:
                    raise Exception('Foi encontrado mais de uma linha no arquivo que cont�m o n�mero de exemplos e tamanho do word vector')
                a += 1
                continue;
            line = line.rstrip()
            line_split = line.split(' ');
            
            
            dict[line_split[0]] = list(map(float, line_split[1:]))  
            
        file.close()
        
        return dict
    
    @staticmethod
    def parseW2vArguments(w2vArguments):
        parStrW2vPassed = w2vArguments.split()
        parStrW2v = {}
        i = 0
        
        while i < len(parStrW2vPassed):
            optionName = parStrW2vPassed[i] 
            if optionName[0] == '-':
                index = 1
            else:
                index = 0
                
            if os.path.isfile(parStrW2vPassed[i + 1]):
                    parStrW2vPassed[i + 1] = os.path.basename(parStrW2vPassed[i + 1])
            
            parStrW2v[optionName[index:]] = parStrW2vPassed[i + 1]
                
            i += 2
        
        return parStrW2v
    
    def __preprocessText(self, token):
        for filter in self.__filters:
            token = filter.filter(token)
            
        return token
    
    def __iterateByToken(self, line, tokenLabelSeparator):
        tokensWithLabels = line.split(' ')
        
        i = 0
        
        for token in tokensWithLabels:
            if token.isspace() or not token:
                continue
            
            # Removendo as tags dos token
            if tokenLabelSeparator != None and tokenLabelSeparator != "":
                t = token.rsplit(tokenLabelSeparator,1)
                
                if len(t[1]) == 0:
                    logging.getLogger("Logger").warn("It was not found the label from "\
                                 "the token " + token + ". We give to this token "\
                                 " a label equal to"\
                                 " the tokenLabelSeparator( " + tokenLabelSeparator +")" )
                      
                    t[1] = tokenLabelSeparator
                str = t[0]
            else:
                str = token
            
            i += 1
            char = ' ' if i != len(tokensWithLabels) else '\n'
            
            yield str, char
    
    def __createDicitionary(self, lines, perc, tokenLabelSeparator):
        databaseDict = {}
        
        for line in lines:
            for token, separator in self.__iterateByToken(line, tokenLabelSeparator):
                token = self.__preprocessText(token)
                
                if token not in databaseDict:
                    databaseDict[token] = 1
                else:
                    databaseDict[token] += 1
                    
        sortedVec = sorted(databaseDict.items(), key=lambda item: item[1])
        
        numberWordToRemove = int(perc * len(sortedVec))
        
        dict = set()
        
        for word, freq in sortedVec:
            if numberWordToRemove != 0:
                numberWordToRemove -= 1
                continue
            
            dict.add(word)
            
        return dict;
        
    def __createDatasetFiltered(self, datasetPath, datasetFilteredFilePath, percWordsBeRemoved , tokenLabelSeparator):
        dataset = codecs.open(datasetPath, 'r', 'utf-8')
        
        if os.path.isfile(datasetFilteredFilePath):
            return
        
        datasetFilteredFilePath = codecs.open(datasetFilteredFilePath, 'w', 'utf-8')
        
        if percWordsBeRemoved > 0.0:
            lines = dataset.readlines()
            dictionary = self.__createDicitionary(lines,percWordsBeRemoved,tokenLabelSeparator)
        else:
            dictionary = None
            lines = dataset
        
        for line in lines:
            line = self.__preprocessText(line)
            
            for token, separator in self.__iterateByToken(line,tokenLabelSeparator):
                if dictionary != None and token not in dictionary:
                    token = self.__unknownGenerateStrategy.getUnknownStr()
                
                datasetFilteredFilePath.write(token)
                datasetFilteredFilePath.write(separator)
                
    @staticmethod
    def parsedW2vArgumentsToString(parsedw2vArguments):
        parStrW2v = ""
        for key, value in parsedw2vArguments.iteritems():
            
            if len(key) < 3:
                end = len(key)
            else:
                end = 3
                
            parStrW2v += key[:end] + "_" + value + "_"
            
        return parStrW2v
    
    def __getFileNameAndPath(self, datasetPath, dirDataWillBeSaved, fileNamePattern, w2vArguments, percWordsBeRemoved,isTxt):
        parStrW2v = Word2VecGenerate.parsedW2vArgumentsToString(Word2VecGenerate.parseW2vArguments(w2vArguments))
        
        extension = ".txt" if isTxt else ".wv"
        
        wvFileOutputName = fileNamePattern + "_" + parStrW2v + "_" + str(percWordsBeRemoved) + extension
        
        return (wvFileOutputName, os.path.join(dirDataWillBeSaved, wvFileOutputName))
    
    def dataExist(self, datasetPath, dirDataWillBeSaved, fileNamePattern, w2vArguments, percWordsBeRemoved=0.0):
        wvFileOutputPath = self.__getFileNameAndPath(datasetPath, dirDataWillBeSaved, fileNamePattern, 
                                                       w2vArguments, percWordsBeRemoved,False)[1]
    
        
        return os.path.isfile(wvFileOutputPath)
    
    def __addUnknownToW2v(self,wordVector,wvFileOutputPath):
        unknownToken = self.__unknownGenerateStrategy.getUnknownStr()
        
        if unknownToken in wordVector:
            return
        
        unknownWordVector = self.__unknownGenerateStrategy.generateUnkown(wordVector,wvFileOutputPath)
        
        wordVectorFile = codecs.open(wvFileOutputPath , "a", encoding='utf8');
        
        wordVectorFile.write(unknownToken)
        wordVectorFile.write(' ')
        
        for i in range(len(unknownWordVector)):    
            wordVectorFile.write(str(unknownWordVector[i]))
            wordVectorFile.write(' ')
            
        wordVectorFile.write('\n')
        wordVector[unknownToken] = unknownWordVector
        
    def generate(self, datasetPath, dirDataWillBeSaved, fileNamePattern, w2vArguments, seed, percWordsBeRemoved=0.0, 
                 tokenLabelSeparator=""):        
        
        if percWordsBeRemoved > 1.0 or percWordsBeRemoved < 0:
            raise NameError("O valor do numberWordToUse tem intervalor de [0,1] quando � passado como uma percentagem. Valor atual " + str(percWordsBeRemoved))
        
        if percWordsBeRemoved != None and (self.__unknownGenerateStrategy == None or self.__unknownGenerateStrategy == ""): 
            raise NameError("The value of percentage of words be removed was set, but the unkownToken was not set.")
        
        datasetFilteredInfo =  self.__getFileNameAndPath(datasetPath, dirDataWillBeSaved, fileNamePattern, w2vArguments, percWordsBeRemoved,True)
        datasetFilteredFileOutputName = datasetFilteredInfo[0]
        datasetFilteredFilePath = datasetFilteredInfo[1]
                
                
        self.__createDatasetFiltered(datasetPath, datasetFilteredFilePath, percWordsBeRemoved, tokenLabelSeparator)
        
        wvFileInfo = self.__getFileNameAndPath(datasetPath, dirDataWillBeSaved, fileNamePattern, w2vArguments, percWordsBeRemoved,False) 
        wvFileOutputName = wvFileInfo[0]
        wvFileOutputPath = wvFileInfo[1]  
        
        isWVNew = False
        if not os.path.isfile(wvFileOutputPath):            
            arguments = "-train " + datasetFilteredFileOutputName + " -output " + wvFileOutputName + " " + w2vArguments + " -seed " + str(seed)
            
            """
            The wordvec is executed in the directory where the file will be saved, because
            the word2vec doesn't support big string as arguments. As the file name plus the directory path
            normally is a big string, so was decided to executed the w2v direct in the directory to just be necessary
            to pass the file name to w2v. 
            """
            util.util.execProcess([self.__w2vPath] + arguments.split(), self.__logger, dirDataWillBeSaved)
            
            isWVNew = True
            
        
        wordVector = Word2VecGenerate.readW2VFile(wvFileOutputPath)
        
        if isWVNew and percWordsBeRemoved == 0.0:
            self.__addUnknownToW2v(wordVector,wvFileOutputPath)
        
                
        return wordVector
        

class InterporlationGenerate(WordFeatureGenerator):
    
    def __init__(self, word2VecGenerate):
        self.__word2VecGenerate = word2VecGenerate
        
    def __getIntermediaryFilePath(self, nmIntermediaryFile, dirWVWillBeSaved, fileNamePattern):        
        numberLinesToGetInSource = nmIntermediaryFile
        numberLinesToGetInTarget = 1
        n = nmIntermediaryFile + 1
        
        while numberLinesToGetInSource > 0:
            pS = float(numberLinesToGetInSource) / n
            pT = float(numberLinesToGetInTarget) / n
            
            fimeName = "%.2f_" % pS + "_%.2f_" % pT + fileNamePattern + ".dataset"
            path = os.path.join(dirWVWillBeSaved, fimeName)
            
            yield path,numberLinesToGetInSource,numberLinesToGetInTarget
            
            numberLinesToGetInSource -= 1
            numberLinesToGetInTarget += 1
    
    def __generateIntermediaryFile(self, sourceFilePath, targetFilePath, nmIntermediaryFile, dirWVWillBeSaved, fileNamePattern):
        fileSource = codecs.open(os.path.join(sourceFilePath), "r", encoding='utf8');
        fileTarget = codecs.open(os.path.join(targetFilePath) , "r", encoding='utf8');
        
        databaseSourceLines = [];
        databaseTargetLines = [];
        
        for line in fileSource:
            databaseSourceLines.append(line.strip())
            
        for line in fileTarget:
            databaseTargetLines.append(line.strip())
        
        files = []
        
        for fileNamePath,numberLinesToGetInSource,numberLinesToGetInTarget in self.__getIntermediaryFilePath(nmIntermediaryFile, 
                                                                                    dirWVWillBeSaved, fileNamePattern):
            indexLineSource = range(len(databaseSourceLines))
            indexLineTarget = range(len(databaseTargetLines))
            
            
            random.shuffle(indexLineSource)
            random.shuffle(indexLineTarget)
            
            files.append(fileNamePath)
            
            if os.path.isfile(fileNamePath):
                continue
            
            fileToWrite = codecs.open(fileNamePath, "w", encoding='utf8');
            
            numberLineWriteSource = 0
            numberLineWriteTarget = 0
            
            keepGoing = True
            while keepGoing :
                for i in range(numberLinesToGetInSource):
                    if len(indexLineSource) == 0: 
                        keepGoing = False
                        break
                    
                    fileToWrite.write(databaseSourceLines[indexLineSource.pop()])
                    fileToWrite.write("\n")
                    numberLineWriteSource += 1
                    
                for i in range(numberLinesToGetInTarget):
                    if len(indexLineTarget) == 0:
                        keepGoing = False
                        break
                    
                    fileToWrite.write(databaseTargetLines[indexLineTarget.pop()])
                    fileToWrite.write("\n")
                    numberLineWriteTarget += 1
                
                if len(indexLineTarget) == 0 or len(indexLineSource) == 0:
                    keepGoing = False
                
        return files
    
    def dataExist(self, sourceFilePath, targetFilePath, nmIntermediaryFile, dirDataWillBeSaved, fileNamePattern, w2vArguments, percWordsBeRemoved=0.0):
        for fileNamePath,numberLinesToGetInSource,numberLinesToGetInTarget in self.__getIntermediaryFilePath(nmIntermediaryFile, dirDataWillBeSaved, fileNamePattern):
            if not self.__word2VecGenerate.dataExist(fileNamePath, dirDataWillBeSaved, fileNamePattern, 
                                                 w2vArguments,  percWordsBeRemoved):
                return False
            
            return True
        
    def generate(self, sourceFilePath, targetFilePath, nmIntermediaryFile, w2vArguments, dirFilesWillBeSaved, fileNamePattern, seed, percWordsBeRemoved=0.0, tokenLabelSeparator=""):
        """Create a list of word embedding generated from the intermediary datasets using the Word2Vec. These datasets
        are generated using the target and source and the number of theses datasets are controlled 
        by the nmIntermediaryFile parameter which receive a integer number bigger than 0. For Instance:
        
        nmIntermediaryFile    % sentences from source/% sentences from target
        1                        50%/50%
        2                        66%/33%    33%/66%
        3                        75%/25%    50%/50%    25%/75%
        4                        80%/20%    60%/40%    40%/60%    20%/80%
        .
        .
        """
        
        intermediaryFiles = self.__generateIntermediaryFile(sourceFilePath, targetFilePath, nmIntermediaryFile, dirFilesWillBeSaved, fileNamePattern)
        
        wvs = []
        
        for f in intermediaryFiles:
            fileNamePatternIntermediary = util.util.removeExtension(util.util.getFileNameInPath(f))
            wv = self.__word2VecGenerate.generate(f, dirFilesWillBeSaved, fileNamePatternIntermediary, 
                                                  w2vArguments, seed, percWordsBeRemoved, tokenLabelSeparator)
            wvs.append(wv)
        
        return wvs
    
    
    
class AverageGenerator(WordFeatureGenerator):
    def __getFilePath(self, dirFilesWillBeSaved, fileNamePattern):
        avgName = "avg_" + fileNamePattern + ".wv"
        return  os.path.join(dirFilesWillBeSaved, avgName)
        
    def dataExist(self, dirFilesWillBeSaved, fileNamePattern):
        return os.path.isfile(self.__getFilePath(dirFilesWillBeSaved, fileNamePattern))
        
        
    def generate(self, sourceWv, targetWv, dirFilesWillBeSaved, fileNamePattern,unknownGenerateStrategy):        
        filePath = self.__getFilePath(dirFilesWillBeSaved, fileNamePattern)
        wvFiles = [sourceWv, targetWv]
        unknownToken = unknownGenerateStrategy.getUnknownStr()
        
        if not os.path.isfile(filePath):
            newWVFile = codecs.open(filePath, "w", encoding='utf8')
            
            wordsSet = set()
            dicts = []
            
            for wv in wvFiles:
                for k in wv.keys():
                    wordsSet.add(k)
                dicts.append(wv)
            
            lengthVector = len(dicts[0].itervalues().next())
            newWVFile.write(str(len(wordsSet)) + " " + str(lengthVector) + "\n")
            avgWordVector = {}
            
            for word in wordsSet:
                avg = []
                
                for j in range(lengthVector):
                    avg.append(0.0)
                    
                for dict in dicts:
                    if word in dict:
                        v = dict[word]
                    else:
                        v = dict[unknownToken]
                    
                    for j in range(len(v)):
                        avg[j] += v[j]
            
                numbersOfDicts = len(dicts)
                
                
                newWVFile.write(word)
                newWVFile.write(' ')
                
                for j in range(len(avg)):
                    avg[j] = avg[j] / numbersOfDicts
                    
                    
                    newWVFile.write(str(avg[j ]))
                    newWVFile.write(' ')
                
                
                avgWordVector[word] = avg
                newWVFile.write('\n')
        else:
            avgWordVector =  Word2VecGenerate.readW2VFile(filePath)
        
        return avgWordVector
    
class RandomWeightGenerator(WordFeatureGenerator):
    def __getFilePath(self, dirFilesWillBeSaved, fileNamePattern):
        avgName = "random_inter_" + fileNamePattern + ".wv"
        return  os.path.join(dirFilesWillBeSaved, avgName)
        
    def dataExist(self, dirFilesWillBeSaved, fileNamePattern):
        return os.path.isfile(self.__getFilePath(dirFilesWillBeSaved, fileNamePattern))
        
        
    def generate(self,sourceWv, targetWv, dirFilesWillBeSaved, fileNamePattern):        
        newWVFileName = self.__getFilePath(dirFilesWillBeSaved, fileNamePattern)
        
        
        if not os.path.isfile(newWVFileName):
            newWVFile = codecs.open(newWVFileName, "w", encoding='utf8')
            wordsSet = set()
            dicts = []
            
            biggestValueWv = sys.float_info.min
            smallestValueWv = sys.float_info.max
            
            wvFiles =[sourceWv,targetWv]
            for wv in wvFiles:
                for k in wv.keys():
                    wordsSet.add(k)
                    
                    wvDic = wv[k]
                    
                    for v in wvDic:
                        if v > biggestValueWv:
                            biggestValueWv = v
                            
                        if v < smallestValueWv:
                            smallestValueWv = v
                            
                lengthVector = len(wv.itervalues().next())
           
            newWVFile.write(str(len(wordsSet)) + " " + str(lengthVector) + "\n")
            randomWordVector = {}
            
            for word in wordsSet:
                
                newWVFile.write(word)
                newWVFile.write(' ')
                randomVec = []
                
                for j in range(lengthVector):
                    randomVec.append(random.uniform(smallestValueWv, biggestValueWv))
                    
                    newWVFile.write(str(randomVec[j]))
                    newWVFile.write(' ')
                
                
                randomWordVector[word] = randomVec
                newWVFile.write('\n')
        else:
            randomWordVector =  Word2VecGenerate.readW2VFile(newWVFileName)

        
        return randomWordVector