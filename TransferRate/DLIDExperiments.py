#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
import os
import logging
import codecs
import calendar
import time
import random
import math
import datetime
from TransferRate.WordFeatureGenerator import Word2VecGenerate,\
    InterporlationGenerate, AverageGenerator, RandomWeightGenerator,\
    RandomUnknownStrategy, MeanLessShowedWordsUnknownStrategy,\
    ChosenUnknownStrategy
from DataOperation.WordVector import WordVector
from util.util import getFileNameInPath, removeExtension
from TransferRate import CRFSuite
import Postag
import importlib
from DataOperation import Lexicon


# import resource
# import sqlite3
# import database

class DLIDExperiments:
    
    w2vStrategy = ["all", "just_source", "without_source", "just_intermediate", "nothing", "without_intermediate"]
    intermediateStrategy = ["random_interpolation", "avg", "files", "random"]
    unknownWordStrategy = ["random", "mean_vector", "word_vocab"]
     
    def getArgumentParser(self):
        parser = argparse.ArgumentParser();
        
        subparsers = parser.add_subparsers(title='Algorithm',
                                       dest='algorithm',
                                       description='The algorithm that will be used to train and test ')
        
        
        base_parser = argparse.ArgumentParser(add_help=False)
        
        base_parser.add_argument('--numberExperiments', dest='numberExperiments', type=int, action='store', default=1
                                 , help='The number of experiments that will be made  using this paramaters')
        
        base_parser.add_argument('--remakeExperiments', dest='remakeExperiment', action='store_true', required=True
                                 , help='All the experiments will label by a number. For instance, if it has never done some experiments with'\
                                 'certain parameters and was set to do 5 experiments, so the first experiment will be labeled'\
                                 ' with 1, the second with 2 and so on. If someone did a number of experiments and need to'\
                                 ' more experiments using the same parameters. So this person need to set experiments numbers'\
                                 ' and set this argument to false, thus the script will label the new experiments with' \
                                 ' differents labels used in a another tests. But if the person set this argument to true, ' \
                                 ' so the script will overwrite data from another test with the same label.')
        
        base_parser.add_argument('--source', dest='source', action='store',
                       help='The source path', required=True)
        
        base_parser.add_argument('--target', dest='target', action='store',
                           help='The target path', required=True)
        
        base_parser.add_argument('--argWordVector', dest='argWordVector', action='store', required=True,
                           help='The arguments which will be pass to w2v')
        
        base_parser.add_argument('--numberEpoch', dest='numberEpoch', type=int, action='store',
                           help='The number of epochs used to train the model', required=True)
        
        base_parser.add_argument('--percWordsBeRemoved', dest='percWordsBeRemoved', type=float, action='store',
                           help='', default=0.05)
        
        base_parser.add_argument('--dirData', dest='dirData', action='store', required=True,
                           help='The place where the word vectors and datasets used to generate the word vectors will be saved')
        
        base_parser.add_argument('--dirOutputTrain', dest='dirOutputTrain', action='store', required=True,
                           help='The place where the model and output of the each experiment will be saved')
        
        base_parser.add_argument('--percGenerateUnkownWord', dest='percGenerateUnkownWord', type=float, action='store',
                           help='', default=0.00)
        
        
        base_parser.add_argument('--additionalWordVector', dest='additionalWordVector', action='store', 
                                 default=[], nargs='*',
                                 help ="The script will read these word vectors and "/
                                 'concatenate with the word vectors generated')
        
        
        base_parser.add_argument('--loadModel', dest='loadModel', action='store',
                           help='The file path where the model is stored')
        
        base_parser.add_argument('--numperepoch', dest='numPerEpoch', action='store', nargs='*', type=int,
                           help="The evaluation on the test corpus will "
                                + "be performed after a certain number of training epoch."
                                + "If the value is an integer, so the evalution will be performed after this value of training epoch "
                                + "If the value is a list of integer, than the evaluation will be performed when the epoch is equal to one of list elements.  ", default=None)
    
        
       
        base_parser.add_argument('--useW2vStrategy', dest='useW2vStrategy', action='store', 
                                 choices=DLIDExperiments.w2vStrategy, default=DLIDExperiments.w2vStrategy[0],
                                 help = 'Specify which word vector( the source, target or intermediate word vectors) '/
                                 'will be used.'/
                                 ' For instance: You can use just intermediate word vectors or the target and source word vectors ')
        
        
        base_parser.add_argument('--intermediateStrategy', dest='intermediateStrategy', choices=DLIDExperiments.intermediateStrategy,
                            default=DLIDExperiments.intermediateStrategy[0], action='store', required=True,
                            help = 'Set which inter')

        base_parser.add_argument('--intermediateFiles', dest='intermediateFiles', 
                                 action='store', default=[], nargs='*',
                                 help = 'This argument set the intermediate datasets that will be used in DLID')
        
        base_parser.add_argument('--numberOfIntermediateDataset', dest='numberOfIntermediateDataset', type=int, action='store',
                           help='This argument set how many intermediate datasets will be generated and used ' \
                           'on DLID.', default=1)
        
        typeOfNormalizationStrategy = ["none", "mean", "without_change_signal", "z_score"]
        
        base_parser.add_argument('--typeOfNormalizationWV', dest='typeOfNormalizationWV', choices=typeOfNormalizationStrategy,
                            default=typeOfNormalizationStrategy[0], action='store', required=False)
        
        #     parser.add_argument('--useJustW2vSource', dest='useJustW2vSource', action='store_true')
        defaultSeed = calendar.timegm(time.gmtime()) + random.randint(-10000000, 10000000)
        
        base_parser.add_argument('--seed', dest='seed', action='store', type=int, default=defaultSeed)
        
        base_parser.add_argument('--w2vPath', dest='w2vPath', action='store', required=True,
                                 help="The path where is the word2vec executable")
        
        base_parser.add_argument('--tokenLabelSeparator', dest='tokenLabelSeparator', action='store', required=False,default="_",
                            help="Specify the character that is being used to separate the token from the label in the dataset.")
        
        
        base_parser.add_argument('--startSymbol', dest='startSymbol', action='store', default="<s>",
                           help='The symbol that represents the beginning of a setence')
        
        base_parser.add_argument('--endSymbol', dest='endSymbol', action='store', default="</s>",
                           help='The symbol that represents the ending of a setence')
        
        base_parser.add_argument('--unknownwordstrategy', dest='unknownWordStrategy', action='store', default=DLIDExperiments.unknownWordStrategy[0]
                            , choices=DLIDExperiments.unknownWordStrategy,
                           help='Choose the strategy that will be used for constructing a word vector of a unknown word.'
                           + 'There are three types of strategy: random(generate randomly a word vector) ,' + 
                           ' mean_all_vector(generate the word vector from the mean of all words)' + 
                           ', word_vocab(use a word vector of one particular word. You have to use the parameter unknownword to set this word)')
        
        base_parser.add_argument('--unknownword', dest='unknownWord', action='store', default=False,
                           help='The word which will be used to represent the unknown word')
        
        base_parser.add_argument('--filters', dest='filters', action='store', default=[], nargs='*',
                       help='The filters which will be applied to the data. You have to pass the module and class name.' + 
                       ' Ex: modulename1 classname1 modulename2 classname2')
        
        
        
        crfSuiteParser = subparsers.add_parser('crfsuite', help='CRF Suite', 
                                       parents=[base_parser])
        
        crfSuiteParser.add_argument('--argCRF', dest='argCRF', action='store',
                           help='The arguments which will be pass to CRFSuite', default="")
        
        crfSuiteParser.add_argument('--useManualFeature', dest='useManualFeature', action='store_true')
        
        crfSuiteParser.add_argument('--windowSize', dest='windowSize', type=int, action='store',
                           help='', default=5)
        
        parser.add_argument('--withCharwnn', dest='withCharwnn', action='store_true',
                           help='Set training with character embeddings')
        
        nnParser = subparsers.add_parser('nn', help='Neural Network', 
                                       parents=[base_parser])
        
        algTypeChoices = ["window_word", "window_sentence"]
        
        nnParser.add_argument('--alg', dest='alg', action='store', default="window_sentence", choices=algTypeChoices,
                           help='The type of algorithm to train and test')
        
        nnParser.add_argument('--hiddenlayersize', dest='hiddenSize', action='store', type=int,
                           help='The number of neurons in the hidden layer', default=300)
        
        nnParser.add_argument('--convolutionalLayerSize', dest='convSize', action='store', type=int,
                           help='The number of neurons in the convolutional layer', default=50)
        
        nnParser.add_argument('--wordWindowSize', dest='wordWindowSize', action='store', type=int,
                           help='The size of words for the wordsWindow', default=5)
        
        nnParser.add_argument('--charWindowSize', dest='charWindowSize', action='store', type=int,
                           help='The size of char for the charsWindow', default=5)
        
        nnParser.add_argument('--batchSize', dest='batchSize', action='store', type=int,
                           help='The size of the batch in the train', default=1)
        
        nnParser.add_argument('--lr', dest='lr', action='store', type=float , default=0.0075,
                           help='The value of the learning rate')
    
        nnParser.add_argument('--c', dest='c', action='store', type=float , default=0.0,
                           help='The larger C is the more the regularization pushes the weights of all our parameters to zero.')    
        
        nnParser.add_argument('--wordVecSize', dest='wordVecSize', action='store', type=int,
                           help='Word vector size', default=100)
        
        nnParser.add_argument('--charVecSize', dest='charVecSize', action='store', type=int,
                           help='Char vector size', default=10)
        
        nnParser.add_argument('--testoosv', dest='testOOSV', action='store_true', default=False,
                           help='Do the test OOSV')
        
        nnParser.add_argument('--testoouv', dest='testOOUV', action='store_true', default=False,
                           help='Do the test OOUV')
        
       
        
        lrStrategyChoices = ["normal", "divide_epoch"]
        
        nnParser.add_argument('--lrupdstrategy', dest='lrUpdStrategy', action='store', default=lrStrategyChoices[0], choices=lrStrategyChoices,
                           help='Set the learning rate update strategy. NORMAL and DIVIDE_EPOCH are the options available')
            
        nnParser.add_argument('--filewithfeatures', dest='fileWithFeatures', action='store_true',
                           help='Set that the training e testing files have features')
        
        vecsInitChoices = ["randomAll", "random", "zeros","z_score","normalize_mean"]
        
        nnParser.add_argument('--charVecsInit', dest='charVecsInit', action='store', default=vecsInitChoices[1], choices=vecsInitChoices,
                           help='Set the way to initialize the char vectors. RANDOM, RANDOMALL, ZEROS, Z_SCORE and NORMALIZE_MEAN are the options available')
        
#         nnParser.add_argument('--wordVecsInit', dest='wordVecsInit', action='store', default=vecsInitChoices[1], choices=vecsInitChoices,
#                            help='Set the way to initialize the char vectors. RANDOM, RANDOMALL, ZEROS, Z_SCORE and NORMALIZE_MEAN are the options available')
#         
        nnParser.add_argument('--charwnnwithact', dest='charwnnWithAct', action='store_true',
                           help='Set training with character embeddings')
        
        
        networkChoices= ["complete","without_hidden_update_wv" ,"without_update_wv"]
        
        nnParser.add_argument('--networkChoice', dest='networkChoice', action='store',default=networkChoices[0],choices=networkChoices)
        
        parser.add_argument('--mean_size', dest='meanSize', action='store', type=float, default=1.0,
                           help='The number of the least used words in the train for unknown word' 
                           + 'Number between 0 and 1 for percentage, number > 1 for literal number to make the mean and negative for mean_all')
        
        vecsUpStrategyChoices = ["normal", "normalize_mean","z_score"]
    
        nnParser.add_argument('--wordvecsupdstrategy', dest='wordVecsUpdStrategy', action='store', default=vecsUpStrategyChoices[0], choices=vecsUpStrategyChoices,
                           help='Set the word vectors update strategy. NORMAL, NORMALIZE_MEAN and Z_SCORE are the options available')
        
        nnParser.add_argument('--charvecsupdstrategy', dest='charVecsUpdStrategy', action='store', default=vecsUpStrategyChoices[0], choices=vecsUpStrategyChoices,
                           help='Set the char vectors update strategy. NORMAL, NORMALIZE_MEAN and Z_SCORE are the options available')
        
        return parser


def getDictionaryName(datasetName, percWordsBeRemoved):
    return datasetName + "_" + str(percWordsBeRemoved)

def calculateBiggestSmallestAvgWv(dictWVByWord, biggestsValueWv, smallestsValueWv, avgValueWv):
    for dimension in range(len(dictWVByWord.itervalues().next())):
        if biggestsValueWv != None:
            biggestsValueWv.append(sys.float_info.max * -1)
        
        if smallestsValueWv != None:
            smallestsValueWv.append(sys.float_info.max)
        
        if avgValueWv != None:
            avgValueWv.append(0.0)
            
        for word in dictWVByWord:
            wv = dictWVByWord[word]
            if biggestsValueWv != None and wv[dimension] > biggestsValueWv[dimension]:
                biggestsValueWv[dimension] = wv[dimension]
            if smallestsValueWv != None and wv[dimension] < smallestsValueWv[dimension]:
                smallestsValueWv[dimension] = wv[dimension]
            
            if avgValueWv != None:
                avgValueWv[dimension] += wv[dimension]
                
    if avgValueWv != None:
        for dimension in range(len(dictWVByWord.itervalues().next())):
            avgValueWv[dimension] = avgValueWv[dimension] / len(dictWVByWord)

def calculateStandardDeviationWv(dictWVByWord, avgValueWv, stdValueWv):
    for dimension in range(len(dictWVByWord.itervalues().next())):    
        stdValueWv.append(0.0)
        
        for word in dictWVByWord:
            wv = dictWVByWord[word]
            
            stdValueWv[dimension] += pow((wv[dimension] - avgValueWv[dimension]), 2)
                
    for dimension in range(len(dictWVByWord.itervalues().next())):
        stdValueWv[dimension] = math.sqrt(stdValueWv[dimension] / len(dictWVByWord))
    
    

def getFilePattern(experimentNumber, percWordsBeRemoved, name):
    dictionaryNameWithoutExtension = getDictionaryName(name,percWordsBeRemoved)
    fileNamePattern = dictionaryNameWithoutExtension + "_" + str(experimentNumber)
    
    return fileNamePattern

def getWordVector(word2VecGenerate,args,experimentNumber,name,logger):
    fileNamePattern = getFilePattern(experimentNumber, args.percWordsBeRemoved, name)
    
    exist = word2VecGenerate.dataExist(args.source, args.dirData, fileNamePattern, 
                                  args.argWordVector, args.percWordsBeRemoved)
        
    if  exist:
        logger.info("O wordvector " + name + " já existe")
        
            
    w = word2VecGenerate.generate(args.source, args.dirData, fileNamePattern, 
                                  args.argWordVector, args.seed, args.percWordsBeRemoved, args.tokenLabelSeparator)
    
    return w,exist


def doOneExperiment(mainExperimentDir, experimentNumber, args, w2vStrategy, intermediateStrategy,typeOfNormalizationStrategy,unknownWordStrategy, remakeExperiment):
    experimentDirName = str(experimentNumber); 
    
    outputDirPath = os.path.join(args.dirOutputTrain, experimentDirName) 
    outputModelDirPath = os.path.join(outputDirPath, 'model') 
    outputFeaturedDataseDirPath = os.path.join(outputDirPath, 'dataset')
    outputIntermediaryDataseDirPath = os.path.join(outputDirPath, 'intermediate')
    
    
    numberPrints = 2
    
    os.mkdir(outputDirPath)
    os.mkdir(outputModelDirPath)
    os.mkdir(outputFeaturedDataseDirPath)
    os.mkdir(outputIntermediaryDataseDirPath)
    
    logger = logging.getLogger("Logger")
    
    fileHandler = logging.FileHandler(os.path.join(outputDirPath, 'output.log'))
    formatter = logging.Formatter('[%(asctime)s]\t%(message)s')
    
    fileHandler.setFormatter(formatter)

    logger.addHandler(fileHandler) 
    logger.setLevel(logging.INFO)

    
    logger.info("Iniciando experimento " + str(experimentNumber) + " com seguintes argumentos: " + str(args))
    logger.info("Horário de início: " + time.ctime())
    logger.info("Pasta de origem: " + mainExperimentDir)
    
    # Ver quais são os arquivos que serão gerados
    if args.useW2vStrategy == w2vStrategy[0]:  # ALL
        useSource = True
        useTarget = True
        useIntermediate = True
    elif args.useW2vStrategy == w2vStrategy[1]:  # JUST_SOURCE
        useSource = True
        useTarget = False
        useIntermediate = False
    elif args.useW2vStrategy == w2vStrategy[2]:  # WITHOUT_SOURCE
        useSource = False
        useTarget = True
        useIntermediate = True
    elif args.useW2vStrategy == w2vStrategy[3]:  # JUST_INTERMEDIARY
        useSource = False
        useTarget = False
        useIntermediate = True
    elif args.useW2vStrategy == w2vStrategy[4]:  # NOTHING
        useSource = False
        useTarget = False
        useIntermediate = False
    elif args.useW2vStrategy == w2vStrategy[5]:  # WITHOUT_INTERMEDIARY
        useSource = True
        useTarget = True
        useIntermediate = False
    else:
        useSource = True
        useTarget = True
        useIntermediate = True
        
    parW2v = Word2VecGenerate.parseW2vArguments(args.argWordVector)
    
    if int(parW2v["-threads"]) != 1:
        raise Exception("O parametro que permite escolher o numero de threads do word2vec foi configurado para um numero diferente de 1."
                                + "Este parametro sempre deve ser igual 1.")
        
    
    if args.unknownWordStrategy == unknownWordStrategy[0]:
        unknownGenerateStrategy = RandomUnknownStrategy()
    elif args.unknownWordStrategy == unknownWordStrategy[1]:
        unknownGenerateStrategy = MeanLessShowedWordsUnknownStrategy(args.meanSize)
    elif args.unknownWordStrategy == unknownWordStrategy[2]:
        unknownGenerateStrategy = ChosenUnknownStrategy(args.unknownWord)
    
    
    word2VecGenerate = Word2VecGenerate(args.w2vPath,unknownGenerateStrategy)
    
    filters = ['DataOperation.TransformLowerCaseFilter','TransformLowerCaseFilter']
    
    filters += args.filters
    
    a = 0
    
    while a < len(args.filters):
        print "Usando o filtro: " + args.filters[a] + " " + args.filters[a + 1]
        module_ = importlib.import_module(args.filters[a])
        word2VecGenerate.addFilter(getattr(module_, args.filters[a + 1])())
        a += 2
        
    
    wordVectors = []
    
    sourceName = removeExtension(getFileNameInPath(args.source))
    targetName = removeExtension(getFileNameInPath(args.target))
    experimentHasAlreadyDone = True
    
    targetVector = sourceVector = None
    
    
    if useSource:
        sourceVector,exist = getWordVector(word2VecGenerate, args, experimentNumber, sourceName, logger)
        
        experimentHasAlreadyDone = experimentHasAlreadyDone and exist
                
        logger.info('Using ' + sourceName)
        wordVectors.append(sourceVector)
        
    if useTarget:
        
        targetVector,exist = getWordVector(word2VecGenerate, args, experimentNumber, targetName, logger)
        experimentHasAlreadyDone = experimentHasAlreadyDone and exist
        wordVectors.append(targetVector)
        logger.info('Using ' + targetName)
    
         
    # Se é para usar intermediário e a estratégia dos intermediário é diferente da média dos word vectors
    if useIntermediate:
        if args.intermediateStrategy == intermediateStrategy[0]:
            logger.info("Utilizando interpolação com " + str(args.interpolationFactor))
            
            interpolation = InterporlationGenerate(word2VecGenerate)
            
            fileNamePattern = getFilePattern(experimentNumber, args.percWordsBeRemoved
                                                               , sourceName + "_" + targetName )
            
            exist = interpolation.dataExist(args.source, args.target, args.interpolationFactor, 
                                       args.dirData, fileNamePattern, args.argWordVector, args.percWordsBeRemoved)
            
            experimentHasAlreadyDone = experimentHasAlreadyDone and exist
            
            if exist:
                logger.info("Wordvector Intermediário já existe")
                
            
            wordVectors += interpolation.generate(args.source, args.target, args.interpolationFactor, 
                                       args.dirData, fileNamePattern, args.argWordVector, args.seed, args.percWordsBeRemoved,args.tokenLabelSeparator)
            
            logger.info('Using Interpolation')
            
            logger.info("Terminou de gerar arquivos intermediarios")
        elif args.intermediateStrategy == intermediateStrategy[1]:
            logger.info("Calculando word2vec intermediario a partir da médiz")
            
            averageGenerator = AverageGenerator()
            
            intermediateName = sourceName + "_" + targetName + "_" + Word2VecGenerate.parsedW2vArgumentsToString(parW2v)
            
            fileNamePattern = fileNamePattern = getFilePattern(experimentNumber, args.percWordsBeRemoved,intermediateName)
            
            exist = averageGenerator.dataExist(args.dirData, fileNamePattern)
            experimentHasAlreadyDone = experimentHasAlreadyDone and exist
                                                            
            if exist:
                logger.info("Wordvector Intermediário já existe")
                
            if sourceVector is None:
                sourceVetor = getWordVector(word2VecGenerate, args, experimentNumber, sourceName, logger)
            
            if targetVector is None:
                targetVector = getWordVector(word2VecGenerate, args, experimentNumber, targetName, logger)
                
            avgWordVector = averageGenerator.generate(sourceVetor, targetVector, args.dirData, fileNamePattern,unknownGenerateStrategy)
            
            wordVectors.append(avgWordVector)
            
            logger.info('Using average intermediare')
        
        elif args.intermediateStrategy == intermediateStrategy[2]:
            logger.info("Usando Arquivos intermediarios")
            
            for intermediate in args.intermediateFiles:
                logger.info(intermediate)
                intermediateName = removeExtension(getFileNameInPath(intermediate))
                w, exist = getWordVector(word2VecGenerate, args, experimentNumber, intermediateName, logger)
                experimentHasAlreadyDone = experimentHasAlreadyDone and exist            
                
                wordVectors.append(w)
                
            logger.info('Using intermediare datasets')
                
        elif args.intermediateStrategy == intermediateStrategy[3]:
                  
            logger.info("Calculando w2v aleatório")
            
            randomWeightGenerator = RandomWeightGenerator()
            
            name = sourceName + "_" + targetName + "_" + word2VecGenerate.parsedW2vArgumentsToString(parW2v)
            
            
            fileNamePattern = getFilePattern(experimentNumber, args.percWordsBeRemoved,name)
            exist = randomWeightGenerator.dataExist(args.dirData, fileNamePattern)
            experimentHasAlreadyDone = experimentHasAlreadyDone and exist 
            
            if exist:
                logger.info("Wordvector Intermediário já existe")
                
            if sourceVector is None:
                sourceVetor = getWordVector(word2VecGenerate, args, experimentNumber, sourceName, logger)
            
            if targetVector is None:
                targetVector = getWordVector(word2VecGenerate, args, experimentNumber, targetName, logger)
            
            randomVector = randomWeightGenerator.generate(sourceVetor, targetVector, args.dirData, fileNamePattern)
                        
            wordVectors.append(randomVector)
            logger.info('Using random intermediare')
    
    logger.info("Word vector que serao usados:")
    
    logger.info(args.typeOfNormalizationWV) 
    
    if args.typeOfNormalizationWV != typeOfNormalizationStrategy[0]:
        if args.typeOfNormalizationWV == typeOfNormalizationStrategy[1]:
            for wvDict in wordVectors:                
                biggestsValueWv = []
                smallestsValueWv = []
                avgValueWv = []
                
                calculateBiggestSmallestAvgWv(wvDict, biggestsValueWv, smallestsValueWv, avgValueWv)
                
                logger.info(biggestsValueWv)
                logger.info(smallestsValueWv)
                logger.info(avgValueWv)
                
#                 newWVFile = codecs.open("normalization_avg_"+ os.path.split(nameWordVectorRead)[1], "w", encoding='utf8')
                
                for word in wvDict:
                    wv = wvDict[word]
#                     newWVFile.write(word)
#                     newWVFile.write(' ')
                    
                    for i in range(len(wv)):
                        wv[i] = (wv[i] - avgValueWv[i]) / (biggestsValueWv[i] - smallestsValueWv[i])                        
#                         newWVFile.write(str(wv[i]))
#                         newWVFile.write(' ')
                    
#                     newWVFile.write('\n')
                     
        elif args.typeOfNormalizationWV == typeOfNormalizationStrategy[2]:
            for wvDict in wordVectors:
                biggestsValueWv = []
                smallestsValueWv = []
                avgValueWv = None
                
                calculateBiggestSmallestAvgWv(wvDict, biggestsValueWv, smallestsValueWv, avgValueWv)
                                        
                logger.info(biggestsValueWv)
                logger.info(smallestsValueWv)
                
#                 newWVFile = codecs.open("normalization_wth_cg_sg_"+ os.path.split(nameWordVectorRead)[1], "w", encoding='utf8')
                
                for word in wv:
                    wv = wvDict[word]
#                     newWVFile.write(word)
#                     newWVFile.write(' ')
 
                    
                    for i in range(len(wv)):
                        wv[i] = wv[i] / (biggestsValueWv[i] - smallestsValueWv[i])
#                         newWVFile.write(str(wv[i]))
#                         newWVFile.write(' ')
                        
#                     newWVFile.write('\n')    
        
        elif args.typeOfNormalizationWV == typeOfNormalizationStrategy[3]:
            for wvDict in wordVectors:
                biggestsValuePosWv = None
                smallestsValueNegWv = None
                avgValueNegWv = []
                stdValueWv = []
                
                
                calculateBiggestSmallestAvgWv(wvDict, biggestsValuePosWv, smallestsValueNegWv, avgValueNegWv)
                calculateStandardDeviationWv(wvDict, avgValueNegWv, stdValueWv)
                
                logger.info(biggestsValuePosWv)
                logger.info(smallestsValueNegWv)
                logger.info(avgValueNegWv)
                logger.info(stdValueWv)
                
#                 newWVFile = codecs.open("normalization_z_"+ os.path.split(nameWordVectorRead)[1], "w", encoding='utf8')
                
                for word in wvDict:
                    wv = wvDict[word]
#                     newWVFile.write(word)
#                     newWVFile.write(' ')
                    
                    for i in range(len(wv)):
                        wv[i] = (wv[i] - avgValueNegWv[i]) / (stdValueWv[i])                        
#                         newWVFile.write(str(wv[i]))
#                         newWVFile.write(' ')
                    
#                     newWVFile.write('\n')          

    sourceFile = codecs.open(args.source , "r", "utf-8");
    targetFile = codecs.open(args.target , "r", "utf-8");
    
    unknownTokens = [unknownGenerateStrategy.getUnknownStr()]
    
    if args.algorithm == "crfsuite":
        crf = CRFSuite()
        noTestByEpoch = True if args.numperepoch != None else False
        
        
        if args.loadModel is None:    
            logger.info("Comecando parte do treino")
            
            modelPath = os.path.join(outputModelDirPath, 'm.model')
            
            outputModelDirPathAbs = os.path.abspath(outputModelDirPath)
            modelPathAbs = os.path.abspath(modelPath)
            
            logger.info(outputModelDirPathAbs)
            logger.info(modelPathAbs)
            
            crf.train(args.source,args.target,wordVectors,args.windowSize,args.useManualFeature,args.numberEpoch,noTestByEpoch
                  ,modelPathAbs,unknownTokens)
    
        
        else:
            modelPath = args.loadModel
            
            if os.path.isabs(modelPath):
                modelPathAbs = modelPath
            else:
                modelPathAbs = os.path.abspath(modelPath)
                
        
        if noTestByEpoch:      
            logger.info("Comecando partes do teste")
            numberCorrect, total = crf.test(targetFile, modelPathAbs, wordVectors,
                         args.windowSize, args.useManualFeature, args.numberEpoch, noTestByEpoch,unknownTokens)
                
            logger.info("Número de Corretas:" + str(numberCorrect))
            logger.info("Número de Total:" + str(total))
            logger.info("Acurácia:" + str(float(numberCorrect) / total))
            
            logger.info("Terminando teste: " + time.ctime())
        
        
    elif args.algorithm == "nn":        
        args.unknownwordstrategy = Postag.ParametersChoices.unknownWordStrategy[2]
        args.unknownword = unknownGenerateStrategy.getUnknownStr()
        args.wordVecsInit = "random"
        args.saveModel = None
        
        
        
        wordsSet = set()
        
        dim = 0
        
        unknownTokens = []
        
        for wv in wordVectors:
            wordsSet.add(wordsSet.keys())
            dim += len(wv.itervalues().next())
            
        lexicon = Lexicon()
        wordVector = WordVector(wordSize=dim)
            
        for word in wordsSet:
            newv = []
            
            for wv in wordVectors:
                if word in wv:
                    newv += wv[word]
                else:
                    for unknownToken in unknownTokens:
                        if unknownToken in wv:
                            newv += wv[word]
                            break;
            
            lexicon.put(word);
            wordVector.append(newv)
            
        args.vocab = lexicon
        args.wordVectors = wordVector
    
        Postag.run(Postag.ParametersChoices.algTypeChoices, Postag.ParametersChoices.unknownWordStrategy, 
                   Postag.ParametersChoices.lrStrategyChoices, Postag.ParametersChoices.networkChoices, args)
    
    
        
    logger.removeHandler(fileHandler) 
    

def main():
    
    parser = argparse.ArgumentParser();
    
   
    try:
        args = parser.parse_args();
    except:
        parser.print_help()
        sys.exit(0)
        
    beginExperimentDate = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
    mainExperimentDirName = args.labelExp + "_" + beginExperimentDate if args.labelExp is not None else beginExperimentDate;
    mainExperimentDir = os.path.join(args.dirOutputTrain, mainExperimentDirName) 
    
    experimentNumber = 1
    numberExperimentsDid = 0
    
    
    logger = logging.getLogger("Logger")
    formatter = logging.Formatter('[%(asctime)s]\t%(message)s')
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.setLevel(logging.INFO)
    
    logger.info("Iniciando experimentos com seguintes argumentos: " + str(args))
    logger.info("Seed: " + str(args.seed))
    
    random.seed(args.seed)
        
    
    while numberExperimentsDid < args.numberExperiments:
        if doOneExperiment(mainExperimentDir, experimentNumber, args, DLIDExperiments.w2vStrategy
                           , DLIDExperiments.intermediateStrategy,DLIDExperiments.typeOfNormalizationStrategy
                           , args.remakeExperiments):
            numberExperimentsDid += 1;
        
        experimentNumber += 1
            
    


if __name__ == '__main__':
    main()
