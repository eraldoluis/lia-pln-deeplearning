from _collections import deque
import numpy as np
import cPickle as pickle

class EvaluateEveryNumEpoch:
    
    def __init__(self, numEpochToTrain, numEpochToEval, evaluate, model, inputData ,correct ,inputDataRaw, unknownDataTestCharIdxs, varsToSave, fileToSave):
        self.model = model
        self.inputData = inputData
        self.inputDataRaw = inputDataRaw
        self.correct = correct
        self.unknownDataTest = unknownDataTestCharIdxs
        self.evaluate = evaluate
        self.deque = deque();
        self.bestAccuracy = 0.0
        self.varsToSave = varsToSave
        self.fileToSave = fileToSave
        self.acc = []
        
        if len(numEpochToEval) > 1:
            self.deque.extend(sorted(numEpochToEval, key=int) )
        else:
            if numEpochToEval[0] < 1:
                return
            
            a = numEpochToEval[0]
            
            while a < numEpochToTrain:
                self.deque.append(a)
                a += numEpochToEval[0]        
        
    
    def afterEpoch(self, numEpoch):
        
        if len(self.deque) == 0:
            return;
        
        numEpochToEval = self.deque[0];
        
        if numEpochToEval > 0 and numEpoch != numEpochToEval:
            return;
        
        print 'Testing...'        
        predicts = self.model.predict(self.inputData, self.inputDataRaw, self.unknownDataTest)
        predicts = np.asarray(predicts).flatten()
        self.correct = np.asarray(self.correct).flatten()
        
        acc = self.evaluate.evaluateWithPrint(predicts, self.correct)
        self.acc.append(acc)
        
        if acc > self.bestAccuracy:
            self.bestAccuracy = acc
            
            if self.fileToSave is not None:
                print 'Saving Model...'
                f = open(self.fileToSave, "w")
                pickle.dump(self.varsToSave, f, pickle.HIGHEST_PROTOCOL)
                
                f.close()
                print 'Model saved with sucess in ' + self.fileToSave
        
        self.deque.popleft()