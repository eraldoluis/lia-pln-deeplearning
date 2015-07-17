from _collections import deque



class EvaluateEveryNumEpoch:
    
    def __init__(self,numEpochToTrain,numEpochToEval,evaluate,model,inputData,correct,inputDataRaw,unknownDataTestCharIdxs):
        self.model = model
        self.inputData = inputData
        self.inputDataRaw = inputDataRaw
        self.correct = correct
        self.unknwnDataTest = unknownDataTestCharIdxs
        self.evaluate = evaluate
        self.deque = deque();
        
        
        if len(numEpochToEval) > 1:
            self.deque.extend(sorted(numEpochToEval, key=int) )
        else:
            if numEpochToEval[0] < 1:
                return
            
            a = numEpochToEval[0]
            
            while a < numEpochToTrain:
                self.deque.append(a)
                a += numEpochToEval[0]        
        
    
    def afterEpoch(self,numEpoch):
        
        if len(self.deque) == 0:
            return;
        
        numEpochToEval = self.deque[0];
        
        if numEpochToEval > 0 and numEpoch != numEpochToEval :
            return;
        
        print 'Testing...'        
        predicts = self.model.predict(self.inputData,self.inputDataRaw,self.unknwnDataTest)
        self.evaluate.evaluateWithPrint(predicts,self.correct)
        
        self.deque.popleft()