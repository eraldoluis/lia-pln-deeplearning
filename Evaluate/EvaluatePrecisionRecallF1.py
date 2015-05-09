import theano
import theano.tensor as T
import numpy


class EvaluatePrecisionRecallF1:
        
    def __init__(self,numberClasses):
        self.numClasses = numberClasses
        self.cl_idx = numpy.arange(self.numClasses)[:, numpy.newaxis]
        
    def evaluate(self,predicts,corrects):
        y = theano.shared(numpy.array(corrects),borrow=True)
        y_pred = theano.shared(numpy.array(predicts),borrow=True)
        
        true_pos = ( T.eq(y, 1) * T.eq(y_pred, 1)).sum()
        false_pos = (T.eq(y, 0) * T.eq(y_pred, 1)).sum()
        false_neg = (T.eq(y, 1) * T.eq(y_pred, 0)).sum()
        
        recall = true_pos / T.cast((true_pos + false_neg),'float64')
        prec = true_pos / T.cast((true_pos  + false_pos),'float64')
        f1_ = 2.0 * prec * recall / (prec + recall)
        
        f = theano.function([],[recall,prec,f1_]); 
        
        rec,prec,f1 = f();
        
        return (rec,prec,f1)