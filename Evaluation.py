import argparse
import cPickle as pickle
import numpy as np

def euclidian(a,b):
    return np.linalg.norm(a-b, axis=1)

def kl_divergence(a,b):
    aux = np.where(b != 0, np.divide(a,b), 0)
    return np.sum(np.where(aux != 0, a * np.log10(aux), 0),axis = 1)

def norm(a):
    return a/np.sum(a)

def totalUtility(a,amostra,sol):
    u  = np.zeros(a.shape)
    amostra = np.reshape(amostra, (a.shape[0],1))
    print 'a.shape'
    print a.shape
    
    print 'sol len'
    print len(sol)
    
    i = 0
    for elem in u:
        elem[sol[i]] = 1.0
        i = i + 1
    
    #mm = np.sqrt((a)*(a))
    #return np.sum(u*(a)/mm)/float(a.shape[0])
    return np.sum(u*(a)/amostra)/float(a.shape[0]) 


def compTotalUtility(a,b,amostra,sol):
    u  = np.zeros(a.shape)
    amostra = np.reshape(amostra, (a.shape[0],1))
    i = 0
    for elem in u:
        elem[sol[i]] = 1.0
        i = i + 1
    
    #mm = np.sqrt((a-b)*(a-b))
    #return np.sum(u*(a-b)/mm)/float(a.shape[0])
    return np.sum(u*(a-b)/amostra)/float(a.shape[0]) 
    
def utility(a,amostra,sol,n,random_sample):
    u  = np.zeros(a.shape)
    
    amostra_soma = np.zeros(len(amostra))
    for i in range(len(amostra)):
        amostra_soma[i] = np.sum(amostra[:i+1])
        
    
    i = 0
    for elem in u:
        elem[sol[i]] = 1.0
        i = i + 1
    
    utili = 0.0
    
    for i in range(n):
        index = np.searchsorted(amostra_soma, random_sample[i], side='left')
        utili += np.sum(u[index]*(a[index])/amostra[index])      
     
    return utili/float(n)    

def compUtility(a,b,amostra,sol,n,random_sample):
    u  = np.zeros(a.shape)
    
    amostra_soma = np.zeros(len(amostra))
    for i in range(len(amostra)):
        amostra_soma[i] = np.sum(amostra[:i+1])
        
    
    i = 0
    for elem in u:
        elem[sol[i]] = 1.0
        i = i + 1
    
    utili = 0.0
    
    for i in range(n):
        index = np.searchsorted(amostra_soma, random_sample[i], side='left')
        utili += np.sum(u[index]*(a[index]-b[index])/amostra[index])      
     
    return utili/float(n)
    
    
def main():
    
    parser = argparse.ArgumentParser();
    
    evaluationStrategyChoices = ["euclidian", "kl_divergence", "all"]
    
    parser.add_argument('--evaluation', dest='evaluation', action='store', default=evaluationStrategyChoices[2], choices=evaluationStrategyChoices,
                       help='Set the EVALUATION strategy. EUCLIDIAN and KL_DIVERGENCE and ALL are the options available')

    samplingStrategyChoices = ["divergent", "all"]
    
    parser.add_argument('--sampling', dest='sampling', action='store', default=samplingStrategyChoices[0], choices=samplingStrategyChoices,
                       help='Set the SAMPLING strategy. DIVERGENT and ALL are the options available')
    
    
    parser.add_argument('--sampling_size', dest='samplingSize', action='store', type=float, default=1.0,
                       help='The size of the sampling' 
                       + 'Number between 0 and 1 for percentage, number > 1 for literal number of samples')
    
    parser.add_argument('--systemFiles', dest='systemFiles',
                                 action='store', default=[], nargs='*',
                                 help='This argument set the system files that will be used in Evaluation')
    args = parser.parse_args();
    
    k = 0
    
    if len(args.systemFiles) < 2:
        print 'Error: not enough files to evaluate'
        return 
    elif len(args.systemFiles) == 2:
        k = 1
    else :
        k = len(args.systemFiles)
    
    
    sysPred_y_given_x = []
    sysPreds = []
    sysSol = []
    lexiconOfLabel = []
    
    
    a_result = []
    b_result = []
    
    a_dictLabel = {}
    a_lexiLabel = []
    a_preds = []
    b_dictLabel = {}
    b_lexiLabel = []
    b_preds = []
    lexiPos = []
    
    if args.systemFiles:
        print 'Loading system files ...' 
        for i in args.systemFiles:
            print i
        print '\n'
        
        for files in args.systemFiles:
                f = open(files, "rb")
                sysPYX, sysP, sysS, lexiL = pickle.load(f)
                f.close()
                if len(sysPreds) == 0:
                    sysSol = np.array(sysS)
                    a_dictLabel = lexiL.getLexiconDict()
                    a_lexiLabel = lexiL.getAllLexicon()
                    a_result = np.array(sysPYX)
                    a_preds = np.array(sysP)
                    sysPreds = sysP
                                        
                else :
                    
                    if len(np.where(np.array(sysS) != sysSol)[0]) > 0 :
                        print 'Error: different target file'
                        return
                        
                    b_dictLabel = lexiL.getLexiconDict()
                    b_lexiLabel = lexiL.getAllLexicon()
                    b_preds = np.array(sysP)
                     
                    
                    if a_dictLabel != b_dictLabel:
                        
                        a_result = np.transpose(a_result)
                        m = len(a_dictLabel)
                        for i in range(m):
                            idx = b_dictLabel.pop(a_lexiLabel[i], None)
                            
                            if idx == None:
                                if len(b_result) == 0:
                                    b_result = np.zeros(len(sysS))
                                else:
                                    b_result = np.vstack((b_result,np.zeros(len(sysS))))
                            else:
                                b_lexiLabel.remove(a_lexiLabel[i])
                                if len(b_result) == 0:
                                    b_result = sysPYX[:,idx]
                                elif idx >= len(sysPYX[0]):
                                    b_result = np.vstack((b_result,np.zeros(len(sysS))))
                                else:
                                    b_result = np.vstack((b_result,sysPYX[:,idx]))
                        for i in range(len(b_dictLabel)):
                            idx = b_dictLabel[b_lexiLabel[i]]                
                            b_result = np.vstack((b_result,sysPYX[:,idx]))
                            
                            a_dictLabel[b_lexiLabel[i]] = len(a_dictLabel)
                            a_lexiLabel.append(b_lexiLabel[i]) 
                            a_result = np.vstack((a_result,np.zeros(len(sysS))))
                        
                        a_result = np.transpose(a_result)
                        b_result = np.transpose(b_result)    
                                            
                    else:
                        b_result = np.array(sysPYX)
                    
                        
                         
                    #sysPred_y_given_x = [sysPred_y_given_x ,sysPYX]
                    sysPreds = [sysPreds, sysP]
                    #sysSol = [sysSol, sysS]
                    
                    if  args.sampling == 'divergent':
                        print 'Apenas as predicoes divergentes'
                        
                        idx = np.where(a_preds!=b_preds)[0]
                        
                        a_result = a_result[idx]
                        b_result = b_result[idx]
                        
                        sysSol = sysSol[idx] 
                    else:
                        print 'Todas as predicoes'
                    
    
    num = 0
    if args.samplingSize <=1.0:
        num = int(args.samplingSize*len(a_result))
    elif args.samplingSize < len(a_result): 
        num = args.samplingSize
    else :
        num = len(a_result)
        
    if num == 0:
        print 'Nothing to be compared'
        return
    Q = np.zeros(len(sysS))
    random_sample = np.random.random_sample((num,))
    
    print 'A: ', args.systemFiles[0]
    print 'B: ', args.systemFiles[1]
    
    
    if args.evaluation == 'euclidian' or args.evaluation =='all':
        
        Q = euclidian(a_result, b_result)
        Q = norm(Q)
        
        print '\n\nUtilidade Total Euclidiana'
        
        print 'A: ', totalUtility(a_result,Q,sysSol)
        print 'B: ', totalUtility(b_result,Q,sysSol)
        print 'Diferenca das Utilidades (UA-UB): ', compTotalUtility(a_result,b_result,Q,sysSol)
        
        
        print '\nUtilidade Amostrada Euclidiana'
        print 'A: ', utility(a_result,Q,sysSol,num,random_sample)
        print 'B: ', utility(b_result,Q,sysSol,num,random_sample)
        print 'Diferenca das Utilidades Amostradas (UA-UB): ', compUtility(a_result,b_result,Q,sysSol,num,random_sample)    
        
      
    if args.evaluation == 'kl_divergence' or args.evaluation =='all':
        Q = kl_divergence(a_result, b_result)
        Q = norm(Q)
        
        print '\n\nUtilidade Total KL_divergencia'
        print 'A: ', totalUtility(a_result,Q,sysSol)
        print 'B: ', totalUtility(b_result,Q,sysSol)
        print 'Diferenca das Utilidades (UA-UB): ', compTotalUtility(a_result,b_result,Q,sysSol)
        
        print '\nUtilidade Amostrada KL_divergencia'
        print 'A: ', utility(a_result,Q,sysSol,num,random_sample)
        print 'B: ', utility(b_result,Q,sysSol,num,random_sample)
        print 'Diferenca das Utilidades Amostradas (UA-UB):', compUtility(a_result,b_result,Q,sysSol,num,random_sample)
                         
                
                
    
if __name__ == '__main__':
    main()
    