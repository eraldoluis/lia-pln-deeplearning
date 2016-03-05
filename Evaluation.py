import argparse
import cPickle as pickle
import numpy




def main():
    
    parser = argparse.ArgumentParser();
    
    evaluationStrategyChoices = ["euclidian", "kl_divergence"]

    parser.add_argument('--evaluation', dest='evaluation', action='store', default=evaluationStrategyChoices[0], choices=evaluationStrategyChoices,
                       help='Set the EVALUATION strategy. EUCLIDIAN and KL_DIVERGENCE are the options available')
    
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
    
    if args.systemFiles:
        print 'Loading system files from ' + ' ...'
        
        for files in args.systemFiles:
                f = open(files, "rb")
                sysPYX, sysP, sysS = pickle.load(f)
                f.close()
                if len(sysPreds) == 0:
                    sysPred_y_given_x = sysPYX 
                    sysPreds = sysP
                    sysSol = sysS
                else :
                    sysPred_y_given_x = [sysPred_y_given_x ,sysPYX]
                    sysPreds = [sysPreds, sysP]
                    
                    i = 0
                    for sol in sysS:
                        if (sol != sysSol[i]):
                            print 'Error Solution'
                
    print  sysPred_y_given_x
    print sysSol
    
    
                
    
if __name__ == '__main__':
    main()
    