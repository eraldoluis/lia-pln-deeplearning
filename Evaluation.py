import argparse
import cPickle as pickle
import numpy



def  




def main():
    
    parser = argparse.ArgumentParser();
    
    parser.add_argument('--solutionFile', dest='solutionFile', action='store',
                       help='Training File Path', required=True)
    
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
    
    
    systems = []
    solution = []
    
    if args.systemFiles:
        print 'Loading system files from ' + args.systemFiles + ' ...'
        
        for files in args.systemFiles:
                f = open(files, "rb")
                sys = pickle.load(f)
        
    
    
    if args.systemFiles:
        print 'Loading system files from ' + args.systemFiles + ' ...'
        
        for files in args.systemFiles:
                f = open(files, "rb")
                sys = pickle.load(f)
                f.close()
                if len(systems) == 0:
                    systems = sys
                else:
                    systems = [systems, sys]
    
    
                

                
    
if __name__ == '__main__':
    main()
    