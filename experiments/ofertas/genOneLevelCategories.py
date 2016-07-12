'''
Created on 12/07/2016

@author: eraldo
'''
from codecs import open
import sys
from sets import Set
import os

def flatTree(tree, node):
    '''
    Compress the given forest so that every node points to the root of its tree.
    '''
    parent = tree[node]
    if (parent not in tree) or (parent == node):
        # Found a root node. Return it.
        return parent

    # Keep looking for the root.
    parent = flatTree(tree, parent)
    tree[node] = parent
    return parent

def docsFile2Dir(fileName):
    f = open(fileName, 'r', 'utf8')
    f.readline()
    # Dictionary of parents for all nodes: 'tree[n]' is the parent of node 'n'.
    tree = {}
    numExs = 0
    for l in f:
        ftrs = [s.strip() for s in l.split('\t')]
        tree[int(ftrs[1])] = int(ftrs[0])
        numExs += 1
        if numExs % 100000 == 0:
            sys.stdout.write('.')
            sys.stdout.flush()
    
    sys.stdout.write('\n')

    print '# examples:', numExs
    print '# nodes:', len(tree)

    # Flat tree to two levels (root is always 1).
    roots = Set()
    for n in tree.iterkeys():
        flatTree(tree, n)
        roots.add(tree[n])

    print '# roots:', len(roots)
    print 'Roots:', roots

    # Dir of     
    baseDir = os.path.dirname(fileName)

    f.close()


if __name__ == '__main__':
    docsFile2Dir(sys.argv[1])
