#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on 12/07/2016

@author: eraldo

A partir de um dataset de ofertas, converte a estrutura hierárquica de 
categorias em uma estrutura flat. A hierarquia de categoria é uma floresta.
Este script substitui a categoria de cada exemplo pela categoria raiz da sua
árvore.

O formato de entrada é o seguinte. Cada linha contém um exemplo (a primeira
linha é o cabeçalho). Cada exemplo segue o seguinte formato:

<id_pai> [TAB] <id> [TAB] <desc_norm> [TAB] <categ_shop_desc_nor> [TAB] <price>

onde, <id_pai> é o ID da categoria pai, <id> é o ID da categoria da oferta,
<desc_norm> é o texto da oferta, <categ_shop_desc_nor> é categoria interna do
anunciante, e <price> é o preço do produto.

A saída segue o mesmo formato da entrada.

"""
import sys
from codecs import open

# Input file includes a header?
hasHeader = False

def flatTree(tree, node):
    """
    Compress the given forest so that every node points to the root of its tree.
    """
    parent = tree[node]
    if (parent not in tree) or (parent == node):
        # Found a root node. Return it.
        return parent

    # Keep looking for the root.
    parent = flatTree(tree, parent)
    tree[node] = parent
    return parent

def docsFile2Dir(fileName):
    # Open input file.
    f = open(fileName, 'r', 'utf8')

    # Skip header.
    if hasHeader:
        f.readline()

    # Dictionary of parents for all nodes: 'tree[n]' is the parent of node 'n'.
    tree = {}

    sys.stdout.write('Reading input examples to build the category tree')
    sys.stdout.flush()
    numExs = 0
    for l in f:
        ftrs = [s.strip() for s in l.split('\t')]
        tree[int(ftrs[1])] = int(ftrs[0])
        numExs += 1
        if numExs % 100000 == 0:
            sys.stdout.write('.')
            sys.stdout.flush()
    sys.stdout.write('\n')

    f.close()

    print '# examples:', numExs
    print '# nodes:', len(tree)

    print 'Done!'

    print 'Making trees flat...'
    # Flat tree to two levels (root is always 1).
    roots = set()
    for n in tree.iterkeys():
        flatTree(tree, n)
        roots.add(tree[n])

    print '# roots:', len(roots)
    print 'Roots:', roots
    print 'Done!'
    
    # Open input file again.
    f = open(fileName, 'r', 'utf8')

    # Read header.
    if hasHeader:
        header = f.readline()
    
    # Open output file.
    fout = open(fileName + '.flat', 'w', 'utf8')

    # Write header.
    if hasHeader:
        fout.write(header)
    
    sys.stdout.write('Reading input examples and writing to the output file using the flatten categories')
    sys.stdout.flush()
    numExs = 0
    for l in f:
        ftrs = [s.strip() for s in l.split('\t')]
        ftrs[1] = ftrs[0] = str(tree[int(ftrs[1])])
        fout.write('\t'.join(ftrs) + '\n')
        numExs += 1
        if numExs % 100000 == 0:
            sys.stdout.write('.')
            sys.stdout.flush()
    sys.stdout.write('\n')

    f.close()
    fout.close()

    print '# written examples:', numExs
    print 'Done!'


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.stderr.write('Erro de sintaxe. Sintaxe esperada:\n')
        sys.stderr.write('\t<dataset de entrada>\n')
        sys.exit(1)

    docsFile2Dir(sys.argv[1])
