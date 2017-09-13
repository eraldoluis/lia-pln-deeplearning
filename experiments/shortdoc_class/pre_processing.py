    #!/usr/bin/env python
    # -*- coding: utf-8 -*-

    """
    Created on 26/05/2017
    
    @author: eraldo
    
    A partir de um arquivo contendo tweets (um em cada linha e com possíveis atributos separados por TAB),
    aplica os passos abaixo e salva em outro arquivo.
        - Tokenização.
        - Converte tudo para minúsculo.
        - Substitui dígitos por 0.
        - Substitui tokens iniciados em 'http:' e 'www.' por ##LINK##.
        - Substitui hashtags por ##HASHTAG##.
        - Substitui @xxx por ##REF##.
        - Substitui repetições de sinais de pontuação por um único sinal.
        - Substitui sequências de pontos por ...
        - Substitui sequências de exclamações por !!!
    """
    import re
    import sys
    from codecs import open

    from tokenizer import getTokenizer

    if __name__ == "__main__":
        if len(sys.argv) != 3:
            print "Syntax error!"
            print "\tArguments: <input file> <output file>"
            print "Both input and output files can be -, which means standard input or output, respectively."
            sys.exit(1)

        inFilename = sys.argv[1]

        inFile = sys.stdin
        if inFilename != "-":
            inFile = open(inFilename, "rt", "utf-8")

        outFilename = sys.argv[2]
        outFile = sys.stdout
        if outFilename != "-":
            outFile = open(outFilename, "wt", "utf-8")

        tokenizer = getTokenizer()

        # Skip header line.
        inFile.readline()
        numExs = 0
        print("Reading input examples...")
        # Recognize digits.
        patDig = re.compile('[0-9]')
        # Recognize sequence of punctuations.
        patPunc = re.compile('([!.,?])+')
        for l in inFile:
            # Each line can be composed by several features.
            ftrs = l.split('\t')

            # Clean possible \n in the end of the line.
            ftrs[-1] = ftrs[-1].strip()

            # Tokenize the text.
            tokens = tokenizer.tokenize(ftrs[0])

            # Apply filters.
            procTokens = []
            for token in tokens:
                if token.startswith('http:') or token.startswith('https:') or token.startswith('www.'):
                    token = "##LINK##"
                elif token.startswith("#"):
                    token = "##HASHTAG##"
                elif token.startswith("@"):
                    token = "##REF##"
                elif token.startswith(".."):
                    token = "..."
                elif token.startswith("!!"):
                    token = "!!!"
                else:
                    token = re.sub(patDig, '0', token)
                    token = token.lower()
                    token = re.sub(patPunc, r'\1', token)

                procTokens.append(token)

            ftrs[0] = " ".join(procTokens)
            outFile.write("\t".join(ftrs) + "\n")

            numExs += 1

        inFile.close()
        outFile.close()

        sys.stderr.write('\n')
        sys.stderr.write('# examples: %d\n' % numExs)
        sys.stderr.write('Done!\n')
