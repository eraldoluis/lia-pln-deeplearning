# NLP Deep Learning Framework using Theano

Designed by: Eraldo R. Fernandes

This reposity includes a Deep Learning Framework for Natural Language Processing problems using Theano.

## Features

* Automatic metric logging
* Training loop with callbacks (using observer design pattern)
* Early stopping
* Two SGD-based algorithms: vanilla SGD and AdaGrad
* Layer stacking with automatic parameter update


## Available Instantiations

All available instantiations can be found in the package [`experiments`](https://github.com/eraldoluis/lia-pln-deeplearning/tree/master/experiments).

There are instantiations for the following NLP problems:
* [Document Classification](https://github.com/eraldoluis/lia-pln-deeplearning/blob/master/experiments/doc_classification/wnn.py)
* [Named Entity Recognition](https://github.com/eraldoluis/lia-pln-deeplearning/blob/master/experiments/ner/wnn.py)
* [Part-of-Speech Tagging](https://github.com/eraldoluis/lia-pln-deeplearning/blob/master/experiments/postag/wnn.py)
