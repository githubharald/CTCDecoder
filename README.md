# CTC Decoding Algorithms
Python implementation of some common **Connectionist Temporal Classification (CTC) decoding algorithms**. 
A minimalistic **language model** is provided.

## Installation
* Go to the root level of the repository
* Execute `pip install .`
* Go to `tests/` and execute `pytest` to see if everything works


## Usage

### Basic usage

Here is a first example on how to use the decoders:

````python
import numpy as np
from ctc_decoder import best_path, beam_search

mat = np.array([[0.4, 0, 0.6], [0.4, 0, 0.6]])
labels = 'ab'

print(f'Best path: "{best_path(mat, labels)}"')
print(f'Beam search: "{beam_search(mat, labels)}"')
````

The output `mat` of the CTC-trained neural network is expected to have shape TxC 
and is passed as the first argument to the decoders.
T is the number of time-steps, and C the number of characters (including CTC-blank).
The labels predicted by the neural network are passed as the `labels` string to the decoder. 
Decoders return the decoded string.  
This should output:

````
Best path: ""
Beam search: "a"
````

To see more examples on how to use the decoders, 
please have a look at the scripts in the `tests/` folder.


### Language model and BK-tree

Beam search and token passing use a language model.
A text corpus and the labels than can be predicted by the neural network are passed to its constructor:

````python
from ctc_decoder import LanguageModel

lm = LanguageModel('aaab', 'ab')
print(f'Bigram "aa": {lm.get_char_bigram("a", "a")}')
print(f'Bigram "ab": {lm.get_char_bigram("a", "b")}')
````

The output shows the bigram probabilities for two character pairs:
````
Bigram "aa": 0.6666666666666666
Bigram "ab": 0.3333333333333333
````

Lexicon search needs a BK-tree instance to get a list of similar dictionary words, given a query word.
The following sample shows how to create an instance of that tree and how to search for similar words:

````python
from ctc_decoder import BKTree

bk_tree = BKTree(['bad', 'bag', 'ball'])
query = 'bar'
print(f"Words similar to '{query}': {bk_tree.query(query, tolerance=1)}")
````

It outputs:

````
Words similar to 'bar': ['bad', 'bag']
````

## List of provided decoders

Recommended decoders:
* `best_path`: best path or greedy decoder, the fastest decoder, however, it might give worse results than the other decoders
* `beam_search`: beam search decoder, optionally integrating a character-level language model, can be tuned via the beam width parameter
* `lexicon_search`: lexicon search decoder, returns the best scoring word from a dictionary

Other decoders, from my experience not really suited for practical purposes, 
but might be used for experiments or research:
* `token_passing`: token passing algorithm
* `prefix_search`: prefix search decoder
* Best path decoder implementation using OpenCL (see `extras/` folder)

[This paper](./doc/comparison.pdf) gives suggestions when to use best path decoding, beam search decoding and token passing.


## Documentation of test cases and data

* Documentation of [test cases](./tests/README.md)
* Documentation of the [data](./data/README.md)


## References

* [Graves - Supervised sequence labelling with recurrent neural networks](https://www.cs.toronto.edu/~graves/preprint.pdf)
* [Hwang - Character-level incremental speech recognition with recurrent neural networks](https://arxiv.org/pdf/1601.06581.pdf)
* [Shi - An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](https://arxiv.org/pdf/1507.05717.pdf)
* [Marti - The IAM-database: an English sentence database for offline handwriting recognition](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)
* [Beam Search Decoding in CTC-trained Neural Networks](https://towardsdatascience.com/5a889a3d85a7)
* [An Intuitive Explanation of Connectionist Temporal Classification](https://towardsdatascience.com/3797e43a86c)
* [Scheidl - Comparison of Connectionist Temporal Classification Decoding Algorithms](./doc/comparison.pdf)
* [Scheidl - Word Beam Search: A Connectionist Temporal Classification Decoding Algorithm](https://repositum.tuwien.ac.at/obvutwoa/download/pdf/2774578)
