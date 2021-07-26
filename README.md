# CTC Decoding Algorithms

**Update 2021: installable Python package**

Python implementation of some common **Connectionist Temporal Classification (CTC) decoding algorithms**. 
A minimalistic **language model** is provided.

## Installation

* Go to the root level of the repository
* Execute `pip install .`
* Go to `tests/` and execute `pytest` to check if installation worked


## Usage

### Basic usage

Here is a minimalistic executable example:

````python
import numpy as np
from ctc_decoder import best_path, beam_search

mat = np.array([[0.4, 0, 0.6], [0.4, 0, 0.6]])
chars = 'ab'

print(f'Best path: "{best_path(mat, chars)}"')
print(f'Beam search: "{beam_search(mat, chars)}"')
````

The output `mat` (numpy array, softmax already applied) of the CTC-trained neural network is expected to have shape TxC 
and is passed as the first argument to the decoders.
T is the number of time-steps, and C the number of characters (the CTC-blank is the last element).
The characters that can be predicted by the neural network are passed as the `chars` string to the decoder.
Decoders return the decoded string.  
Running the code outputs:

````
Best path: ""
Beam search: "a"
````

To see more examples on how to use the decoders, 
please have a look at the scripts in the `tests/` folder.



### Language model and BK-tree

Beam search can optionally integrate a character-level language model.
Text statistics (bigrams) are used by beam search to improve reading accuracy.

````python
from ctc_decoder import beam_search, LanguageModel

# create language model instance from a (large) text
lm = LanguageModel('this is some text', chars)

# and use it in the beam search decoder
res = beam_search(mat, chars, lm=lm)
````

The lexicon search decoder computes a first approximation with best path decoding.
Then, it uses a BK-tree to retrieve similar words, scores them and finally returns the best scoring word.
The BK-tree is created by providing a list of dictionary words.
A tolerance parameter defines the maximum edit distance from the query word to the returned dictionary words.

````python
from ctc_decoder import lexicon_search, BKTree

# create BK-tree from a list of words
bk_tree = BKTree(['words', 'from', 'a', 'dictionary'])

# and use the tree in the lexicon search
res = lexicon_search(mat, chars, bk_tree, tolerance=2)
````

### Usage with deep learning frameworks
Some notes:
* No adapter for TensorFlow or PyTorch is provided
* Apply softmax already in the model
* Convert to numpy array
* Usually, the output of an RNN layer `rnn_output` has shape TxBxC, with B the batch dimension 
  * Decoders work on single batch elements of shape TxC
  * Therefore, iterate over all batch elements and apply the decoder to each of them separately
  * Example: extract matrix of batch element 0 `mat = rnn_output[:, 0, :]`
* The CTC-blank is expected to be the last element along the character dimension
  * TensorFlow has the CTC-blank as last element, so nothing to do here
  * PyTorch, however, has the CTC-blank as first element by default, so you have to move it to the end, or change the default setting 

## List of provided decoders

Recommended decoders:
* `best_path`: best path (or greedy) decoder, the fastest of all algorithms, however, other decoders often perform better
* `beam_search`: beam search decoder, optionally integrates a character-level language model, can be tuned via the beam width parameter
* `lexicon_search`: lexicon search decoder, returns the best scoring word from a dictionary

Other decoders, from my experience not really suited for practical purposes, 
but might be used for experiments or research:
* `prefix_search`: prefix search decoder
* `token_passing`: token passing algorithm
* Best path decoder implementation in OpenCL (see `extras/` folder)

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
