# Data files
The data files for the **Word example** are located in `data/word/` and the files for the **Line example** in `data/line`.
Each of these directories contains:
* `rnnOutput.csv`: output of RNN layer (softmax not yet applied), which contains 32 or 100 time-steps and 80 label scores per time-step.
* `corpus.txt`: the text from which the language model is generated.
* `img.png`: the input image of the neural network. It is contained as an illustration, however, the decoding algorithms do not use it.

