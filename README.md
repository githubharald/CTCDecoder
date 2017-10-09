# CTC Decoding Algorithms with Language Model

## Algorithms
- Best Path Decoding: takes best labels per time-step and then removes repeated labels and blanks. Function: ctcBestPath. [1]
- Beam Search Decoding: iteratively searches for best labelling, uses a character bigram language model. Function: ctcBeamSearch. [2]
- Token Passing: searches for most probable word sequence, words are from a dictionary. Can be extended to also use a word-bigram language model. Function: ctcTokenPassing. [1]

## Run
`python ctcDecoder.py`

Expected results:
`
=====Mini example=====
BEST PATH  : ""
BEAM SEARCH: "a"
=====Real example=====
BEST PATH  : "the fak friend of the fomly hae tC"
BEAM SEARCH: "the fake friend of the family, lie th"
TOKEN      : "the fake friend of the family fake the"
`


[1] Graves - Supervised sequence labelling with recurrent neural networks
[2] Hwang - Character-level incremental speech recognition with recurrent neural networks