CONTENTS OF THIS FILE
---------------------

*   Introduction
*   Setup
*   Getting started

INTRODUCTION
------------

A collection of utilities to work with word embeddings.

SETUP
-----
```
pip install git+https://github.com/amatusNLP/nlp_embeddings_lab
```

GETTING STARTED
---------------
Generate ALaCarte matrix (Please refer to: https://github.com/NLPrinceton/ALaCarte)
```
from embeddings_lab import alacarte_utils

DUMPROOT = "sample/alacarte/matrix"
LINESENTENCE = "sample/sample_linesentence.txt"
VECTORS = "sample/w2v_model/w2v_model.w2v"
WINDOW = 5

alacarte_utils.gen_alacarte_matrix(DUMPROOT, LINESENTENCE, VECTORS, WINDOW)
```

Induce ALaCarte vectors (Please refer to: https://github.com/NLPrinceton/ALaCarte)
```
from embeddings_lab import alacarte_utils

DUMPROOT = "sample/alacarte/vectors"
TARGETS = "sample/alacarte/targets.txt"
MATRIX = "sample/alacarte/matrix_transform.bin"
LINESENTENCE = "sample/sample_linesentence.txt"
VECTORS = "sample/w2v_model/w2v_model.w2v"
WINDOW = 5

alacarte_utils.gen_vecs(DUMPROOT, TARGETS, MATRIX, LINESENTENCE, VECTORS,
                         WINDOW)

alacarte_utils.join_multiwords(w2v_file=DUMPROOT + '_alacarte.txt', vector_size=250)
```

Generate dataset to train chars2vec
```
from embeddings_lab import morphological


TSV = "sample/morphology/related.tsv"
JSON = "sample/morphology/related.json"
DATASET = "sample/morphology/dataset.pkl"


morphological.relsets_from_tsv(TSV, JSON)
morphological.gen_dataset(JSON, DATASET)
```

Train chars2vec (Please refer to: https://github.com/IntuitionEngineeringTeam/chars2vec)
```
import chars2vec
from embeddings_lab import morphological


VECTOR_SIZE = 50
MODEL = "sample/morphology/c2v"
DATASET = "sample/morphology/dataset.pkl"


model = MODEL + '_' + str(VECTOR_SIZE)
X_train, y_train = morphological.load_dataset(DATASET)
model_chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
               'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
               'à', 'è', 'é', 'ì', 'ò', 'ù']
c2v_model = chars2vec.train_model(VECTOR_SIZE, X_train, y_train, model_chars)
chars2vec.save_model(c2v_model, model)
```

Generate morphological embeddings with chars2vec (Please refer to: https://github.com/IntuitionEngineeringTeam/chars2vec)
```
from embeddings_lab import morphological


MODEL = "sample/morphology/c2v_50"
WORDLIST = "sample/morphology/targets.txt"
EMBEDDINGS = "sample/morphology/morphological_vectors.w2v"


morphological.gen_embeddings(MODEL, WORDLIST, EMBEDDINGS)
```