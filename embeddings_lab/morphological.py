import codecs
import json
from collections import defaultdict
from itertools import combinations, chain
from sklearn.utils import shuffle
import pickle
import numpy as np
import random
import chars2vec

def relsets_from_tsv(tsv_file, json_file):
    with codecs.open(tsv_file, 'r', 'utf8') as filein:
        lines = filein.readlines()
        lines = [l.strip().split('\t') for l in lines]
    deriv_list = defaultdict(set)
    for v, k in lines:
        deriv_list[k].add(v)
    new_data = {i:list(v.union({k})) for i, (k, v) in 
                enumerate(deriv_list.items())}
    with codecs.open(json_file, "w", "utf8") as fileout:
        json.dump(new_data, fileout, indent=1)

def gen_dataset(relset_file, pickle_file, negtive_multiplier=1):
    with codecs.open(relset_file, "r", "utf8") as filein:
        relset = json.load(filein)
    relset = list(relset.values())
    X_positive = list(chain.from_iterable([combinations(s, 2) 
                                           for s in relset]))
    n = len((X_positive))
    y_positive = np.zeros(n)
    n = n*negtive_multiplier
    X_negative = list()
    for _ in range(n):
        i1, i2 = random.sample(range(len(relset)), 2)
        s1, s2 = list(relset[i1]), list(relset[i2])
        w1, w2 = random.choice(s1), random.choice(s2)
        X_negative.append((w1, w2))
    n = len(X_negative)
    y_negative = np.ones(n)
    X_train = X_positive + X_negative
    y_train = np.concatenate((y_positive, y_negative))
    X_train, y_train = shuffle(X_train, y_train)
    data = {"X":X_train, "y":y_train}
    with open(pickle_file, "wb") as fileout:
        pickle.dump(data, fileout)

def load_dataset(pickle_file):
    with open(pickle_file, "rb") as filein:
        data = pickle.load(filein)
    return data["X"], data["y"]

def gen_embeddings(path_to_model, wordlist, w2v_file):
    c2v_model = chars2vec.load_model(path_to_model)
    if type(wordlist) != list:
        with codecs.open(wordlist, "r", "utf8") as filein:
            wordlist = filein.read()
        wordlist = wordlist.split("\n")
    word_embeddings = c2v_model.vectorize_words(wordlist)
    with codecs.open(w2v_file, "w", "utf8") as fileout:
        for i, w in enumerate(wordlist):
            line = w + " " + " ".join(str(v) for v in word_embeddings[i]) + "\n"
            fileout.write(line)