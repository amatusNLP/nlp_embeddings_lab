from gensim.models.word2vec import LineSentence, Word2Vec
from tqdm import tqdm
import json

def train_w2v(outroot, linesentence, **w2v_parameters):
    with open(linesentence + '.tot', 'r') as filein:
        tot = int(filein.read())
    sentences = LineSentence(linesentence)
    model = Word2Vec(tqdm(sentences, total=tot), **w2v_parameters)
    model.save(outroot + ".model")
    model.wv.save(outroot + ".wv")
    model.wv.save_word2vec_format(outroot + ".w2v", write_header = False)
    with open(outroot + '_parameters.json', 'w') as fileout:
        json.dump(w2v_parameters, fileout)

