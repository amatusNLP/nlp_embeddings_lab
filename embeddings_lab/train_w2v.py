from gensim.models.word2vec import LineSentence, Word2Vec
import json

def train_w2v(dumproot, linesentence, **w2v_parameters):
    sentences = LineSentence(linesentence)
    model = Word2Vec(sentences, **w2v_parameters)
    model.save(dumproot + ".model")
    model.wv.save(dumproot + ".wv")
    model.wv.save_word2vec_format(dumproot + ".w2v", write_header = False)
    with open(dumproot + '_parameters.json', 'w') as fileout:
        json.dump(w2v_parameters, fileout)
        

