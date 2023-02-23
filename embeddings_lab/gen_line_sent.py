import codecs
from conll_iterator import ConllIterator, WAC
from tqdm import tqdm

DEFAULT_PARAMS = {
    'fields': ['form'],
    'lower': ['form'],
    'join_char': '/'
    }

class stopwords:
    def __init__(self, file):
        with codecs.open(file, 'r', 'utf8') as filein:
            self.stopwords = set(l.strip() for l in filein.readlines())
    
    def remove_stopwords(self, sentences):
        for s in sentences:
            s = list(filter(lambda x: x not in self.stopwords, s))
            yield s

def gen_line_sent(file_out, corpus, stopwords_file='', **iterator_parameters):
    for k, v in DEFAULT_PARAMS.items():
        iterator_parameters.setdefault(k, v)
    iterator_parameters['mode'] = 'sentences'
    sentences = ConllIterator(corpus, **iterator_parameters)
    tot = sentences.sentences
    if stopwords_file:
        sw = stopwords(stopwords_file)
        sentences = sw.remove_stopwords(sentences)
    with codecs.open(file_out + '.tot', "w", "utf8") as fileout:
        fileout.write(str(tot))
    with codecs.open(file_out, "w", "utf8") as fileout:
        for sent in tqdm(sentences, total=tot):
            fileout.write(" ".join(sent) + "\n")
