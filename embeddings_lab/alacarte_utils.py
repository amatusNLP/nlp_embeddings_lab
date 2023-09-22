import os
import codecs
from pathlib import Path
from typing import List


ALACARTE = os.path.dirname(os.path.realpath(__file__)) + \
    '/ALaCarte/alacarte.py'


def gen_alacarte_matrix(dumproot: str, linesentence: str, source_vectors: str,
                        window=5):
    for file in (linesentence, source_vectors):
        if not Path(file).is_file():
            raise FileNotFoundError(file)
    kwargs = {
        'alacarte': ALACARTE,
        'vectors': source_vectors,
        'corpus': linesentence,
        'window': window,
        'dumproot': dumproot
    }
    comand = 'python "{alacarte}" "{dumproot}" -s "{vectors}" -c "{corpus}"' \
        ' -w {window}'
    os.system(comand.format_map(kwargs))
    

def gen_vecs(dumproot: str, targets: str, matrix: str, linesentence: str,
             source_vectors: str, window=5):
    for file in (targets, matrix, linesentence, source_vectors):
        if not Path(file).is_file():
            raise FileNotFoundError(file)
    kwargs = {
        'alacarte': ALACARTE,
        'matrix': matrix,
        'vectors': source_vectors,
        'corpus': linesentence,
        'window': window,
        'targets': targets,
        'dumproot': dumproot
    }
    comand = 'python "{alacarte}" -v -m "{matrix}" -s "{vectors}" -w "{window}"' \
         ' -c "{corpus}" -t "{targets}" "{dumproot}" --create-new'
    os.system(comand.format_map(kwargs))


def list2targets(wordlist: List[str], txt_file: str):
    with codecs.open(txt_file, 'w', 'utf8') as fileout:
        fileout.write('\n'.join(wordlist))


def join_multiwords(w2v_file: str, vector_size: int, join_char: str='_',
                    fileout: str=None):
    lines = list()
    if not fileout:
        fileout = w2v_file
    with codecs.open(w2v_file, 'r', 'utf8') as filein:
        for line in filein:
            line = line.split(' ')
            word, vector = line[:-vector_size], line[-vector_size:]
            line = "{word} {vector}" .format(word=join_char.join(word),
                                             vector =' '.join(vector))
            lines.append(line)
    with codecs.open(fileout, 'w', 'utf8') as fileout:
        fileout.writelines(lines)
