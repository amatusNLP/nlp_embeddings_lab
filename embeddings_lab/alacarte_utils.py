import os
import codecs

ALACARTE = os.path.dirname(os.path.realpath(__file__)) + \
    '/ALaCarte/alacarte.py'

def gen_alacarte_matrix(dumproot, linesentence, source_vectors, window=5):
    kwargs = {
        'alacarte': ALACARTE,
        'vectors': source_vectors,
        'corpus': linesentence,
        'window': window,
        'dumproot': dumproot
    }
    comand = 'python {alacarte} {dumproot} -s {vectors} -c {corpus}' \
        ' -w {window}'.format_map(kwargs)
    os.system(comand.format_map(kwargs))
    

def gen_vecs(dumproot, targets, matrix, linesentence, source_vectors,
             window=5):
    kwargs = {
        'alacarte': ALACARTE,
        'matrix': matrix,
        'vectors': source_vectors,
        'corpus': linesentence,
        'window': window,
        'targets': targets,
        'dumproot': dumproot
    }
    comand = 'python {alacarte} -v -m {matrix} -s {vectors} -w {window}' \
         ' -c {corpus} -t {targets} {dumproot} --create-new'
    os.system(comand.format_map(kwargs))

def list2targets(wordlist, txt_file):
    with codecs.open(txt_file, 'w', 'utf8') as fileout:
        fileout.write('\n'.join(wordlist))