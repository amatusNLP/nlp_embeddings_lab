import os
from itertools import chain
from collections import defaultdict
import json
from tqdm import tqdm
import torch
from gensim.models import KeyedVectors
from transformers import BertTokenizerFast, BertModel
from conll_iterator import ConllIterator, Extractor

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PATTERN = "CAND: {<ADJ>|<ADV>|<INTJ>|<NOUN>|<PROPN>|<VERB>|<ADP>|<AUX>|"\
    "<CCONJ>|<DET>|<PART>|<PRON>|<SCONJ>}"

class BertEmbeddings:

    def __init__(self, model='bert-base-multilingual-uncased', word_count=None) -> None:
        self.tokenizer = BertTokenizerFast.from_pretrained(model)
        self.model = BertModel.from_pretrained(model,
                                  output_hidden_states = True,
                                  ).to(DEVICE)
        self.model.eval()
        self.embeddings_dict = dict()
        self.word_count = word_count
        self.parameters = {"model": model}

    def sent2embeddings(self, tokenized_text: list):
        input_dic = self.tokenizer.encode_plus(
                            tokenized_text,
                            truncation=True,
                            is_split_into_words=True,
                            return_tensors="pt"
                            )
        for k in input_dic:
            input_dic[k] = input_dic[k].to(DEVICE)
        with torch.no_grad():
            outputs = self.model(**input_dic)
            hidden_states = outputs[2]
        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = torch.flatten(token_embeddings, start_dim=1,
                                         end_dim=2)
        token_embeddings = token_embeddings.permute(1,0,2)
        sum_vec = [torch.sum(token[-4:], dim=0) for token in token_embeddings]
        return sum_vec, input_dic.word_ids()[1:-1]


    def to_keyed_vectors(self):
        kv = KeyedVectors(768)
        keys, weights = zip(*self.embeddings_dict.items())
        weights = [w.cpu().numpy() for w in weights]
        kv.add_vectors(keys, weights)
        return kv

    def gen_vocab(self, conll_file: str, minfreq=10, lemmatized=True,
              lower=True, verbose=True):
        if lemmatized:
            iterator_fields = ["lemma", "upos"]
        else:
            iterator_fields = ["form", "upos"]
        if lower:
            iterator_lower = iterator_fields[0]
        else:
            iterator_lower = list()  
        sentences = ConllIterator(conll_file, fields=iterator_fields,
                                  lower=iterator_lower)
        if verbose:
            sentences = tqdm(sentences, total=sentences.sentences)
        extr = Extractor(PATTERN)
        self.word_count = {w: f for ((w, _), ), f
                           in extr.count(sentences).most_common() 
                           if f >=minfreq}
        self.embeddings_dict = {w: torch.zeros(768).to(DEVICE) 
                                for w in self.word_count}
        
        
    def gen_embeddings(self, conll_file: str, minfreq=10, lemmatized=True,
              lower=True, verbose=True):
        if not self.word_count:
            self.gen_vocab(conll_file, minfreq, lemmatized, lower, verbose)
        self.parameters = {
            "corpus": conll_file,
            "min frequency": minfreq,
            "lemmatized": lemmatized,
            "lowercase": lower,
            "n vectors": len(self.word_count) 
        }
        if lemmatized:
            iterator_fields = ["form", "lemma"]
        else:
            iterator_fields = ["form", "form"]    
        if lower:
            iterator_lower = iterator_fields
        else:
            iterator_lower = list()  
        sentences = ConllIterator(conll_file, fields=iterator_fields,
                                  lower=iterator_lower)
        if verbose:
            sentences = tqdm(sentences, total=sentences.sentences)
        for sent in sentences:
            sent, words = zip(*sent)
            vecs, word_ids = self.sent2embeddings(sent)
            s_vecs = defaultdict(list)
            for i, v in zip(word_ids, vecs):
                v1 = self.embeddings_dict.get(words[i], None) 
                if v1 is not None:
                     s_vecs[words[i]].append(v1)
            for word, vectors in s_vecs.items():
                    self.embeddings_dict[word] = v + torch.mean(torch.stack(vectors))
        for w, f in self.word_count.items():
            self.embeddings_dict[w] /= f
    
    def save_model(self, dumproot: str):
            kv = self.to_keyed_vectors()
            kv.save(dumproot + ".wv")
            kv.save_word2vec_format(dumproot + ".w2v", write_header = False)
            with open(dumproot + '_parameters.json', 'w') as fileout:
                json.dump(self.parameters, fileout)
