# -*- coding: UTF-8 -*-

import nltk
import numpy as np
import logging as logging
from preprocess import grammar as gram
LOGGER = logging.getLogger(__name__)


def get_tokenizer(cfg):
    long_tokens = filter(lambda a: len(a) > 1, cfg._lexical_index.keys())
    long_tokens = list(long_tokens)
    print(long_tokens)
    replacements = ['$', '%', '^'] #,'&']

    assert len(list(long_tokens)) == len(replacements)
    for token in replacements:
        assert not (token in cfg._lexical_index)

    def tokenize(smiles):
        for i, token in enumerate(long_tokens):
            smiles = smiles.replace(token, replacements[i])
        tokens = []
        for token in smiles:
            try:
                ix = replacements.index(token)
                tokens.append(long_tokens[ix])
            except:
                tokens.append(token)
        return tokens

    return tokenize


class ZincGrammarModel(object):
    def __init__(self):
        self._grammar = gram
        self._productions = self._grammar.GCFG.productions()
        self._prod_map = {}
        for ix, prod in enumerate(self._productions):
            self._prod_map[prod] = ix + 1
        self._parser = nltk.ChartParser(self._grammar.GCFG)
        self._tokenize = get_tokenizer(self._grammar.GCFG)
        self._n_chars = len(self._productions)
        self._lhs_map = {}
        for ix, lhs in enumerate(self._grammar.lhs_list):
            self._lhs_map[lhs] = ix

    def encode_smiles(self, smiles):
        """Encode a single SMILES string into a production rule index sequence"""
        try:
            tokenized = self._tokenize(smiles)
            parse_tree = self._parser.parse(tokenized).__next__()
            productions = parse_tree.productions()
            indices = np.array([self._prod_map[prod] for prod in productions], dtype=int)
            indices = np.append(indices, gram.D)  # Add end token
            # indices += 1
            return indices.tolist()
        except Exception as e:
            LOGGER.error(f"Failed to encode SMILES: {smiles} - {str(e)}")
            return None

    def get_vocabulary(self):
        return self._prod_map.copy()
