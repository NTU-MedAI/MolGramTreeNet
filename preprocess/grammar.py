# -*- coding: UTF-8 -*-

import nltk  # NLP toolikt
import numpy
import six

# Grammar production rules

gram = """smiles -> chain
atom -> bracket_atom
atom -> aliphatic_organic
atom -> aromatic_organic
aliphatic_organic -> 'B'
aliphatic_organic -> 'C'
aliphatic_organic -> 'N'
aliphatic_organic -> 'O'
aliphatic_organic -> 'S'
aliphatic_organic -> 'P'
aliphatic_organic -> 'F'
aliphatic_organic -> 'I'
aliphatic_organic -> 'Cl'
aliphatic_organic -> 'Br'
aromatic_organic -> 'c'
aromatic_organic -> 'n'
aromatic_organic -> 'o'
aromatic_organic -> 's'
bracket_atom -> '[' BAI ']'
BAI -> isotope symbol BAC
BAI -> symbol BAC
BAI -> isotope symbol
BAI -> symbol
BAC -> chiral BAH
BAC -> BAH
BAC -> chiral
BAH -> hcount BACH
BAH -> BACH
BAH -> hcount
BACH -> charge class
BACH -> charge
BACH -> class
symbol -> aliphatic_organic
symbol -> aromatic_organic
isotope -> DIGIT
isotope -> DIGIT DIGIT
isotope -> DIGIT DIGIT DIGIT
DIGIT -> '1'
DIGIT -> '2'
DIGIT -> '3'
DIGIT -> '4'
DIGIT -> '5'
DIGIT -> '6'
DIGIT -> '7'
DIGIT -> '8'
chiral -> '@'
chiral -> '@@'
hcount -> 'H'
hcount -> 'H' DIGIT
charge -> '-'
charge -> '-' DIGIT
charge -> '-' DIGIT DIGIT
charge -> '+'
charge -> '+' DIGIT
charge -> '+' DIGIT DIGIT
bond -> '-'
bond -> '='
bond -> '#'
bond -> '/'
bond -> '\\'
bond -> '.'
ringbond -> DIGIT
ringbond -> bond DIGIT
branched_atom -> atom
branched_atom -> atom RB
branched_atom -> atom BB
branched_atom -> atom RB BB
RB -> RB ringbond
RB -> ringbond
BB -> BB branch
BB -> branch
branch -> '(' chain ')'
branch -> '(' bond chain ')'
chain -> branched_atom
chain -> chain branched_atom
chain -> chain bond branched_atom
symbol -> element_symbols
aromatic_organic -> 'p'
element_symbols -> 'H' 
class -> DIGIT
Nothing -> None"""
# atom -> '[UNK]'
# bond -> '[UNK]'
# bond -> '.'

GCFG = nltk.CFG.fromstring(gram)
start_index = GCFG.productions()[0].lhs()
all_lhs = [a.lhs().symbol() for a in GCFG.productions()]
lhs_list = []
for a in all_lhs:
    if a not in lhs_list:
        lhs_list.append(a)
D = len(GCFG.productions())

rhs_map = [None] * D
count = 0
for a in GCFG.productions():
    rhs_map[count] = []
    for b in a.rhs():
        if not isinstance(b, six.string_types):
            s = b.symbol()
            rhs_map[count].extend(list(numpy.where(numpy.array(lhs_list) == s)[0]))
    count = count + 1

masks = numpy.zeros((len(lhs_list), D))
count = 0
for sym in lhs_list:
    is_in = numpy.array([a == sym for a in all_lhs], dtype=int).reshape(1, -1)
    masks[count] = is_in
    count = count + 1

index_array = []
for i in range(masks.shape[1]):
    index_array.append(numpy.where(masks[:, i] == 1)[0][0])
ind_of_ind = numpy.array(index_array)
max_rhs = max([len(l) for l in rhs_map])