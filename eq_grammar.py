import nltk
import numpy as np
import six
import pdb

gram = """S -> S '+' T
S -> S '*' T
S -> S '/' T
S -> S '-' T
S -> T
S -> '-' T

T -> '(' S ')'
T -> '(' S ')^2'
T -> 'sin(' S ')'
T -> 'exp(' S ')'
T -> 'log(' S ')'
T -> 'cos(' S ')'
T -> 'sqrt(' S ')'
T -> 'tanh(' S ')'

T -> T '^' D 
T -> 'pi'
T -> 'x'
T -> 'y'
T -> 't'
T -> 'x^2'
T -> 'x^3' 
T -> 'y^2' 
T -> 'y^3' 

T -> D
T -> D '.' D

T -> '-' D
T -> '-' D '.' D



T -> T D

D -> D '0' | D '1' | D '2' | D '3' | D '4' | D '5' | D '6' | D '7' | D '8' | D '9'
D -> '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9'

D -> 'e-1' | 'e-2' | 'e-3' | 'e-4'

Nothing -> None
"""

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
            rhs_map[count].extend(list(np.where(np.array(lhs_list) == s)[0]))
    count = count + 1

masks = np.zeros((len(lhs_list), D))
count = 0
# all_lhs.append(0)
for sym in lhs_list:
    is_in = np.array([a == sym for a in all_lhs], dtype=int).reshape(1, -1)
    # pdb.set_trace()
    masks[count] = is_in
    count = count + 1

index_array = []
for i in range(masks.shape[1]):
    index_array.append(np.where(masks[:, i] == 1)[0][0])
ind_of_ind = np.array(index_array)
