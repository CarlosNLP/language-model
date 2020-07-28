import os
from utils import *
import pickle

# Opening and loading the vocabulary list
with open("pickled/vocab.pickle", "rb") as f:
    vocab = pickle.load(f)

# Opening and loading the probability dictionary
with open("pickled/probs.pickle", "rb") as f:
    probs = pickle.load(f)

print('Unique words in corpus:', len(vocab))

# Testing the spell checker
word = "automatoin" # incorrectly spelled word
print('Word:', word)

if word not in vocab:
    corrections = get_corrections(word, probs, vocab, 3)
    print('Corrections:', [correction[0] for correction in corrections])
else:
    print('The word is spelled correctly.')


