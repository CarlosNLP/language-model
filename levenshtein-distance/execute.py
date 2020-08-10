import pandas as pd
from utils import *

# Initializing strings to be compared
source =  'I love NLP'
target = 'He loves NLP'

# Specifying cost values for insertion, deletion and replacement
ins_cost = 1
del_cost = 1
rep_cost = 2

# Getting matrix and getting minimum edit distance
matrix, min_edits = min_edit_distance(source, target, ins_cost, del_cost, rep_cost)

# Levehnstein score as a percentage
max_edit_distance = min(len(source), len(target)) * rep_cost + abs(len(source) - len(target))
lev_distance = round(min_edits / max_edit_distance * 100, 2)
lev_similarity = round(100 - lev_distance, 2)

print("Source:", source)
print("Target:", target)
print("Minimum edit distance:", min_edits)
print("Levenshtein distance (%):", lev_distance)
print("Levenshtein similarity (%):", lev_similarity, "\n")

# Building matrix visually to be printed out with pandas
idx = list('#' + source)
cols = list('#' + target)
df = pd.DataFrame(matrix, index=idx, columns= cols)
print("Matrix with edit distances:\n")
print(df)
