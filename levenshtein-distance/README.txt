Levenshtein distance

The purpose of this model is to calculate the Levenshtein distance (minimum edit distance) between two strings.

Python files:
utils.py >> helper file to get the methods called from execute.py
execute.py >> main file that will call a specific method from utils.py

For this model, no datasets have been downloaded. With dynamic programming we can divide a problem into subsets of problems and reuse what has been calculated previously for other subsets to increase the efficiency and reduce computing times.

The costs for insertion, deletion and replacement are 1, 1 and 2, respectively. It's 2 for replacement because that is the same as deleting a character and then inserting a new one. The maximum edit distance between two strings is the replacement cost applied to every character from the string with less characters "min(len(source), len(target))" plus the absolute value of the length difference between them (times the insertion cost, which is 1, so we can ignore that). With this in mind, we can calculate not only the edit distance but also the percentage of distance or similarity between the strings.

This model could be plugged into any kind of string extractor to calculate the Levenshtein distance between a sentence coming from machine translation and the corresponding post-edited version, for example.