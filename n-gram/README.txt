n-gram language model

The purpose of this model is to predict the next words over an input sequence provided by the user

Python files:
utils.py >> helper file to get the methods called from execute.py
execute.py >> main file that will call a specific method from utils.py

For this model, the dataset Cornell Movie-Dialog Corpus has been downloaded and used. More than 300K dialog utterances extracted from movie scripts will be used to train the model. The downloaded file has been cleaned up to just keep the dialogs themselves by removing any other information provided in the file. It can be found under dataset/movie_lines.txt.

The file with dialogs is parsed and split by utterances, which are then pre-processed to generate a list of tokens per utterance. For the sake of keeping training and testing sets, 99% of the data will be used for training and the remaining 1% can be used for testing, if required. This n-gram model is basically a set of Python dictionaries with probability distributions for the next word given a previous sequence. Depending on the N used for the model, it will first count the number of words (replacing them by <UNK> if its count is less than the threshold) which are then converted into probabilities. The input sequence can be modified in line 41 of file execute.py.

When generating text (as of line 46 of execute.py), the word with the highest probability distribution will be chosen and will become part of the previous sequence of words to continue until the next word chosen is the end of sentence token <e>, or in case the last 3 words are the same... which would genuinely mean that the generated text has got into an infinite loop.

This model is one of the most basics that can be developed for language models, since it fully relies on probabilities. It doesn't take into account the semantics, but it's still a good way to get started with language models and see how they have evolved.

