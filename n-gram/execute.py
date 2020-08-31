import nltk
from utils import *

### CLEANING AND PRE-PROCESSING THE DATA ###

# Loading the data
with open("dataset/movie_lines.txt", "r", encoding="utf-8") as f:
    data = f.read()

# Tokenizing the data and retrieving a list of sentences where each entry is a again a list of tokens
print('Tokenizing data...\n')
tokenized_data = get_tokenized_data(data)

# Specifying the size of the data: 99% for training and 1% for testing
train_size = int(len(tokenized_data) * 0.99)
train_data = tokenized_data[0:train_size]
test_data = tokenized_data[train_size:]

print("Data size:", len(tokenized_data))
print("Training set:", len(train_data))
print("Testing set:", len(test_data), '\n')

# Replacing word under the minimum frequency with <unk> tokens for training and testing sets
minimum_freq = 2
train_set, test_set, vocabulary = preprocess_data(train_data, test_data, minimum_freq)


### DEVELOPING N-GRAM BASED LANGUAGE MODEL ###

# Implementing n-gram model
n_gram_counts_list = []
for n in range(3, 6):
    print("Computing n-gram counts with n =", n, "...")
    n_model_counts = count_n_grams(train_set, n)
    n_gram_counts_list.append(n_model_counts)

### TESTING THE LANGUAGE MODEL ###

# Entering and cleaning input sentence
sentence = "It will take a little"
sentence = sentence.lower()
previous_tokens = nltk.word_tokenize(sentence)
print('\nStart of the sentence:', previous_tokens)

# Generating text
while True:
    suggestions = get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0) # getting the suggestions as a list
    suggestions_dict = {} # initializing the suggestions dictionary
    for suggestion in suggestions: # looping through the suggestions and populating the suggestions dictionary
        word, prob = suggestion[0], suggestion[1] # since every suggestion (within the suggestions list) is made of a tuple (word, probability)
        if word not in suggestions_dict:
            suggestions_dict[word] = prob # key: word; value: probability
        else:
            if prob > suggestions_dict[word]: # overwriting existing probability in case the word already exists and the new option has more probability
                suggestions_dict[word] = prob

    word = max(suggestions_dict, key=suggestions_dict.get) # easily retrieving the key (word) with the highest value (probability)
    if word == "<e>" or len(set(previous_tokens[-3:])) == 1: # exiting the loop if the next word is the end of sentence token or probable infinite loop (3 same tokens)
        break
    else:
        previous_tokens.append(word) # appending the next word to the previous tokens for the next loop

print(previous_tokens)
