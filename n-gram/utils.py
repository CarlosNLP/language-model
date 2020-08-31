import math
import random
import numpy as np
import pandas as pd
import nltk

def split_to_sentences(data):
    """
    Split data by line break "\n"
    
    Args:
        data: str
    
    Returns:
        A list of sentences
    """
    # Splitting data by line break
    sentences = data.split('\n')
    
    sentences = [s.strip() for s in sentences] # removing leading and trailing spaces from each sentence
    sentences = [s for s in sentences if len(s) > 0] # dropping sentences if they are empty strings
    
    return sentences 


def tokenize_sentences(sentences):
    """
    Tokenize sentences into tokens (words)
    
    Args:
        sentences: List of strings
    
    Returns:
        List of lists of tokens
    """
    
    # Initializing the list of lists of tokenized sentences
    tokenized_sentences = []
    
    # Going through each sentence
    for sentence in sentences:
        sentence = sentence.lower() # converting to lowercase letters
        tokenized = nltk.word_tokenize(sentence) # converting into a list of words
        tokenized_sentences.append(tokenized) # appending the list of words to the list of lists
    
    return tokenized_sentences


def get_tokenized_data(data):
    """
    Make a list of tokenized sentences
    
    Args:
        data: String
    
    Returns:
        List of lists of tokens
    """
    
    # Getting the sentences by splitting up the data
    sentences = split_to_sentences(data)
    
    # Getting the list of lists of tokens by tokenizing the sentences
    tokenized_sentences = tokenize_sentences(sentences)
    
    return tokenized_sentences


def count_words(tokenized_sentences):
    """
    Count the number of word appearence in the tokenized sentences
    
    Args:
        tokenized_sentences: List of lists of strings
    
    Returns:
        dict that maps word (str) to the frequency (int)
    """
        
    word_counts = {}
    
    # Looping through each sentence
    for sentence in tokenized_sentences:
        # Going through each token in the sentence
        for token in sentence:
            # If the token is not in the dictionary yet, setting the count to 1
            if token not in word_counts:
                word_counts[token] = 1
            
            # If the token is already in the dictionary, incrementing the count by 1
            else:
                word_counts[token] += 1
    
    return word_counts


def get_words_with_nplus_frequency(tokenized_sentences, count_threshold):
    """
    Find the words that appear N times or more
    
    Args:
        tokenized_sentences: List of lists of sentences
        count_threshold: minimum number of occurrences for a word to be in the closed vocabulary.
    
    Returns:
        List of words that appear N times or more
    """
    # Initializing an empty list to contain the words that appear at least 'count_threshold' times.
    closed_vocab = []
    
    # Getting the word counts of the tokenized sentences
    word_counts = count_words(tokenized_sentences)

    # Looping through word-count key-value pairs
    for word, cnt in word_counts.items():
        if cnt >= count_threshold: # checking that the count is equal or more than the threshold
            closed_vocab.append(word) # appending the word to the list
    
    return closed_vocab


def replace_oov_words_by_unk(tokenized_sentences, vocabulary, unknown_token="<unk>"):
    """
    Replace words not in the given vocabulary with '<unk>' token.
    
    Args:
        tokenized_sentences: List of lists of strings
        vocabulary: List of strings that we will use
        unknown_token: A string representing unknown (out-of-vocabulary) words
    
    Returns:
        List of lists of strings, with words not in the vocabulary replaced
    """
    
    # Place vocabulary into a set for faster search
    vocabulary = set(vocabulary)
    
    # Initializing a list that will hold the sentences after less frequent words are replaced by the unknown token
    replaced_tokenized_sentences = []
    
    # Going through each sentence
    for sentence in tokenized_sentences:
        
        # Initializing the list that will contain a single sentence with "unknown_token" replacements
        replaced_sentence = []

        # Going through each token in the sentence
        for token in sentence:
            if token in vocabulary: # checking if the token is in the closed vocabulary
                replaced_sentence.append(token) # appending the word to the replaced sentence
            else:
                replaced_sentence.append(unknown_token) # otherwise, appending the unknown token
        
        # Appending the list of tokens to the list of lists
        replaced_tokenized_sentences.append(replaced_sentence)
        
    return replaced_tokenized_sentences


def preprocess_data(train_data, test_data, count_threshold):
    """
    Preprocess data, i.e.,
        - Find tokens that appear at least N times in the training data.
        - Replace tokens that appear less than N times by "<unk>" both for training and test data.        
    Args:
        train_data, test_data: List of lists of strings.
        count_threshold: Words whose count is less than this are 
                      treated as unknown.
    
    Returns:
        Tuple of
        - training data with low frequent words replaced by "<unk>"
        - test data with low frequent words replaced by "<unk>"
        - vocabulary of words that appear n times or more in the training data
    """

    # Getting the closed vocabulary using the training data
    vocabulary = get_words_with_nplus_frequency(train_data, count_threshold)
    
    # For the training data, replacing less common words with "<unk>"
    train_data_replaced = replace_oov_words_by_unk(train_data, vocabulary, unknown_token="<unk>")
    
    # For the testing data, replacing less common words with "<unk>"
    test_data_replaced = replace_oov_words_by_unk(test_data, vocabulary, unknown_token="<unk>")
    
    return train_data_replaced, test_data_replaced, vocabulary


def count_n_grams(data, n, start_token='<s>', end_token = '<e>'):
    """
    Count all n-grams in the data
    
    Args:
        data: List of lists of tokens
        n: number of tokens in a sequence for the n-gram model
    
    Returns:
        A dictionary that maps a tuple of n-words to its frequency
    """
    
    # Initializing dictionary of n-grams and their counts
    n_grams = {}
    
    # Going through each sentence in the data
    for sentence in data:
        
        # Prepending start token n times, and  appending <e> one time (kind of padding for this model)
        sentence = [start_token] * n + sentence + [end_token]
        
        # Convert list to tuple so that the sequence of words can be used as a key in the dictionary
        sentence = tuple(sentence)
        
        for i in range(len(sentence) - n + 1): # looping through tokens to get the n-grams

            # Getting the n-gram from i to i+n
            n_gram = sentence[i:i+n]

            # Checking if the n-gram is in the dictionary
            if n_gram in n_grams:
                n_grams[n_gram] += 1 # incrementing the count for this n-gram
            else:
                n_grams[n_gram] = 1 # initializing this n-gram count to 1
    
    return n_grams


def estimate_probability(word, previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=1.0):
    """
    Estimate the probabilities of a next word using the n-gram counts with k-smoothing
    
    Args:
        word: next word
        previous_n_gram: A sequence of words of length n
        n_gram_counts: Dictionary of counts of n-grams
        n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
        vocabulary_size: number of words in the vocabulary
        k: positive constant, smoothing parameter
    
    Returns:
        A probability
    """
    # Converting list to tuple to use it as a dictionary key
    previous_n_gram = tuple(previous_n_gram)
    
    # If the previous n-gram exists in the dictionary of n-gram counts, we get its count; otherwise, we set the count to zero
    previous_n_gram_count = n_gram_counts.get(previous_n_gram, 0)
        
    # Calculating the denominator using the count of the previous n gram and applying k-smoothing
    denominator = previous_n_gram_count + k * vocabulary_size

    # Defining n plus 1 gram as the previous n-gram plus the current word as a tuple
    n_plus1_gram = previous_n_gram + (word,)
  
    # Setting the count to the count in the dictionary; otherwise, 0 if not in the dictionary
    n_plus1_gram_count = n_plus1_gram_counts.get(n_plus1_gram, 0)
        
    # Defining the numerator using the count of the n-gram plus current word and applying smoothing
    numerator = n_plus1_gram_count + k

    # Calculating the probability as the numerator divided by denominator
    probability = numerator / denominator
    
    return probability


def estimate_probabilities(previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary, k=1.0):
    """
    Estimate the probabilities of next words using the n-gram counts with k-smoothing
    
    Args:
        previous_n_gram: A sequence of words of length n
        n_gram_counts: Dictionary of counts of n-grams
        n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
        vocabulary: List of words
        k: positive constant, smoothing parameter
    
    Returns:
        A dictionary mapping from next words to the probability.
    """
    
    # Converting list to tuple to use it as a dictionary key
    previous_n_gram = tuple(previous_n_gram)
    
    # Adding <e> <unk> to the vocabulary
    vocabulary = vocabulary + ["<e>", "<unk>"]
    vocabulary_size = len(vocabulary)
    
    probabilities = {} # initializing the dictionary where we will store the word with each probability
    for word in vocabulary:
        probability = estimate_probability(word, previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=k) # estimating probability of word
        probabilities[word] = probability # the value of the word (key) will be the probability itself

    return probabilities


def suggest_a_word(previous_tokens, n_gram_counts, n_plus1_gram_counts, vocabulary, k=1.0, start_with=None):
    """
    Get suggestion for the next word
    
    Args:
        previous_tokens: The sentence you input where each token is a word. Must have length > n 
        n_gram_counts: Dictionary of counts of n-grams
        n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
        vocabulary: List of words
        k: positive constant, smoothing parameter
        start_with: If not None, specifies the first few letters of the next word
        
    Returns:
        A tuple of 
          - string of the most likely next word
          - corresponding probability
    """
    
    # Length of previous words
    n = len(list(n_gram_counts.keys())[0]) 
    
    # From the words that the user already typed, getting the most recent 'n' words as the previous n-gram
    previous_n_gram = previous_tokens[-n:]

    # Estimating the probabilities that each word in the vocabulary is the next word
    probabilities = estimate_probabilities(previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary, k=k)
    
    # Initializing suggested word to None; this will be set to the word with highest probability
    suggestion = None
    
    # Initializing the highest word probability to 0; this will be set to the highest probabilityof all words to be suggested
    max_prob = 0
    
    # Looping through every word and probability
    for word, prob in probabilities.items():
        
        # If the optional start_with string is set
        if start_with != None:
            if word.startswith(start_with) is False: # checking if the beginning of the word does not match with the letters in 'start_with'
                continue # if they don't match, skip this word and move onto the next word
        
        if prob > max_prob: # checking if this word's probability is greater than the current maximum probability
            suggestion = word # the suggestion becomes the word itself
            max_prob = prob # saving the new maximum probability
    
    return suggestion, max_prob


def get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0, start_with=None):
    """
    Get suggestions from the input sequence

    Args:
        previous_tokens: The sentence you input where each token is a word. Must have length > n 
        n_gram_counts_list: List of dictionaries with n-gram counts
        vocabulary: List of words
        k: positive constant, smoothing parameter
        start_with: If not None, specifies the first few letters of the next word

    Returns:
        A list of suggestions in the form of a tuple for each entry (suggestion, probability)
    """
    model_counts = len(n_gram_counts_list)
    suggestions = []
    for i in range(model_counts-1):
        n_gram_counts = n_gram_counts_list[i] # getting n-gram counts of each model
        n_plus1_gram_counts = n_gram_counts_list[i+1] # getting n-gram plus 1 counts of each model
        
        suggestion = suggest_a_word(previous_tokens, n_gram_counts, n_plus1_gram_counts, vocabulary, k=k, start_with=start_with) # getting suggestion
        suggestions.append(suggestion) # appending to suggestions
    return suggestions
