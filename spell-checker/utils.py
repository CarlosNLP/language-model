import sqlite3
import re
import xml.etree.ElementTree as ET
import string
import numpy as np
from nltk.tokenize import word_tokenize
from collections import Counter

# Method to parse a TMX file and dump its contents into a database
def tmx_to_sql(db_path, tmx_path):
    # Opening and connecting to database
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Creating table if not exists
    cur.execute("""CREATE TABLE IF NOT EXISTS "translations" (
                "source_text"    TEXT,
                "target_text"    TEXT,
                "source_lang"   TEXT,
                "target_lang"   TEXT
                );""")
    
    # Running some replacements in TMX file
    try:
        with open(tmx_path, 'rt', encoding="utf-8-sig") as f:
            content = f.read()
            print('Reading', tmx_path)
            content = content.replace("xml:lang", "lang") # modifying lang attribute so the parser captures it
            content = re.sub('<bpt[^>]*?>', '', content) # removing bpt tags from TMX/XLIFF
            content = re.sub('<ept[^>]*?>', '', content) # removing ept tags from TMX/XLIFF
        with open(tmx_path, 'wt', encoding="utf-8-sig") as f:
            f.write(content)
    except: # trying reading with UTF-16 encoding and writing as UTF-8 BOM so the ET parser recognizes as XML
        try:
            with open(tmx_path, 'rt', encoding='utf16') as f:
                content = f.read()
                print('Reading', tmx_path)
                content = content.replace("xml:lang", "lang") # modifying lang attribute so the parser captures it
                content = re.sub('<bpt[^>]*?>', '', content) # removing bpt tags from TMX/XLIFF
                content = re.sub('<ept[^>]*?>', '', content) # removing ept tags from TMX/XLIFF
                content = content.replace("UTF-16LE", "utf-8") # avoiding encoding issue when parsing TMX
            with open(tmx_path, 'wt', encoding="utf-8-sig") as f:
                f.write(content)
        except: # showing file with error
            print('Error reading file', tmx_path)
    
    # Parsing TMX file
    doc = ET.parse(tmx_path)
    root = doc.getroot()
    tu_list = root.findall("./body/tu")
    
    # Looping through every translation unit
    for tu in tu_list:
        # Getting the translation unit pairs
        tuv = tu.findall("tuv")

        for i in range(1, len(tuv)):
            # Retrieving specific values for our database
            source_text = tuv[0].find("seg").text
            target_text = tuv[i].find("seg").text
            source_lang = tuv[0].get("lang").lower()
            target_lang = tuv[i].get("lang").lower()
            
            # Inserting entries into our database
            cur.execute("INSERT INTO translations VALUES (?, ?, ?, ?)", (source_text, target_text, source_lang, target_lang))
    
    # Committing the changes into the database
    conn.commit()
    
    # Closing database connection
    cur.close()
    conn.close()


# Method to clean or pre-process the sentence before its use
def process_sentence(sentence):
    '''
    Input:
        sentence: a string containing the retrieved sentence
    Output:
        sentence_clean: a list of words containing the processed sentence
    '''
    # Removing punctuation from the sentence
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    sentence = re.sub('[0-9]', '', sentence) # removing numbers
    
    # Tokenizing sentence
    sentence_tokens = word_tokenize(sentence)
    
    sentence_clean = []
    for word in sentence_tokens:
        sentence_clean.append(word.lower()) # lowering word
    
    return sentence_clean


# Method to build the frequency dictionary
def build_freqs(words_all):
    '''
    Input:
        words_all: a list of words representing the corpus. 
    Output:
        freqs: The wordcount dictionary where key is the word and value is its frequency.
    '''
    
    freqs = {}
    for word in words_all:
        if word in freqs.keys(): # incrementing the count if the word already exists in the dictionary
            freqs[word] += 1
        else: # setting to 1 since the word doesn't exist yet in the dictionary
            freqs[word] = 1
    
    return freqs


# Method to build the probabilities dictionary
def get_probs(freqs, num_words):
    '''
    Input:
        freqs: The wordcount dictionary where key is the word and value is its frequency
        num_words: The number of total words from the words_all list (no duplicates)
    Output:
        probs: A dictionary where keys are the words and the values are the probability that a word will occur
    '''
    probs = {}
    
    for word, freq in freqs.items():
        probs[word] = freq / num_words # getting the probability of a word appearing in the whole corpus (frequency of a word divided by total number of words)
    
    return probs


# Method to find every single occurrence of output words when a character is deleted
def delete_letter(word):
    '''
    Input:
        word: input word
    Output:
        delete_l: a list of all possible words obtained by deleting 1 character from word
    '''

    # Using list comprehensions
    split_l = [(word[:i], word[i:]) for i in range(len(word) + 1)] # splitting a word into all possible tuples with two elements (left and right)
    delete_l = [L + R[1:] for L, R in split_l if len(R) > 0] # building word by removing the first character in the right part of every tuple

    return delete_l


def switch_letter(word):
    '''
    Input:
        word: input string
     Output:
        switches: a list of all possible strings with one adjacent charater switched
    ''' 

    # Using list comprehensions
    split_l = [(word[:i], word[i:]) for i in range(len(word) + 1)] # splitting a word into all possible tuples with two elements (left and right)
    switch_l = [L + R[1] + R[0] + R[2:] for L, R in split_l if len(R) > 1] # building word by switching first and second characters from the right part of every tuple

    return switch_l


def replace_letter(word):
    '''
    Input:
        word: input word 
    Output:
        replaces: a list of all possible strings where we replaced one letter from the original word
    ''' 
    
    letters = 'abcdefghijklmnopqrstuvwxyz'
    replace_l = []
    
    split_l = [(word[:i], word[i:]) for i in range(len(word) + 1)] # splitting a word into all possible tuples with two elements (left and right)
    for L, R in split_l:
        if len(R) > 0:
            for letter in letters:
                replace_l.append(L+R[0].replace(R[0], letter) + R[1:]) # building word by replacing the first character from the right part of the tuple with any letter
    replace_set = set(replace_l)
    replace_set.discard(word)
    replace_l = list(replace_set)
    
    return replace_l


def insert_letter(word):
    '''
    Input:
        word: input word
    Output:
        inserts: a set of all possible strings with one new letter inserted at every position
    ''' 
    letters = 'abcdefghijklmnopqrstuvwxyz'
    insert_l = []
    
    split_l = [(word[:i], word[i:]) for i in range(len(word) + 1)] # splitting a word into all possible tuples with two elements (left and right)
    for L, R in split_l:
        for letter in letters:
            insert_l.append(L + letter + R[0:]) # building word by inserting a letter before the first character from the right part of the tuple
    
    return insert_l

def edit_one_letter(word):
    """
    Input:
        word: the input word for which we will generate all possible words that are one edit away
    Output:
        edit_one_set: a set of words with one possible edit
    """
    
    edit_one_list = list()

    # Appending every possible word that can be formed by deleting, switching, replacing or inserting a letter from the word
    edit_one_list += delete_letter(word)
    edit_one_list += switch_letter(word)
    edit_one_list += replace_letter(word)
    edit_one_list += insert_letter(word)
    
    edit_one_set = set(edit_one_list)

    return edit_one_set


def edit_two_letters(word):
    '''
    Input:
        word: the input word for which we will generate all possible words that are two edits away
    Output:
        edit_two_set: a set of words with all possible two edits
    '''
    
    edit_one_list = list()
    edit_two_list = list()

    # Appending every possible word that can be formed by deleting, switching, replacing or inserting a letter from the word
    edit_one_list += delete_letter(word)
    edit_one_list += switch_letter(word)
    edit_one_list += replace_letter(word)
    edit_one_list += insert_letter(word)

    # Doing the same as above (being two edit distance now) from every possible word formed previously
    for edit_one_word in edit_one_list:
        edit_two_list += delete_letter(edit_one_word)
        edit_two_list += switch_letter(edit_one_word)
        edit_two_list += replace_letter(edit_one_word)
        edit_two_list += insert_letter(edit_one_word)
    
    edit_two_set = set(edit_two_list)
    
    return edit_two_set


def get_corrections(word, probs, vocab, n):
    '''
    Input: 
        word: a user entered word to check for suggestions
        probs: a dictionary that maps each word to its probability in the corpus
        vocab: a list containing all the vocabulary
        n: number of possible word corrections you want returned in the dictionary
    Output: 
        n_best: a list of tuples with the most probable n corrected words and their probabilities
    '''
    
    suggestions = [] # list where all the possible suggestions will be appended
    suggestions_dict = {} # dictionary where only the suggested words appering in the vocabulary will be added, along with its probability in the corpus
    n_best = [] # list of tuples where the specific number of final suggestions (n) will be returned

    # Getting suggestions at one edit distance
    suggestions = list(edit_one_letter(word))

    # Looping through every suggestion in the vocabulary and building the suggestions dictionary with their probabilities
    for suggestion in suggestions:
        if suggestion in vocab:
            prob = probs[suggestion]
            suggestions_dict[suggestion] = prob

    # If we don't get any suggestions at one edit distance, we'll look for suggestions at two edit distance
    if len(suggestions_dict) == 0:
        further_suggestions = list(edit_two_letters(word))
        for further_suggestion in further_suggestions:
            if further_suggestion in vocab:
                prob = probs[further_suggestion]
                suggestions_dict[suggestion] = prob

    # Sorting the dictionary with the helper function Counter to get the most common words
    c = Counter(suggestions_dict)
    n_best = c.most_common(n) # list of tuples (word, probability) sorted by most common
    
    return n_best
