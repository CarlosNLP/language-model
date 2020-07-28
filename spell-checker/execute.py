import numpy as np
import os
from utils import *
import pickle

# Defining the path to the DB and TMX folder
db_path = "dataset/translations.db"
tmx_folder = "dataset"

# Set to True if you would like to convert the TMX to the database, or False otherwise
convert_tmx_to_sql = False

if convert_tmx_to_sql == True:
    print('Converting TMX to SQL: True')
    
    # Creating a list with all the TMX files
    tmx_list = []
    for root, dirs, files in os.walk(tmx_folder):
        for name in files:
            if name.endswith('.tmx'):
                tmx_list.append(os.path.join(root, name))
    
    # Dumping TMX data into SQL database
    print('Number of TMX files:', len(tmx_list))
    for tmx_path in tmx_list:
        tmx_to_sql(db_path, tmx_path)

else: # printing for reference
    print('Converting TMX to SQL: False')

# Path and opening database
conn = sqlite3.connect(db_path)
cur = conn.cursor()

# Initializing English sentences lists
english_sentences = []

# Getting a list of English sentences from the SQL database (in this case, all the source segments are in English)
cur.execute("SELECT source_text FROM translations")
data = cur.fetchall()

# Updating the English sentences list
print('\nRetrieving English sentences...')
for entry in data:
    english_sentences.append(entry[0])

# Closing database connection
cur.close()
conn.close()

# Keeping just the unique values from the list
english_sentences = list(set(english_sentences))

print('English sentences in database:', len(english_sentences))

# Processing every single segment and retrieve the cleaned words
words_all = []
print('\nRetrieving words from corpus...')
for sentence in english_sentences:
    words_all += process_sentence(sentence)

vocab = sorted(list(set(words_all))) # keeping just unique words and ordered alphabetically
print('Total words in corpus:', len(words_all))
print('Unique words in corpus:', len(vocab))

### Saving vocabulary as a pickle
##with open("pickled/vocab.pickle", "wb") as f:
##    pickle.dump(vocab, f)

# Building frequency dictionary where the key is a word from the corpus and the value is its frequency
print('\nBuilding frequency dictionary...')
freqs = build_freqs(words_all)

# Building probability dictionary where the key is a word from the corpus and the value is its probability of occurring
print('Building probability dictionary...')
probs = get_probs(freqs, len(words_all))

### Saving the probability dictionary as a pickle
##with open("pickled/probs.pickle", "wb") as f:
##    pickle.dump(probs, f)

# Testing the spell checker
word = "automatoin" # incorrectly spelled word
print('Word:', word)

if word not in vocab:
    corrections = get_corrections(word, probs, vocab, 3)
    print('Corrections:', [correction[0] for correction in corrections])
else:
    print('The word is spelled correctly.')


