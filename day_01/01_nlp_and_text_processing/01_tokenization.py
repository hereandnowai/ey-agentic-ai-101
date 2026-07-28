# What is tokenization in NLP?
# Tokenization is the process of breaking down text into smaller units called tokens. In Natural Language Processing (NLP), these tokens can be words, phrases, or even characters,
# depending on the level of granularity required for analysis.
# Tokenization is a crucial step in text preprocessing, as it allows for easier manipulation and analysis of textual data.
# By converting text into tokens, we can perform various NLP tasks such as sentiment analysis,
# part-of-speech tagging, and machine learning model training more effectively.


# the customer paid the emi on the third of may

# don't  - do n't
# 3.5% - one piece 


# naive way
# nltk
# spaCy

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

nlp = spacy.load("en_core_web_sm")

sentence = "Mandy didn't pay the EMI of Rs.45,000 on 3rd May; the bank charged 3.5% interest."

# Way 1: Using naive split (split by space)
print("\n 1. Navie Split (split by space):")
naive = sentence.split()
print(naive)
print("count:", len(naive))

# Way 2: Using NLTK tokenizer
print("\n 2. NLTK Tokenizer:")
nltk_tokens = word_tokenize(sentence)
print(nltk_tokens)
print("count:", len(nltk_tokens))

# Way 3: Using spaCy tokenizer
print("\n 3. spaCy Tokenizer:")
doc = nlp(sentence)
print([token.text for token in doc])
print("count:", len(doc))

print("\n piece-by-piece comparison of tokens:")
print(f" {'Token':<12} {'IS IT A WORD?':<15} {"IS IT PUNCTUATION?":<20} {'IS IT A NUMBER?':<20}")
for token in doc:
    print(f" {token.text:<12} {str(token.is_alpha):<15} {str(token.is_punct):<20} {token.like_num:<20}")
