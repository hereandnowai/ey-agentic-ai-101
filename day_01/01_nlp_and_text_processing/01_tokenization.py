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