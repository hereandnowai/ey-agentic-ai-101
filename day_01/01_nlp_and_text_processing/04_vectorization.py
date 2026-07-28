# what is vectorization?

# loan
# the customer applied for a loan

# a vector is a mathematical representation of data in the form of an array or list of numbers.
# In the context of natural language processing (NLP) and text processing,
# vectorization refers to the process of converting text data into numerical vectors that
# can be used as input for machine learning algorithms.

# one hot - one column per word; put a 1 in the word's column if it appears in the document, otherwise put a 0
# bag of words - one column per word; put a count of the word's occurrences in the document in the word's column

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from scipy.sparse import csr_matrix
