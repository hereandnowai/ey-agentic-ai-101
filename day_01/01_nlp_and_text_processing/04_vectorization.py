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

# part 1 - one hot encoding, built by hand
print("\n 1. One hot encoding, built by hand, no libraries")

vocabulary = ["loan", "card", "customer", "applied", "for", "fraud", "refund", "branch"]

print("Vocabulary (columns):", vocabulary)
print("\n {'WORD':<10} {'VECTOR (list of numbers)':<20}")
print("-" * 40)

for word in vocabulary:
    vector = [1 if v == word else 0 for v in vocabulary]
    print(f" {word:<10} {vector}")

# part 2 - bag of words
complaints = [
    "my loan payment failed again"
    "the loan payment was not processed"
    "my card was blocked by the bank"
    "fraud on my card please block it"
]

print("\n 2. Bag of words, using sklearn's CountVectorizer")
for i, c in enumerate(complaints):
    print(f"Complaint {i} : {c}")

vectorizer = CountVectorizer()
matrix = vectorizer.fit_transform(complaints)
vocab = vectorizer.get_feature_names_out()

print("\nVocabulary (columns):", vocab)
print(" ", list(vocab))
print(" total columns:", len(vocab))

print("\n the table of numbers (each row in one complaint, each column is a word in the vocabulary):")
counts = csr_matrix(matrix).toarray()

header = "   row.  " + " ".join(f"{w[:6]:>7}" for w in vocab)
print(header)
for i, row in enumerate(counts):
    line = f"   {i:<5}  " + " ".join(f"{n:>7}" for n in row)
    print(line)

print("\n Every complaint is now a row of", len(vocab), "numbers, one for each word in the vocabulary.")