# ==============================================================================
#  DAY 1 · NLP & TEXT PROCESSING · 07 — VECTORIZATION
# ==============================================================================
#
#  THE PROBLEM WE ARE SOLVING
#     Every machine learning model, without exception, does maths. Addition,
#     multiplication, averages. You cannot multiply the word "loan". You cannot
#     take the average of "the customer is unhappy".
#     So before any model can touch our text, we must convert words into NUMBERS.
#
#  WHAT IS VECTORIZATION?
#     A "vector" is just a fancy word for a LIST OF NUMBERS. That is all.
#     Vectorization is the job of turning a piece of text into such a list, so
#     that maths becomes possible.
#     Think of it as a scoresheet. We decide on a fixed set of columns (the
#     vocabulary), then for each document we fill in a number under each column.
#     Every document ends up as a row of numbers of exactly the same length.
#
#  THE TWO CLASSIC RECIPES
#     ONE-HOT       — one column per word; put a 1 in the word's own column and
#                     0 everywhere else. Represents a SINGLE word.
#     BAG OF WORDS  — one column per word; put the COUNT of how many times that
#                     word appears. Represents a WHOLE document.
#     Both are called "sparse", because the rows are mostly zeros.
#
#  WHY IS IT CALLED A "BAG"?
#     Because it throws the word order away, like tipping all the words of a
#     sentence into a bag and shaking it. "dog bites man" and "man bites dog"
#     produce the exact same numbers. That is this method's big weakness, and we
#     demonstrate it at the end of this file.
#
#  ABOUT THIS FILE
#     We build one-hot vectors by hand first, so you can see there is no magic in
#     it. Then we use scikit-learn's CountVectorizer to turn four customer
#     complaints into a proper table of numbers. We inspect that table, use it to
#     compare documents, and finally show the "bag" weakness in action.
# ==============================================================================

from sklearn.feature_extraction.text import CountVectorizer   # the ready-made tool that counts words into a table
import numpy as np                                   # a library for working with grids of numbers
from scipy.sparse import csr_matrix                 # the concrete sparse type our matrices really are


# ------------------------------------------------------------------------------
#  PART 1 — ONE-HOT ENCODING, built by hand
# ------------------------------------------------------------------------------
print("\n1) ONE-HOT ENCODING (built by hand, no library)")   # heading

vocabulary = ["loan", "card", "fraud", "refund", "branch"]    # our fixed set of columns; only these 5 words exist here

print("   our vocabulary (the columns):", vocabulary)         # show the agreed columns
print(f"\n   {'WORD':<10}{'ITS VECTOR (list of numbers)'}")    # column titles
print("   " + "-" * 45)                              # a divider line

for word in vocabulary:                              # take each word in our vocabulary
    vector = [1 if v == word else 0 for v in vocabulary]   # put 1 in this word's own column, 0 in all the others
    print(f"   {word:<10}{vector}")                  # show the word next to its list of numbers

print("\n   Notice: each list is 5 long because our vocabulary has 5 words.")   # the length always matches the vocabulary
print("   Every word gets exactly one '1'. That is why it is called ONE-hot.")  # explain the name
print("   PROBLEM: 'loan' and 'card' are equally distant from each other,")     # the flaw of this method
print("   and so are 'loan' and 'fraud'. The numbers carry NO meaning at all.")  # numbers say nothing about relatedness


# ------------------------------------------------------------------------------
#  PART 2 — BAG OF WORDS on real customer complaints
# ------------------------------------------------------------------------------
complaints = [                                       # four short, realistic customer complaints
    "my loan payment failed again",
    "the loan payment was not processed",
    "my card was blocked by the bank",
    "fraud on my card please block it",
]

print("\n2) BAG OF WORDS — counting words into a table")   # heading
for i, c in enumerate(complaints):                   # show the complaints with numbers
    print(f"   complaint {i}: {c}")                  # print each one

vectorizer = CountVectorizer()                       # create the counting tool
matrix = vectorizer.fit_transform(complaints)        # learn the vocabulary AND count the words in one go
vocab = vectorizer.get_feature_names_out()           # ask which words became the columns

print("\n   the vocabulary it learned (the columns):")   # heading
print("  ", list(vocab))                             # show every column name
print("   total columns:", len(vocab))               # how many different words there are in total

print("\n   the table of numbers (each row is one complaint):")   # heading
counts = csr_matrix(matrix).toarray()                # convert to a normal grid of numbers so we can print it

header = "   row  " + " ".join(f"{w[:6]:>7}" for w in vocab)   # build a header line from the column names
print(header)                                        # print the header
for i, row in enumerate(counts):                     # go through each complaint's row of numbers
    line = f"   {i:<5}" + " ".join(f"{n:>7}" for n in row)   # line up the numbers under their columns
    print(line)                                      # print the row

print("\n   Every complaint is now a row of", len(vocab), "numbers.")   # the key result
print("   All rows are the same length, so maths is finally possible.")  # why that matters
print("   Most of the numbers are 0 — this is what 'sparse' means.")     # explain the jargon word


# ------------------------------------------------------------------------------
#  PART 3 — Using those numbers to compare documents
# ------------------------------------------------------------------------------
print("\n3) COMPARING COMPLAINTS USING THE NUMBERS")   # heading


def cosine(v1, v2):                                  # measures how similar two lists of numbers are
    dot = np.dot(v1, v2)                             # multiply the two lists position by position, then add it all up
    size1 = np.linalg.norm(v1)                       # the "length" of the first list
    size2 = np.linalg.norm(v2)                       # the "length" of the second list
    return dot / (size1 * size2) if size1 and size2 else 0.0   # divide, giving a score from 0.0 to 1.0


print("   cosine similarity: 1.00 means identical, 0.00 means nothing in common")   # explain the scale
pairs = [(0, 1), (2, 3), (0, 2)]                     # three pairs worth comparing
for a, b in pairs:                                   # go through each pair
    score = cosine(counts[a], counts[b])             # score how similar their number rows are
    print(f"   complaint {a} vs complaint {b} = {score:.2f}")   # show the score
print("   >> 0 and 1 are both about loans, so they score higher.")   # the expected pattern
print("   >> 0 and 2 are about different topics, so they score lower.")   # the contrast
print("   This is exactly how a search engine finds relevant documents.")   # the real-world use


# ------------------------------------------------------------------------------
#  PART 4 — THE WEAKNESS: word order is thrown away
# ------------------------------------------------------------------------------
print("\n4) THE 'BAG' WEAKNESS — order is lost")     # heading

opposites = [                                        # two sentences with identical words, opposite meanings
    "the bank rejected the customer",
    "the customer rejected the bank",
]

order_vec = CountVectorizer()                        # a fresh counting tool
order_counts = csr_matrix(order_vec.fit_transform(opposites)).toarray()   # count the words in both sentences

print("   sentence A:", opposites[0])                # show the first sentence
print("   sentence B:", opposites[1])                # show the second sentence
print("   A as numbers:", order_counts[0])           # show its row of numbers
print("   B as numbers:", order_counts[1])           # show the other row of numbers
print("   are the two rows identical?", np.array_equal(order_counts[0], order_counts[1]))   # compare them
print(f"   their similarity score: {cosine(order_counts[0], order_counts[1]):.2f}")   # a perfect 1.00
print("   >> The numbers say these sentences are IDENTICAL. They are opposites.")   # spell out the failure
print("   >> This is the price of putting words in a bag and shaking it.")   # why it happens
print("   >> File 05 showed the fix for meaning: look at sentence structure.")   # one solution
print("   >> Files 08 and 09 show the modern fix: embeddings.")   # the better solution


# ==============================================================================
#  KEY TAKEAWAY
#     Vectorization = turning text into a list of numbers, because models do maths.
#     ONE-HOT      : one word -> a list with a single 1. Carries no meaning.
#     BAG OF WORDS : one document -> counts of each word. Simple and useful.
#     Both are SPARSE (mostly zeros) and both IGNORE word order.
#     Two problems remain, and the next files solve them:
#       - common words like "the" dominate the counts  -> fixed by TF-IDF (file 10)
#       - the numbers carry no meaning                 -> fixed by embeddings (file 08)
# ==============================================================================

# -> each vocabulary word becomes a list with exactly one 1 in it
# -> four complaints become four equal-length rows of numbers
# -> the two loan complaints score as more similar than unrelated ones
# -> two opposite sentences produce IDENTICAL vectors, scoring a perfect 1.00
