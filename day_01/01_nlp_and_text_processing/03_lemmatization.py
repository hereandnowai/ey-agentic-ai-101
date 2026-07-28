# what is lemmatization?
# lemmatization is the process of reducing a word to its base or root form, known as a lemma. Unlike stemming, which simply removes prefixes or suffixes, lemmatization considers the context and meaning of the word to ensure that the root form is a valid word in the language. For example, the words "running," "ran," and "runs" would all be reduced to the lemma "run." This process is particularly useful in natural language processing (NLP) tasks such as text analysis, information retrieval, and machine learning, where understanding the underlying meaning of words is important.

# better -> good
# was -> be
# paid -> pay
# banks -> bank

# stemming is chopping the endings off
# lemmatization is more sophisticated and considers the context of the word

# "saw"
# as a verb saw -> see
# as a noun -> saw (cutting tool)

import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
import spacy

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
nlp = spacy.load("en_core_web_sm")

# part 1 - nltk - we must tell it the word's job
print("\n 1. lemmatization word by word (nltk)")
print(" treating each word as a noun")
for w in ["banks", "payments", "policies", "analyses", "earnings"]:
    print(f"{w:<15} -> {lemmatizer.lemmatize(w, pos='n')}")

print(" treating each word as a verb")
for w in ["paid", "approving", "invested", "defaulted", "was"]:
    print(f"{w:<15} -> {lemmatizer.lemmatize(w, pos='v')}")

print(" the SAME word, read two different ways")
print(f" saw as a noun -> {lemmatizer.lemmatize('saw', pos='n')}")
print(f" saw as a verb -> {lemmatizer.lemmatize('saw', pos='v')}")
