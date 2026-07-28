# invest, investing, invested, investment, investments, investor, investors
# stemming is chopping the endings off
# invest

# business -> busi

import nltk
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

porter = PorterStemmer()
snowball = SnowballStemmer("english")

# part 1
words = [
    "invest", "investing", "invested", "investment", "investments", "investor", "investors",
    "banking", "banked", "banker", "bankers", "banks",
    "payment", "payments", "paying", "paid",
    "approval", "approvals", "approving", "approved",
    ]

print("\n 1. stemming word by word")
print(f"{'ORIGINAL WORD':<15} {'PORTER STEMMER':<20} {'SNOWBALL STEMMER':<20}")
print("-" * 60)
for w in words:
    print(f"{w:<15} {porter.stem(w):<20} {snowball.stem(w):<20}")