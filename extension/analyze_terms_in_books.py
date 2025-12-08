import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Load book names
node_path = '/Users/wang.sizh/workspace/cs7332-ml_with_graphs/Graph_Transformer_Networks/hgb_temp/NC/data/Freebase/node.dat'

book_names = []
with open(node_path, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        node_type = int(parts[2])
        if node_type == 0:  # BOOK
            name = parts[1].replace('_', ' ').lower()
            name = re.sub(r'[^a-z0-9\s]', ' ', name)
            book_names.append(name)

print("="*60)
print("FINDING OPTIMAL FEATURE DIMENSION")
print("="*60)

# Key insight: what's the max useful dimension?
# Terms appearing in fewer than min_df documents are noise

for min_df in [2, 3, 5, 10]:
    vectorizer = TfidfVectorizer(min_df=min_df, stop_words='english')
    vectorizer.fit(book_names)
    vocab_size = len(vectorizer.vocabulary_)
    print(f"\nmin_df={min_df}: {vocab_size} unique terms")

# With min_df=2, we have ~10K terms, but most are rare
# Let's see the distribution of term frequencies
print("\n" + "="*60)
print("TERM FREQUENCY DISTRIBUTION")
print("="*60)

vectorizer = TfidfVectorizer(min_df=2, stop_words='english')
X = vectorizer.fit_transform(book_names)

# Document frequency for each term
doc_freq = np.array((X > 0).sum(axis=0)).flatten()
vocab = vectorizer.get_feature_names_out()

# Sort by frequency
sorted_idx = np.argsort(doc_freq)[::-1]

print("\nTop 20 terms by document frequency:")
for i in range(20):
    idx = sorted_idx[i]
    print(f"  {vocab[idx]:20s}: appears in {doc_freq[idx]:,} books")

print("\n" + "="*60)
print("COVERAGE ANALYSIS")
print("="*60)

# How many terms to cover X% of total term occurrences?
cumsum = np.cumsum(doc_freq[sorted_idx])
total = cumsum[-1]

for coverage in [0.5, 0.7, 0.8, 0.9, 0.95]:
    n_terms = np.searchsorted(cumsum, coverage * total) + 1
    print(f"  {coverage*100:.0f}% coverage: {n_terms} terms")

print("\n" + "="*60)
print("RECOMMENDATION")
print("="*60)
print("""
Dataset comparison:
  - ACM:  1,902 dim, 109.7 avg nnz (from paper abstracts - RICH text)
  - DBLP:   334 dim,   5.6 avg nnz (from author keywords - SPARSE)
  - Freebase: book titles only - SPARSE like DBLP

Suggested dimensions:
  - 500:  Conservative, ~80% coverage
  - 1000: Moderate, ~90% coverage  
  - 1500: Aggressive, ~95% coverage

Since Freebase is "tougher", recommend 1000-dim as middle ground:
  - Larger than DBLP (334)
  - Smaller than ACM (1902) since our text is shorter
  - ~90% term coverage
""")