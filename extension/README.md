# Freebase_book Dataset

A BOOK-centric subgraph extracted from HGB Freebase for GTN evaluation.

## Quick Start

```bash
# Convert from HGB format (requires hgb_temp/NC/data/Freebase)
python convert_freebase_book_v2.py

# Train GTN
python ../main.py --model GTN --dataset Freebase_book --data_path ./extension --epoch 100 --num_layers 2 --num_channels 2 --lr 0.01
```

## Dataset Statistics

| Property | Freebase_book | DBLP | ACM |
|----------|---------------|------|-----|
| Total nodes | 38,128 | 18,405 | 8,994 |
| Target nodes | 27,858 (BOOK) | 4,057 | 3,025 |
| Edge types | 4 | 4 | 4 |
| Feature dim | 2,000 | 334 | 1,902 |
| Classes | 7 | 4 | 3 |
| Train | 1,116 | 800 | 600 |
| Val | 559 | 400 | 300 |
| Test | 3,885 | 2,857 | 2,125 |

## Schema

```
Node Types (unified ID space):
  BOOK:   0 - 27,857   (27,858 nodes) - target
  PEOPLE: 27,858 - 35,862 (8,005 nodes)
  ORG:    35,863 - 38,127 (2,265 nodes)

Edge Types (like DBLP):
  0: BP (BOOK -> PEOPLE)   16,567 edges
  1: PB (PEOPLE -> BOOK)   16,567 edges  = BP.T
  2: BO (BOOK -> ORG)      15,724 edges
  3: OB (ORG -> BOOK)      15,724 edges  = BO.T
```

## Node Features

- **2000-dim TF-IDF** from book names
- Books with no features (31%) were removed
- PEOPLE/ORG features propagated from connected books via averaging
- Sparsity: 99.88% (comparable to IMDB's 99.57%)

## Labels (7 book categories)

| Class | Category | Samples | % |
|-------|----------|---------|---|
| 0 | scholarly_work | 384 | 6.9% |
| 1 | book_character/subject | 2,254 | 40.5% |
| 2 | publication/published_work | 1,432 | 25.8% |
| 3 | short_story | 124 | 2.2% |
| 4 | magazine/issue/genre | 789 | 14.2% |
| 5 | newspaper | 251 | 4.5% |
| 6 | journal_article/journal | 326 | 5.9% |

Note: Class imbalance exists (Class 1: 40.5%, Class 3: 2.2%)

## Files

```
extension/
├── Freebase_book/
│   ├── edges.pkl          # [BP, PB, BO, OB] sparse matrices
│   ├── node_features.pkl  # (38128, 2000) float32
│   └── labels.pkl         # [train, val, test] arrays
├── convert_freebase_book_v2.py  # Conversion script
└── README.md
```

## Source

Extracted from [HGB (Heterogeneous Graph Benchmark)](https://github.com/THUDM/HGB) Freebase dataset.
