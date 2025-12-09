"""
Convert HGB Freebase to BOOK-centric subgraph - TRIMMED version (v3).

Key changes from v2:
1. Keep only top 3 classes (1, 2, 4) -> remap to (0, 1, 2)
2. Filter books to only those with labels in top 3 classes
3. Reduce feature dimension from 2000 to 1256 (top features by variance)
4. Remove disconnected PEOPLE/ORG nodes
5. Output to data/FREEBASE_V3 (matching IMDB-like structure)

Target statistics (similar to IMDB):
- Total nodes: ~12,000-15,000
- Target nodes: ~4,500
- Edge types: 4
- Feature dim: 1,256
- Classes: 3
"""

import os
import numpy as np
import pickle
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import re


# Node type IDs in HGB Freebase
BOOK_TYPE = 0
PEOPLE_TYPE = 4
ORG_TYPE = 6

TYPE_NAMES = {BOOK_TYPE: 'BOOK', PEOPLE_TYPE: 'PEOPLE', ORG_TYPE: 'ORGANIZATION'}

# Edge types we need from HGB
EDGE_BOOK_ORG = 4      # BOOK -> ORGANIZATION
EDGE_PEOPLE_BOOK = 14  # PEOPLE -> BOOK

# Classes to keep (top 3 by count from v2 analysis)
# Original class 1: book_character/subject (40.5%)
# Original class 2: publication/published_work (25.8%)
# Original class 4: magazine/issue/genre (14.2%)
CLASSES_TO_KEEP = {1, 2, 4}
CLASS_REMAP = {1: 0, 2: 1, 4: 2}  # Remap to 0, 1, 2

# Target feature dimension (matching IMDB)
TARGET_FEATURE_DIM = 1256


def load_all_nodes(node_path):
    """Load all nodes with their types and names."""
    nodes = {}  # node_id -> type
    names = {}  # node_id -> name

    with open(node_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            node_id = int(parts[0])
            name = parts[1]
            node_type = int(parts[2])

            nodes[node_id] = node_type
            names[node_id] = name

    print(f"Total nodes loaded: {len(nodes)}")
    return nodes, names


def tokenize_name(name):
    """Tokenize entity name for BoW."""
    text = name.replace('_', ' ').lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text


def load_labels_and_filter_classes(label_path, label_test_path, nodes):
    """
    Load labels and filter to only keep books in top 3 classes.

    Returns:
        labeled_books: dict of book_id -> remapped_label (only for top 3 classes)
        train_book_ids: set of book IDs in train split
        test_book_ids: set of book IDs in test split
    """
    labeled_books = {}
    train_book_ids = set()
    test_book_ids = set()

    class_counts = defaultdict(int)

    # Train labels
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            node_id = int(parts[0])
            label = int(parts[3].split(',')[0])

            if label in CLASSES_TO_KEEP:
                labeled_books[node_id] = CLASS_REMAP[label]
                train_book_ids.add(node_id)
                class_counts[label] += 1

    # Test labels
    with open(label_test_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            node_id = int(parts[0])
            label = int(parts[3].split(',')[0])

            if label in CLASSES_TO_KEEP:
                labeled_books[node_id] = CLASS_REMAP[label]
                test_book_ids.add(node_id)
                class_counts[label] += 1

    print(f"\nLabel filtering (keeping classes {CLASSES_TO_KEEP}):")
    print(f"  Class 1 (-> 0): {class_counts[1]} samples")
    print(f"  Class 2 (-> 1): {class_counts[2]} samples")
    print(f"  Class 4 (-> 2): {class_counts[4]} samples")
    print(f"  Total labeled books: {len(labeled_books)}")
    print(f"  Train books: {len(train_book_ids)}")
    print(f"  Test books: {len(test_book_ids)}")

    return labeled_books, train_book_ids, test_book_ids


def filter_books_by_vocabulary_and_labels(nodes, names, labeled_books, max_features=2000):
    """
    Filter books to only those:
    1. In the labeled set (top 3 classes)
    2. Have at least one feature from vocabulary

    Returns:
        kept_book_ids: set of book IDs to keep
        book_features: sparse feature matrix for kept books
        feature_indices: indices of top features by variance (for dimension reduction)
    """
    # Get book IDs that have labels in top 3 classes
    candidate_book_ids = [nid for nid, ntype in nodes.items()
                         if ntype == BOOK_TYPE and nid in labeled_books]
    book_names = [tokenize_name(names[nid]) for nid in candidate_book_ids]

    print(f"\nCandidate books (with labels in top 3 classes): {len(candidate_book_ids)}")

    # Build vocabulary from these books
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=2,
        stop_words='english'
    )

    all_features = vectorizer.fit_transform(book_names)
    print(f"Initial vocabulary size: {len(vectorizer.vocabulary_)}")

    # Find books with at least one feature
    row_sums = np.array(all_features.sum(axis=1)).flatten()
    has_features = row_sums > 0

    kept_indices = np.where(has_features)[0]
    kept_book_ids = set(candidate_book_ids[i] for i in kept_indices)
    book_features = all_features[kept_indices]

    removed = len(candidate_book_ids) - len(kept_book_ids)
    print(f"Books with features: {len(kept_book_ids)} ({100*len(kept_book_ids)/len(candidate_book_ids):.1f}%)")
    print(f"Books removed (no features): {removed}")

    # Select top features by variance for dimension reduction
    print(f"\nReducing features from {book_features.shape[1]} to {TARGET_FEATURE_DIM}...")

    # Convert to dense for variance calculation
    features_dense = book_features.toarray()
    feature_variances = np.var(features_dense, axis=0)

    # Get indices of top features by variance
    top_feature_indices = np.argsort(feature_variances)[-TARGET_FEATURE_DIM:]
    top_feature_indices = np.sort(top_feature_indices)  # Keep original order

    # Reduce features
    reduced_features = features_dense[:, top_feature_indices]

    print(f"Feature reduction complete: {book_features.shape[1]} -> {reduced_features.shape[1]}")
    print(f"Variance retained: {feature_variances[top_feature_indices].sum() / feature_variances.sum() * 100:.1f}%")

    return kept_book_ids, reduced_features, top_feature_indices, candidate_book_ids, kept_indices


def load_edges_for_types(link_path, nodes, kept_book_ids):
    """
    Load edges connecting BOOK-PEOPLE and BOOK-ORG.
    Only keep edges where the book is in kept_book_ids.
    """
    pb_edges = []  # PEOPLE -> BOOK
    bo_edges = []  # BOOK -> ORG
    connected_people = set()
    connected_orgs = set()

    with open(link_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            h_id, t_id, r_id = int(parts[0]), int(parts[1]), int(parts[2])

            if r_id == EDGE_PEOPLE_BOOK:  # PEOPLE -> BOOK
                if t_id in kept_book_ids:
                    h_type = nodes.get(h_id, -1)
                    if h_type == PEOPLE_TYPE:
                        pb_edges.append((h_id, t_id))
                        connected_people.add(h_id)

            elif r_id == EDGE_BOOK_ORG:  # BOOK -> ORG
                if h_id in kept_book_ids:
                    t_type = nodes.get(t_id, -1)
                    if t_type == ORG_TYPE:
                        bo_edges.append((h_id, t_id))
                        connected_orgs.add(t_id)

    print(f"\nEdges loaded:")
    print(f"  PEOPLE -> BOOK: {len(pb_edges)}")
    print(f"  BOOK -> ORG: {len(bo_edges)}")
    print(f"  Connected PEOPLE: {len(connected_people)}")
    print(f"  Connected ORG: {len(connected_orgs)}")

    return pb_edges, bo_edges, connected_people, connected_orgs


def create_unified_node_mapping(kept_book_ids, connected_people, connected_orgs):
    """
    Create unified node ID space: [BOOK nodes] [PEOPLE nodes] [ORG nodes]
    """
    old_to_new = {}

    # Books first (sorted for reproducibility)
    book_list = sorted(kept_book_ids)
    for i, old_id in enumerate(book_list):
        old_to_new[old_id] = i
    book_end = len(book_list)

    # People next
    people_list = sorted(connected_people)
    for i, old_id in enumerate(people_list):
        old_to_new[old_id] = book_end + i
    people_end = book_end + len(people_list)

    # Organizations last
    org_list = sorted(connected_orgs)
    for i, old_id in enumerate(org_list):
        old_to_new[old_id] = people_end + i
    org_end = people_end + len(org_list)

    node_counts = {
        'BOOK': len(book_list),
        'PEOPLE': len(people_list),
        'ORG': len(org_list),
        'TOTAL': org_end
    }

    node_ranges = {
        'BOOK': (0, book_end),
        'PEOPLE': (book_end, people_end),
        'ORG': (people_end, org_end)
    }

    print(f"\nUnified node space:")
    print(f"  BOOK: 0 - {book_end-1} ({node_counts['BOOK']} nodes)")
    print(f"  PEOPLE: {book_end} - {people_end-1} ({node_counts['PEOPLE']} nodes)")
    print(f"  ORG: {people_end} - {org_end-1} ({node_counts['ORG']} nodes)")
    print(f"  TOTAL: {node_counts['TOTAL']} nodes")

    return old_to_new, node_counts, node_ranges, book_list


def create_edge_matrices(pb_edges, bo_edges, old_to_new, num_nodes):
    """
    Create 4 edge type matrices like DBLP/IMDB:
    - BP: BOOK -> PEOPLE
    - PB: PEOPLE -> BOOK (= BP.T)
    - BO: BOOK -> ORG
    - OB: ORG -> BOOK (= BO.T)
    """
    # PEOPLE -> BOOK edges (PB)
    pb_rows = [old_to_new[p] for p, b in pb_edges]
    pb_cols = [old_to_new[b] for p, b in pb_edges]
    pb_data = [1.0] * len(pb_edges)

    PB = csr_matrix((pb_data, (pb_rows, pb_cols)), shape=(num_nodes, num_nodes), dtype=np.float32)
    BP = PB.T.tocsr()  # BOOK -> PEOPLE is transpose

    # BOOK -> ORG edges (BO)
    bo_rows = [old_to_new[b] for b, o in bo_edges]
    bo_cols = [old_to_new[o] for b, o in bo_edges]
    bo_data = [1.0] * len(bo_edges)

    BO = csr_matrix((bo_data, (bo_rows, bo_cols)), shape=(num_nodes, num_nodes), dtype=np.float32)
    OB = BO.T.tocsr()  # ORG -> BOOK is transpose

    edges = [BP, PB, BO, OB]
    edge_names = ['BP (BOOK->PEOPLE)', 'PB (PEOPLE->BOOK)',
                  'BO (BOOK->ORG)', 'OB (ORG->BOOK)']

    print(f"\nEdge matrices (4 types):")
    for i, (e, name) in enumerate(zip(edges, edge_names)):
        print(f"  Type {i}: {name} - {e.nnz} edges")

    return edges


def create_node_features(book_features, pb_edges, bo_edges, old_to_new,
                         node_counts, node_ranges, book_list):
    """
    Create node features:
    - BOOK: TF-IDF (already reduced to 1256-dim)
    - PEOPLE: Aggregated from connected books
    - ORG: Aggregated from connected books
    """
    num_books = node_counts['BOOK']
    num_people = node_counts['PEOPLE']
    num_orgs = node_counts['ORG']
    feat_dim = book_features.shape[1]

    print(f"\nCreating node features ({feat_dim}-dim):")
    print(f"  BOOK features: {book_features.shape}")

    # Build book_id -> feature_row mapping
    book_id_to_row = {book_id: i for i, book_id in enumerate(book_list)}

    # Create PEOPLE features by aggregating connected book features
    people_start, people_end = node_ranges['PEOPLE']

    person_to_books = defaultdict(list)
    for p_old, b_old in pb_edges:
        if b_old in book_id_to_row:
            p_new = old_to_new[p_old]
            book_row = book_id_to_row[b_old]
            person_to_books[p_new].append(book_row)

    people_features = np.zeros((num_people, feat_dim), dtype=np.float32)
    for p_new in range(people_start, people_end):
        book_rows = person_to_books[p_new]
        if book_rows:
            people_features[p_new - people_start] = book_features[book_rows].mean(axis=0)

    print(f"  PEOPLE features: ({num_people}, {feat_dim})")
    people_nonzero = np.sum(np.any(people_features != 0, axis=1))
    print(f"    Non-zero rows: {people_nonzero} ({100*people_nonzero/num_people:.1f}%)")

    # Create ORG features by aggregating connected book features
    org_start, org_end = node_ranges['ORG']

    org_to_books = defaultdict(list)
    for b_old, o_old in bo_edges:
        if b_old in book_id_to_row:
            o_new = old_to_new[o_old]
            book_row = book_id_to_row[b_old]
            org_to_books[o_new].append(book_row)

    org_features = np.zeros((num_orgs, feat_dim), dtype=np.float32)
    for o_new in range(org_start, org_end):
        book_rows = org_to_books[o_new]
        if book_rows:
            org_features[o_new - org_start] = book_features[book_rows].mean(axis=0)

    print(f"  ORG features: ({num_orgs}, {feat_dim})")
    org_nonzero = np.sum(np.any(org_features != 0, axis=1))
    print(f"    Non-zero rows: {org_nonzero} ({100*org_nonzero/num_orgs:.1f}%)")

    # Concatenate all features: [BOOK, PEOPLE, ORG]
    all_features = np.vstack([book_features, people_features, org_features])

    print(f"\n  Combined features: {all_features.shape}")

    # Calculate sparsity
    sparsity = (all_features == 0).sum() / all_features.size * 100
    print(f"  Sparsity: {sparsity:.2f}%")

    return all_features.astype(np.float32)


def create_labels(labeled_books, train_book_ids, test_book_ids, kept_book_ids, old_to_new):
    """
    Create train/val/test label splits.
    Uses 2:1 train:val split like DBLP/ACM.
    """
    train_labels = []
    test_labels = []

    for book_id in train_book_ids:
        if book_id in kept_book_ids:
            new_id = old_to_new[book_id]
            label = labeled_books[book_id]
            train_labels.append([new_id, label])

    for book_id in test_book_ids:
        if book_id in kept_book_ids:
            new_id = old_to_new[book_id]
            label = labeled_books[book_id]
            test_labels.append([new_id, label])

    print(f"\nLabels created:")
    print(f"  Train (before split): {len(train_labels)}")
    print(f"  Test: {len(test_labels)}")

    return train_labels, test_labels


def save_gtn_format(edges, features, train_labels, test_labels, output_path):
    """Save in GTN format with 2:1 train:val split."""

    # Convert labels to arrays
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    # Split train into train/val with 2:1 ratio
    np.random.seed(42)
    indices = np.random.permutation(len(train_labels))
    split = int(len(train_labels) * (2/3))

    train_set = train_labels[indices[:split]]
    val_set = train_labels[indices[split:]]

    labels = [train_set, val_set, test_labels]

    # Get class distribution
    all_labels_arr = np.concatenate([train_set[:, 1], val_set[:, 1], test_labels[:, 1]])
    unique_labels, counts = np.unique(all_labels_arr, return_counts=True)

    print(f"\nLabel splits (2:1 train:val ratio):")
    print(f"  Train: {len(train_set)}")
    print(f"  Val: {len(val_set)}")
    print(f"  Test: {len(test_labels)}")
    print(f"  Ratio train:val = {len(train_set)/len(val_set):.2f}:1")
    print(f"\nClass distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  Class {label}: {count} ({100*count/len(all_labels_arr):.1f}%)")

    # Save
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, 'edges.pkl'), 'wb') as f:
        pickle.dump(edges, f)

    with open(os.path.join(output_path, 'node_features.pkl'), 'wb') as f:
        pickle.dump(features, f)

    with open(os.path.join(output_path, 'labels.pkl'), 'wb') as f:
        pickle.dump(labels, f)

    print(f"\nSaved to {output_path}")
    print(f"  edges.pkl: {len(edges)} edge types")
    print(f"  node_features.pkl: {features.shape}")
    print(f"  labels.pkl: train={len(train_set)}, val={len(val_set)}, test={len(test_labels)}")


def main():
    # Source from HGB Freebase (same as v2)
    hgb_path = '../hgb_temp/NC/data/Freebase'
    output_path = '../data/FREEBASE_V3'

    print("=" * 70)
    print("Converting Freebase to BOOK-centric subgraph - TRIMMED (v3)")
    print("=" * 70)
    print("\nKey features:")
    print("  - Keep only top 3 classes (1, 2, 4) -> remap to (0, 1, 2)")
    print("  - Filter books to those with labels + features")
    print("  - Reduce features from 2000 to 1256 (top by variance)")
    print("  - 4 edge types like IMDB: BP, PB, BO, OB")
    print("  - 2:1 train:val split")
    print("  - Target: IMDB-like statistics")
    print("=" * 70)

    # Step 1: Load all nodes
    print("\n[1/8] Loading nodes...")
    nodes, names = load_all_nodes(os.path.join(hgb_path, 'node.dat'))

    # Step 2: Load labels and filter to top 3 classes
    print("\n[2/8] Loading labels and filtering classes...")
    labeled_books, train_book_ids, test_book_ids = load_labels_and_filter_classes(
        os.path.join(hgb_path, 'label.dat'),
        os.path.join(hgb_path, 'label.dat.test'),
        nodes
    )

    # Step 3: Filter books by vocabulary and reduce features
    print("\n[3/8] Filtering books and reducing features...")
    kept_book_ids, book_features, top_feature_indices, _, _ = \
        filter_books_by_vocabulary_and_labels(nodes, names, labeled_books, max_features=2000)

    # Step 4: Load edges and find connected PEOPLE/ORG
    print("\n[4/8] Loading edges...")
    pb_edges, bo_edges, connected_people, connected_orgs = \
        load_edges_for_types(os.path.join(hgb_path, 'link.dat'), nodes, kept_book_ids)

    # Step 5: Create unified node mapping
    print("\n[5/8] Creating unified node mapping...")
    old_to_new, node_counts, node_ranges, book_list = \
        create_unified_node_mapping(kept_book_ids, connected_people, connected_orgs)

    # Step 6: Create edge matrices
    print("\n[6/8] Creating edge matrices...")
    edges = create_edge_matrices(pb_edges, bo_edges, old_to_new, node_counts['TOTAL'])

    # Step 7: Create node features with propagation
    print("\n[7/8] Creating node features...")
    features = create_node_features(
        book_features, pb_edges, bo_edges, old_to_new,
        node_counts, node_ranges, book_list
    )

    # Step 8: Create labels
    print("\n[8/8] Creating labels...")
    train_labels, test_labels = create_labels(
        labeled_books, train_book_ids, test_book_ids, kept_book_ids, old_to_new
    )

    # Save
    print("\n" + "=" * 70)
    print("Saving GTN format...")
    print("=" * 70)
    save_gtn_format(edges, features, train_labels, test_labels, output_path)

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY - Comparison with IMDB")
    print("=" * 70)
    total_labeled = len(train_labels) * 2 // 3 + len(train_labels) // 3 + len(test_labels)
    print(f"{'':25} {'FREEBASE_V3':>15} {'IMDB':>10}")
    print(f"{'Total nodes':25} {node_counts['TOTAL']:>15,} {12772:>10,}")
    print(f"{'Target nodes':25} {node_counts['BOOK']:>15,} {4658:>10,}")
    print(f"{'Edge types':25} {len(edges):>15} {4:>10}")
    print(f"{'Feature dim':25} {features.shape[1]:>15} {1256:>10}")
    print(f"{'Classes':25} {3:>15} {3:>10}")
    print(f"{'Train':25} {len(train_labels)*2//3:>15} {300:>10}")
    print(f"{'Val':25} {len(train_labels)//3:>15} {300:>10}")
    print(f"{'Test':25} {len(test_labels):>15} {2339:>10}")

    print("\nDone!")


if __name__ == '__main__':
    main()
