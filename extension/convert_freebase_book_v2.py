"""
Convert HGB Freebase to BOOK-centric subgraph with BoW features (v2).

Key changes from v1:
1. Filter books to only those with top-2000 frequent terms (remove ~31% empty-feature books)
2. Use 4 edge types like DBLP: BP, PB (=BP.T), BO, OB (=BO.T)
3. Propagate features to PEOPLE/ORG via adjacency multiplication
4. Use 2:1 train:val split (matching ACM/DBLP)

Schema: BOOK + PEOPLE + ORGANIZATION (like DBLP's Paper-Author-Venue)
"""

import os
import numpy as np
import pickle
import scipy.sparse as sp
from scipy.sparse import csr_matrix, vstack
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize
import re


# Node type IDs in HGB Freebase
BOOK_TYPE = 0
PEOPLE_TYPE = 4
ORG_TYPE = 6

TYPE_NAMES = {BOOK_TYPE: 'BOOK', PEOPLE_TYPE: 'PEOPLE', ORG_TYPE: 'ORGANIZATION'}

# Edge types we need from HGB (will create 4 output types)
# Edge 4: BOOK -> ORGANIZATION
# Edge 14: PEOPLE -> BOOK
EDGE_BOOK_ORG = 4      # BOOK-about-ORGANIZATION
EDGE_PEOPLE_BOOK = 14  # PEOPLE-to-BOOK


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


def filter_books_by_vocabulary(nodes, names, max_features=2000):
    """
    Filter books to only those containing at least one term from top-2000 vocabulary.

    Returns:
        kept_book_ids: set of book IDs to keep
        vectorizer: fitted TfidfVectorizer
        book_features: sparse feature matrix for kept books
    """
    # Get all book IDs and their tokenized names
    book_ids = [nid for nid, ntype in nodes.items() if ntype == BOOK_TYPE]
    book_names = [tokenize_name(names[nid]) for nid in book_ids]

    print(f"\nTotal books: {len(book_ids)}")

    # First pass: build vocabulary from all books
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=2,
        stop_words='english'
    )

    # Fit and transform all books
    all_features = vectorizer.fit_transform(book_names)

    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Sample terms: {list(vectorizer.vocabulary_.keys())[:10]}")

    # Find books with at least one feature (non-zero row)
    row_sums = np.array(all_features.sum(axis=1)).flatten()
    has_features = row_sums > 0

    kept_indices = np.where(has_features)[0]
    kept_book_ids = set(book_ids[i] for i in kept_indices)

    # Extract features for kept books only
    book_features = all_features[kept_indices]

    removed = len(book_ids) - len(kept_book_ids)
    print(f"\nBooks with features: {len(kept_book_ids)} ({100*len(kept_book_ids)/len(book_ids):.1f}%)")
    print(f"Books removed (no features): {removed} ({100*removed/len(book_ids):.1f}%)")

    # Return mapping from kept book ID to its feature row index
    book_id_to_feat_idx = {book_ids[i]: idx for idx, i in enumerate(kept_indices)}

    return kept_book_ids, vectorizer, book_features, book_id_to_feat_idx


def load_edges_for_types(link_path, nodes, kept_book_ids):
    """
    Load edges connecting BOOK-PEOPLE and BOOK-ORG.
    Only keep edges where the book is in kept_book_ids.

    Returns:
        pb_edges: list of (people_id, book_id) - PEOPLE -> BOOK
        bo_edges: list of (book_id, org_id) - BOOK -> ORG
        connected_people: set of people IDs connected to kept books
        connected_orgs: set of org IDs connected to kept books
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
                if t_id in kept_book_ids:  # book must be kept
                    h_type = nodes.get(h_id, -1)
                    if h_type == PEOPLE_TYPE:
                        pb_edges.append((h_id, t_id))
                        connected_people.add(h_id)

            elif r_id == EDGE_BOOK_ORG:  # BOOK -> ORG
                if h_id in kept_book_ids:  # book must be kept
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

    Returns:
        old_to_new: mapping from original ID to new unified ID
        node_counts: dict with counts per type
        node_ranges: dict with (start, end) ranges per type
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
    Create 4 edge type matrices like DBLP:
    - BP: BOOK -> PEOPLE
    - PB: PEOPLE -> BOOK (= BP.T)
    - BO: BOOK -> ORG
    - OB: ORG -> BOOK (= BO.T)

    Returns list of 4 sparse matrices.
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

    print(f"\nEdge matrices (4 types like DBLP):")
    for i, (e, name) in enumerate(zip(edges, edge_names)):
        print(f"  Type {i}: {name} - {e.nnz} edges")

    return edges


def create_node_features(book_features, pb_edges, bo_edges, old_to_new,
                         node_counts, node_ranges, book_list):
    """
    Create node features:
    - BOOK: TF-IDF from book names (already computed)
    - PEOPLE: Aggregated from connected books (like ACM)
    - ORG: Aggregated from connected books (like ACM)

    All node types share the same feature space (2000-dim).
    """
    num_books = node_counts['BOOK']
    num_people = node_counts['PEOPLE']
    num_orgs = node_counts['ORG']
    feat_dim = book_features.shape[1]

    # Book features are already in order (book_list is sorted same as our mapping)
    # But we need to reorder based on old_to_new mapping
    # Since book_list is sorted and old_to_new assigns 0..n-1 to sorted books,
    # the features are already in correct order

    print(f"\nCreating node features ({feat_dim}-dim):")
    print(f"  BOOK features: {book_features.shape}")

    # Create PEOPLE features by aggregating connected book features
    # For each person, average the features of books they're connected to
    people_start, people_end = node_ranges['PEOPLE']

    # Build person -> list of book feature indices
    person_to_books = defaultdict(list)
    for p_old, b_old in pb_edges:
        p_new = old_to_new[p_old]
        b_new = old_to_new[b_old]  # This is the book's index in unified space (= feature row)
        person_to_books[p_new].append(b_new)

    # Create people feature matrix
    people_features = np.zeros((num_people, feat_dim), dtype=np.float32)
    for p_new in range(people_start, people_end):
        book_indices = person_to_books[p_new]
        if book_indices:
            # Average of connected book features
            book_feats = book_features[book_indices].toarray() if sp.issparse(book_features) else book_features[book_indices]
            people_features[p_new - people_start] = book_feats.mean(axis=0)

    print(f"  PEOPLE features: ({num_people}, {feat_dim})")
    people_nonzero = np.sum(np.any(people_features != 0, axis=1))
    print(f"    Non-zero rows: {people_nonzero} ({100*people_nonzero/num_people:.1f}%)")

    # Create ORG features by aggregating connected book features
    org_start, org_end = node_ranges['ORG']

    org_to_books = defaultdict(list)
    for b_old, o_old in bo_edges:
        o_new = old_to_new[o_old]
        b_new = old_to_new[b_old]
        org_to_books[o_new].append(b_new)

    org_features = np.zeros((num_orgs, feat_dim), dtype=np.float32)
    for o_new in range(org_start, org_end):
        book_indices = org_to_books[o_new]
        if book_indices:
            book_feats = book_features[book_indices].toarray() if sp.issparse(book_features) else book_features[book_indices]
            org_features[o_new - org_start] = book_feats.mean(axis=0)

    print(f"  ORG features: ({num_orgs}, {feat_dim})")
    org_nonzero = np.sum(np.any(org_features != 0, axis=1))
    print(f"    Non-zero rows: {org_nonzero} ({100*org_nonzero/num_orgs:.1f}%)")

    # Concatenate all features: [BOOK, PEOPLE, ORG]
    book_feat_dense = book_features.toarray() if sp.issparse(book_features) else book_features
    all_features = np.vstack([book_feat_dense, people_features, org_features])

    print(f"\n  Combined features: {all_features.shape}")

    return all_features.astype(np.float32)


def load_labels(label_path, label_test_path, nodes, old_to_new, kept_book_ids):
    """
    Load labels for BOOK nodes (only those in kept_book_ids).
    """
    train_labels = []
    test_labels = []

    # Train labels
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            node_id = int(parts[0])

            if node_id in kept_book_ids:
                label = int(parts[3].split(',')[0])
                new_id = old_to_new[node_id]
                train_labels.append([new_id, label])

    # Test labels
    with open(label_test_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            node_id = int(parts[0])

            if node_id in kept_book_ids:
                label = int(parts[3].split(',')[0])
                new_id = old_to_new[node_id]
                test_labels.append([new_id, label])

    print(f"\nLabels (for kept BOOK nodes):")
    print(f"  Train (from HGB): {len(train_labels)}")
    print(f"  Test (from HGB): {len(test_labels)}")

    return train_labels, test_labels


def save_gtn_format(edges, features, train_labels, test_labels, output_path):
    """Save in GTN format with 2:1 train:val split."""

    num_nodes = features.shape[0]

    # Convert labels to arrays
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    # Split train into train/val with 2:1 ratio (67/33)
    np.random.seed(42)
    indices = np.random.permutation(len(train_labels))
    split = int(len(train_labels) * (2/3))  # 2:1 ratio

    train_set = train_labels[indices[:split]]
    val_set = train_labels[indices[split:]]

    labels = [train_set, val_set, test_labels]

    # Get class distribution
    all_labels = np.concatenate([train_set[:, 1], val_set[:, 1], test_labels[:, 1]])
    unique_labels = np.unique(all_labels)

    print(f"\nLabel splits (2:1 train:val ratio):")
    print(f"  Train: {len(train_set)}")
    print(f"  Val: {len(val_set)}")
    print(f"  Test: {len(test_labels)}")
    print(f"  Ratio train:val = {len(train_set)/len(val_set):.2f}:1")
    print(f"  Number of classes: {len(unique_labels)}")

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
    hgb_path = '../hgb_temp/NC/data/Freebase'
    output_path = './Freebase_book'

    print("=" * 70)
    print("Converting Freebase to BOOK-centric subgraph (v2)")
    print("=" * 70)
    print("\nKey features:")
    print("  - Filter books to those with top-2000 term vocabulary")
    print("  - 4 edge types like DBLP: BP, PB, BO, OB")
    print("  - Feature propagation to PEOPLE/ORG")
    print("  - 2:1 train:val split")
    print("=" * 70)

    # Step 1: Load all nodes
    print("\n[1/7] Loading nodes...")
    nodes, names = load_all_nodes(os.path.join(hgb_path, 'node.dat'))

    # Step 2: Filter books by vocabulary (remove those with no features)
    print("\n[2/7] Filtering books by vocabulary...")
    kept_book_ids, vectorizer, book_features, book_id_to_feat_idx = \
        filter_books_by_vocabulary(nodes, names, max_features=2000)

    # Step 3: Load edges and find connected PEOPLE/ORG
    print("\n[3/7] Loading edges...")
    pb_edges, bo_edges, connected_people, connected_orgs = \
        load_edges_for_types(os.path.join(hgb_path, 'link.dat'), nodes, kept_book_ids)

    # Step 4: Create unified node mapping
    print("\n[4/7] Creating unified node mapping...")
    old_to_new, node_counts, node_ranges, book_list = \
        create_unified_node_mapping(kept_book_ids, connected_people, connected_orgs)

    # Step 5: Create edge matrices (4 types like DBLP)
    print("\n[5/7] Creating edge matrices...")
    edges = create_edge_matrices(pb_edges, bo_edges, old_to_new, node_counts['TOTAL'])

    # Step 6: Create node features with propagation
    print("\n[6/7] Creating node features...")
    features = create_node_features(
        book_features, pb_edges, bo_edges, old_to_new,
        node_counts, node_ranges, book_list
    )

    # Step 7: Load labels
    print("\n[7/7] Loading labels...")
    train_labels, test_labels = load_labels(
        os.path.join(hgb_path, 'label.dat'),
        os.path.join(hgb_path, 'label.dat.test'),
        nodes, old_to_new, kept_book_ids
    )

    # Save
    print("\n" + "=" * 70)
    print("Saving GTN format...")
    print("=" * 70)
    save_gtn_format(edges, features, train_labels, test_labels, output_path)

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY - Comparison with DBLP/ACM")
    print("=" * 70)
    print(f"{'':25} {'Freebase_book':>15} {'DBLP':>10} {'ACM':>10}")
    print(f"{'Total nodes':25} {node_counts['TOTAL']:>15,} {18405:>10,} {8994:>10,}")
    print(f"{'Target nodes':25} {node_counts['BOOK']:>15,} {4057:>10,} {3025:>10,}")
    print(f"{'Edge types':25} {len(edges):>15} {4:>10} {4:>10}")
    print(f"{'Feature dim':25} {features.shape[1]:>15} {334:>10} {1902:>10}")
    print(f"{'Classes':25} {7:>15} {4:>10} {3:>10}")
    print(f"{'Train':25} {len(train_labels)*2//3:>15} {800:>10} {600:>10}")
    print(f"{'Val':25} {len(train_labels)//3:>15} {400:>10} {300:>10}")
    print(f"{'Test':25} {len(test_labels):>15} {2857:>10} {2125:>10}")

    print("\nDone!")


if __name__ == '__main__':
    main()
