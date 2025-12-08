"""
Convert HGB Freebase to BOOK-centric subgraph with BoW features.

Subgraph: BOOK + PEOPLE + ORGANIZATION (like DBLP's Paper-Author-Venue)
- 6 edge types (vs original 36)
- BoW features from entity names (vs 8-dim type one-hot)

This creates a fair comparison with ACM/DBLP while still being a different domain.
"""

import os
import numpy as np
import pickle
import scipy.sparse as sp
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import re


# Schema
KEEP_NODE_TYPES = {0, 4, 6}  # BOOK, PEOPLE, ORGANIZATION
TYPE_NAMES = {0: 'BOOK', 4: 'PEOPLE', 6: 'ORGANIZATION'}

# Edge types to keep (both endpoints in KEEP_NODE_TYPES)
KEEP_EDGE_TYPES = {
    0: 'BOOK-and-BOOK',
    4: 'BOOK-about-ORGANIZATION',
    14: 'PEOPLE-to-BOOK',
    18: 'PEOPLE-and-PEOPLE',
    20: 'PEOPLE-in-ORGANIZATION',
    28: 'ORGANIZATION-and-ORGANIZATION'
}


def load_nodes(node_path):
    """Load nodes and filter to keep types."""
    nodes = {}
    names = {}

    with open(node_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            node_id = int(parts[0])
            name = parts[1]
            node_type = int(parts[2])

            nodes[node_id] = node_type
            names[node_id] = name

    # Filter to keep types
    kept_nodes = {nid: ntype for nid, ntype in nodes.items() if ntype in KEEP_NODE_TYPES}
    kept_names = {nid: names[nid] for nid in kept_nodes}

    print(f"Total nodes: {len(nodes)} -> Kept: {len(kept_nodes)}")
    for t in sorted(KEEP_NODE_TYPES):
        count = sum(1 for ntype in kept_nodes.values() if ntype == t)
        print(f"  {TYPE_NAMES[t]}: {count:,}")

    return kept_nodes, kept_names, nodes  # Return all nodes for edge filtering


def create_node_mapping(kept_nodes):
    """Create old_id -> new_id mapping for kept nodes."""
    old_to_new = {}
    new_id = 0
    for old_id in sorted(kept_nodes.keys()):
        old_to_new[old_id] = new_id
        new_id += 1
    return old_to_new


def load_edges(link_path, all_nodes, old_to_new):
    """Load edges and filter to kept node types."""
    edges = defaultdict(list)

    with open(link_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            h_id, t_id, r_id = int(parts[0]), int(parts[1]), int(parts[2])
            weight = float(parts[3]) if len(parts) > 3 else 1.0

            # Check if both endpoints are in kept types
            h_type = all_nodes.get(h_id, -1)
            t_type = all_nodes.get(t_id, -1)

            if h_type in KEEP_NODE_TYPES and t_type in KEEP_NODE_TYPES:
                if r_id in KEEP_EDGE_TYPES:
                    # Remap to new node IDs
                    new_h = old_to_new[h_id]
                    new_t = old_to_new[t_id]
                    edges[r_id].append((new_h, new_t, weight))

    print(f"\nEdge counts:")
    total = 0
    for r_id in sorted(edges.keys()):
        print(f"  {KEEP_EDGE_TYPES[r_id]}: {len(edges[r_id]):,}")
        total += len(edges[r_id])
    print(f"  TOTAL: {total:,}")

    return edges


def load_labels(label_path, label_test_path, all_nodes, old_to_new):
    """Load labels for BOOK nodes."""
    train_labels = []
    test_labels = []

    # Train labels
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            node_id = int(parts[0])
            node_type = all_nodes.get(node_id, -1)

            if node_type == 0:  # BOOK
                label = int(parts[3].split(',')[0])  # Take first label
                new_id = old_to_new[node_id]
                train_labels.append([new_id, label])

    # Test labels
    with open(label_test_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            node_id = int(parts[0])
            node_type = all_nodes.get(node_id, -1)

            if node_type == 0:  # BOOK
                label = int(parts[3].split(',')[0])
                new_id = old_to_new[node_id]
                test_labels.append([new_id, label])

    print(f"\nLabels: train={len(train_labels)}, test={len(test_labels)}")

    return train_labels, test_labels


def create_bow_features(kept_names, old_to_new, max_features=2000):
    """Create bag-of-words features from entity names."""

    # Prepare documents in new ID order
    num_nodes = len(old_to_new)
    documents = [''] * num_nodes

    for old_id, name in kept_names.items():
        new_id = old_to_new[old_id]
        # Tokenize: replace underscores with spaces, lowercase
        text = name.replace('_', ' ').lower()
        # Remove special characters
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        documents[new_id] = text

    # Create TF-IDF features
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=2,  # Ignore terms that appear in fewer than 2 documents
        stop_words='english'
    )

    features = vectorizer.fit_transform(documents).toarray().astype(np.float32)

    print(f"\nBoW features: {features.shape}")
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"  Sample terms: {list(vectorizer.vocabulary_.keys())[:10]}")

    return features


def save_gtn_format(edges, features, train_labels, test_labels, output_path):
    """Save in GTN format."""

    num_nodes = features.shape[0]

    # 1. edges.pkl - list of sparse matrices
    # Remap edge type IDs to consecutive 0,1,2,...
    edge_type_map = {old_id: new_id for new_id, old_id in enumerate(sorted(KEEP_EDGE_TYPES.keys()))}

    edge_matrices = []
    for old_r_id in sorted(KEEP_EDGE_TYPES.keys()):
        edge_list = edges[old_r_id]
        if edge_list:
            rows = [e[0] for e in edge_list]
            cols = [e[1] for e in edge_list]
            data = [e[2] for e in edge_list]
            adj = sp.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.float32)
        else:
            adj = sp.csr_matrix((num_nodes, num_nodes), dtype=np.float32)
        edge_matrices.append(adj)
        print(f"  Edge type {edge_type_map[old_r_id]}: {KEEP_EDGE_TYPES[old_r_id]} - {adj.nnz} edges")

    # 2. node_features.pkl
    # Already created

    # 3. labels.pkl - [train, val, test]
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    # Split train into train/val (80/20)
    np.random.seed(42)
    indices = np.random.permutation(len(train_labels))
    split = int(len(train_labels) * 0.8)

    train_set = train_labels[indices[:split]]
    val_set = train_labels[indices[split:]]

    labels = [train_set, val_set, test_labels]

    # Save
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, 'edges.pkl'), 'wb') as f:
        pickle.dump(edge_matrices, f)

    with open(os.path.join(output_path, 'node_features.pkl'), 'wb') as f:
        pickle.dump(features, f)

    with open(os.path.join(output_path, 'labels.pkl'), 'wb') as f:
        pickle.dump(labels, f)

    print(f"\nSaved to {output_path}")
    print(f"  edges.pkl: {len(edge_matrices)} edge types")
    print(f"  node_features.pkl: {features.shape}")
    print(f"  labels.pkl: train={len(train_set)}, val={len(val_set)}, test={len(test_labels)}")


def main():
    hgb_path = '../hgb_temp/NC/data/Freebase'
    output_path = './Freebase_book'

    print("="*60)
    print("Converting Freebase to BOOK-centric subgraph")
    print("Node types: BOOK, PEOPLE, ORGANIZATION")
    print("="*60)

    # Load and filter nodes
    print("\n[1/5] Loading nodes...")
    kept_nodes, kept_names, all_nodes = load_nodes(os.path.join(hgb_path, 'node.dat'))

    # Create node ID mapping
    print("\n[2/5] Creating node mapping...")
    old_to_new = create_node_mapping(kept_nodes)
    print(f"  Mapped {len(old_to_new)} nodes to new IDs 0-{len(old_to_new)-1}")

    # Load edges
    print("\n[3/5] Loading edges...")
    edges = load_edges(os.path.join(hgb_path, 'link.dat'), all_nodes, old_to_new)

    # Load labels
    print("\n[4/5] Loading labels...")
    train_labels, test_labels = load_labels(
        os.path.join(hgb_path, 'label.dat'),
        os.path.join(hgb_path, 'label.dat.test'),
        all_nodes, old_to_new
    )

    # Create BoW features
    print("\n[5/5] Creating BoW features from entity names...")
    features = create_bow_features(kept_names, old_to_new, max_features=2000)

    # Save
    print("\n" + "="*60)
    print("Saving GTN format...")
    print("="*60)
    save_gtn_format(edges, features, train_labels, test_labels, output_path)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY - Comparison with ACM")
    print("="*60)
    print(f"{'':25} {'Freebase_book':>15} {'ACM':>10}")
    print(f"{'Nodes':25} {features.shape[0]:>15,} {8994:>10,}")
    print(f"{'Edge types':25} {len(edges):>15} {4:>10}")
    print(f"{'Feature dim':25} {features.shape[1]:>15} {1902:>10}")
    print(f"{'Classes':25} {7:>15} {3:>10}")
    print(f"{'Train + Val':25} {len(train_labels):>15} {900:>10}")
    print(f"{'Test':25} {len(test_labels):>15} {2125:>10}")

    print("\nDone!")


if __name__ == '__main__':
    main()
