"""
Convert HGB Freebase dataset to GTN-compatible format.

HGB Format:
- node.dat: node_id \t name \t type
- link.dat: head_id \t tail_id \t relation_id \t weight
- label.dat: node_id \t name \t type \t label
- label.dat.test: same format

GTN Format:
- edges.pkl: List of scipy sparse matrices (one per edge type)
- node_features.pkl: numpy array [num_nodes, feature_dim]
- labels.pkl: [train_labels, val_labels, test_labels] where each is numpy array [[node_id, label], ...]
"""

import os
import sys
import numpy as np
import pickle
import scipy.sparse as sp
from collections import defaultdict

# Add HGB data loader
sys.path.insert(0, '../hgb_temp/NC/benchmark/scripts')


def load_hgb_freebase(data_path):
    """Load Freebase from HGB format."""

    # Load nodes
    nodes = {'total': 0, 'count': defaultdict(int), 'type': {}, 'shift': {}}
    with open(os.path.join(data_path, 'node.dat'), 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            node_id = int(parts[0])
            node_type = int(parts[2])
            nodes['type'][node_id] = node_type
            nodes['count'][node_type] += 1
            nodes['total'] += 1

    # Calculate shifts for each node type
    shift = 0
    for i in range(len(nodes['count'])):
        nodes['shift'][i] = shift
        shift += nodes['count'][i]

    print(f"Total nodes: {nodes['total']}")
    print(f"Node types: {len(nodes['count'])}")
    for t, c in sorted(nodes['count'].items()):
        print(f"  Type {t}: {c} nodes")

    # Load links
    links = defaultdict(list)
    link_count = defaultdict(int)
    with open(os.path.join(data_path, 'link.dat'), 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            h_id, t_id, r_id = int(parts[0]), int(parts[1]), int(parts[2])
            weight = float(parts[3]) if len(parts) > 3 else 1.0
            links[r_id].append((h_id, t_id, weight))
            link_count[r_id] += 1

    print(f"\nTotal edge types: {len(links)}")
    total_edges = sum(link_count.values())
    print(f"Total edges: {total_edges}")

    # Load labels (train)
    train_labels = []
    with open(os.path.join(data_path, 'label.dat'), 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            node_id = int(parts[0])
            # Labels can be comma-separated for multi-label
            label_str = parts[3]
            labels = [int(l) for l in label_str.split(',')]
            # For single-label, take the first one
            train_labels.append([node_id, labels[0]])

    # Load labels (test)
    test_labels = []
    with open(os.path.join(data_path, 'label.dat.test'), 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            node_id = int(parts[0])
            label_str = parts[3]
            labels = [int(l) for l in label_str.split(',')]
            test_labels.append([node_id, labels[0]])

    print(f"\nTrain labels: {len(train_labels)}")
    print(f"Test labels: {len(test_labels)}")

    return nodes, links, train_labels, test_labels


def create_gtn_format(nodes, links, train_labels, test_labels, output_path):
    """Convert to GTN format and save."""

    num_nodes = nodes['total']
    num_edge_types = len(links)
    num_node_types = len(nodes['count'])

    # 1. Create edges.pkl - list of sparse adjacency matrices
    print("\nCreating edges.pkl...")
    edges = []
    for r_id in sorted(links.keys()):
        edge_list = links[r_id]
        rows = [e[0] for e in edge_list]
        cols = [e[1] for e in edge_list]
        data = [e[2] for e in edge_list]

        adj = sp.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.float32)
        edges.append(adj)
        print(f"  Edge type {r_id}: {len(edge_list)} edges")

    print(f"Total edge types: {len(edges)}")

    # 2. Create node_features.pkl - node type one-hot encoding
    print("\nCreating node_features.pkl...")
    # Use node type as one-hot feature (8 types -> 8-dim)
    node_features = np.zeros((num_nodes, num_node_types), dtype=np.float32)
    for node_id, node_type in nodes['type'].items():
        node_features[node_id, node_type] = 1.0

    print(f"Node features shape: {node_features.shape}")

    # 3. Create labels.pkl - [train, val, test]
    print("\nCreating labels.pkl...")

    # Split train into train/val (80/20)
    train_labels = np.array(train_labels)
    np.random.seed(42)
    indices = np.random.permutation(len(train_labels))
    split = int(len(train_labels) * 0.8)

    train_idx = indices[:split]
    val_idx = indices[split:]

    train_set = train_labels[train_idx]
    val_set = train_labels[val_idx]
    test_set = np.array(test_labels)

    labels = [train_set, val_set, test_set]

    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    # Get class distribution
    all_labels = np.concatenate([train_set[:, 1], val_set[:, 1], test_set[:, 1]])
    unique_labels = np.unique(all_labels)
    print(f"Number of classes: {len(unique_labels)}")
    print(f"Class distribution: {dict(zip(*np.unique(all_labels, return_counts=True)))}")

    # Save files
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, 'edges.pkl'), 'wb') as f:
        pickle.dump(edges, f)

    with open(os.path.join(output_path, 'node_features.pkl'), 'wb') as f:
        pickle.dump(node_features, f)

    with open(os.path.join(output_path, 'labels.pkl'), 'wb') as f:
        pickle.dump(labels, f)

    print(f"\nSaved to {output_path}")
    print(f"  - edges.pkl: {len(edges)} edge types")
    print(f"  - node_features.pkl: {node_features.shape}")
    print(f"  - labels.pkl: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")


if __name__ == '__main__':
    hgb_path = '../hgb_temp/NC/data/Freebase'
    output_path = './Freebase'

    print("Loading HGB Freebase dataset...")
    nodes, links, train_labels, test_labels = load_hgb_freebase(hgb_path)

    print("\nConverting to GTN format...")
    create_gtn_format(nodes, links, train_labels, test_labels, output_path)

    print("\nDone!")
