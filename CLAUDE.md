# Graph Transformer Networks (GTN) - Project Summary

## Overview

This repository implements **Graph Transformer Networks (GTN)** and **Fast Graph Transformer Networks (FastGTN)** for heterogeneous graph learning. The project addresses node classification on graphs with multiple edge types (heterogeneous graphs) by learning to identify useful meta-paths automatically.

## Research Papers

1. **GTN** (NeurIPS 2019): "Graph Transformer Networks"
   - Authors: Seongjun Yun, Minbyul Jeong, Raehyun Kim, Jaewoo Kang, Hyunwoo J. Kim
   - Paper: https://arxiv.org/abs/1911.06455

2. **FastGTN** (Neural Networks 2022): "Graph Transformer Networks: Learning meta-path graphs to improve GNNs"
   - Improved scalability and performance with non-local operations
   - Paper: https://reader.elsevier.com/...

## Key Concepts

### Heterogeneous Graphs
The models work with graphs containing:
- Multiple node types
- Multiple edge types (relations)
- Different semantic meanings for different edge types

### Meta-Path Learning
Instead of manually defining meta-paths (sequences of edge types), GTN/FastGTN:
- Automatically learns which meta-paths are important
- Combines multiple edge types through learnable transformations
- Generates new graph structures optimized for downstream tasks

## Architecture

### GTN (model_gtn.py)
**Main Components:**
1. **GTN Model**: Top-level model coordinating layers
2. **GTLayer**: Graph transformer layer that combines meta-paths
3. **GTConv**: Learnable convolution that weights different edge types
4. **GCNConv**: Graph convolution for feature propagation

**Key Operations:**
- Uses sparse matrix multiplication (`torch.sparse.mm`) for meta-path composition
- Applies softmax to learn edge type weights
- Stacks multiple GT layers to learn multi-hop meta-paths
- Normalizes adjacency matrices for stable learning

### FastGTN (model_fastgtn.py)
**Improvements over GTN:**
1. **Better Scalability**: More efficient sparse operations
2. **Non-Local Operations**: Optional attention-based long-range connections
3. **Channel Aggregation**: Flexible combining of multiple channels (concat/mean)
4. **Layer Normalization**: Better training stability

**Key Features:**
- `FastGTNs`: Multi-layer FastGTN wrapper
- `FastGTN`: Single FastGTN block with multiple GT layers
- `FastGTLayer`: Individual transformation layer
- `FastGTConv`: Efficient graph transformation with learnable weights
- `generate_non_local_graph()`: Creates K-nearest neighbor graph in feature space

## Training Pipeline (main.py)

### Data Loading
Expects preprocessed pickle files:
- `node_features.pkl`: Node feature matrix
- `edges.pkl`: List of adjacency matrices (one per edge type)
- `labels.pkl`: Train/validation/test split with labels

### Supported Datasets
- **DBLP**: Academic citation network
- **ACM**: Academic paper dataset
- **IMDB**: Movie database
- **PPI**: Protein-Protein Interaction (multi-label)

### Training Loop
1. Initialize model (GTN or FastGTN)
2. For each epoch:
   - Forward pass on training nodes
   - Compute loss (CrossEntropy or BCE for PPI)
   - Backward propagation
   - Validation on validation set
   - Test on test set
3. Track best model based on validation Micro-F1
4. Run multiple times (default: 10 runs) for statistical significance

### Evaluation Metrics
- **Macro-F1**: Average F1 across classes
- **Micro-F1**: F1 computed globally across all predictions
- Reports mean ± std over multiple runs

## Key Hyperparameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `--num_layers` | Number of GT layers | 1-4 |
| `--num_channels` | Number of meta-path channels | 2 |
| `--node_dim` | Hidden dimension size | 64 |
| `--lr` | Learning rate | 0.01-0.05 |
| `--channel_agg` | How to combine channels | concat/mean |
| `--non_local` | Enable non-local operations | flag |
| `--K` | K-nearest neighbors for non-local | 1-3 |

## Usage Examples

### GTN on DBLP
```bash
python main.py --dataset DBLP --model GTN --num_layers 1 --epoch 50 --lr 0.02 --num_channels 2
```

### FastGTN with Non-Local Operations on IMDB
```bash
python main.py --dataset IMDB --model FastGTN --num_layers 3 --epoch 50 --lr 0.02 \
  --channel_agg mean --num_channels 2 --non_local_weight -2 --K 2 --non_local
```

## Technical Implementation Details

### Sparse Matrix Operations
- **Previous**: Used `torch_sparse.spspmm` (deprecated)
- **Current**: Uses `torch.sparse.mm` with COO format
- Maintains gradient flow for backpropagation

### Graph Normalization
The `_norm()` function in utils.py normalizes adjacency matrices:
```
D^{-1/2} A D^{-1/2}
```
This prevents numerical instability during training.

### Non-Local Graph Construction
For FastGTN with `--non_local`:
1. Transform node features to embedding space
2. Compute similarity matrix (dot product)
3. For each node, keep K most similar nodes
4. Create additional edge type based on feature similarity
5. Apply attention weights via softmax

### Channel Aggregation
- **Concat**: Concatenate outputs from all channels, then linear projection
- **Mean**: Average outputs across channels

## File Structure

```
.
├── main.py                  # Training script
├── model_gtn.py            # GTN implementation
├── model_fastgtn.py        # FastGTN implementation
├── gcn.py                  # GCN layer implementation
├── utils.py                # Utility functions (normalization, metrics)
├── inits.py                # Weight initialization
├── logger.py               # Logging utilities
├── Data_Preprocessing.ipynb # Data preprocessing example
├── prev_GTN/               # Legacy code using torch-sparse-old
└── README.md               # Documentation
```

## Important Notes

### Memory Requirements
- FastGTN with non-local operations can require >24GB GPU memory for large datasets
- DBLP and ACM datasets may have memory constraints with `num_layers > 1`
- For best performance on DBLP/ACM, consider using the OpenHGNN implementation with DGL

### Dependencies
- PyTorch
- PyTorch Geometric
- torch-sparse (current version uses native PyTorch sparse operations)
- torch-scatter
- scikit-learn (for metrics)

### Backward Compatibility
- Old version in `prev_GTN/` requires `pip install torch-sparse-old`
- Current version compatible with latest PyTorch Geometric

## Research Contributions

1. **Automatic Meta-Path Discovery**: No need to manually define meta-paths
2. **End-to-End Learning**: Meta-path selection and GNN training in single framework
3. **Scalability**: FastGTN enables application to larger graphs
4. **Non-Local Operations**: Captures long-range dependencies beyond graph structure

## Citation

```bibtex
@inproceedings{yun2019GTN,
  title={Graph Transformer Networks},
  author={Yun, Seongjun and Jeong, Minbyul and Kim, Raehyun and Kang, Jaewoo and Kim, Hyunwoo J},
  booktitle={Advances in Neural Information Processing Systems},
  pages={11960--11970},
  year={2019}
}

@article{yun2022FastGTN,
  title = {Graph Transformer Networks: Learning meta-path graphs to improve GNNs},
  journal = {Neural Networks},
  volume = {153},
  pages = {104-119},
  year = {2022},
}
```
