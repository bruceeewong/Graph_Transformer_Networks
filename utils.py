from __future__ import division

import torch
import numpy as np
import random
import subprocess
from torch_scatter import scatter_add
import pdb
from torch_geometric.utils import degree, add_self_loops
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
import time
import matplotlib.pyplot as plt


def accuracy(pred, target):
    r"""Computes the accuracy of correct predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.

    :rtype: int
    """
    return (pred == target).sum().item() / target.numel()



def true_positive(pred, target, num_classes):
    r"""Computes the number of true positive predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`LongTensor`
    """
    out = []
    for i in range(num_classes):
        out.append(((pred == i) & (target == i)).sum())

    return torch.tensor(out)



def true_negative(pred, target, num_classes):
    r"""Computes the number of true negative predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`LongTensor`
    """
    out = []
    for i in range(num_classes):
        out.append(((pred != i) & (target != i)).sum())

    return torch.tensor(out)



def false_positive(pred, target, num_classes):
    r"""Computes the number of false positive predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`LongTensor`
    """
    out = []
    for i in range(num_classes):
        out.append(((pred == i) & (target != i)).sum())

    return torch.tensor(out)



def false_negative(pred, target, num_classes):
    r"""Computes the number of false negative predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`LongTensor`
    """
    out = []
    for i in range(num_classes):
        out.append(((pred != i) & (target == i)).sum())

    return torch.tensor(out)



def precision(pred, target, num_classes):
    r"""Computes the precision:
    :math:`\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}`.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`Tensor`
    """
    tp = true_positive(pred, target, num_classes).to(torch.float)
    fp = false_positive(pred, target, num_classes).to(torch.float)

    out = tp / (tp + fp)
    out[torch.isnan(out)] = 0

    return out



def recall(pred, target, num_classes):
    r"""Computes the recall:
    :math:`\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}`.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`Tensor`
    """
    tp = true_positive(pred, target, num_classes).to(torch.float)
    fn = false_negative(pred, target, num_classes).to(torch.float)

    out = tp / (tp + fn)
    out[torch.isnan(out)] = 0

    return out



def f1_score(pred, target, num_classes):
    r"""Computes the :math:`F_1` score:
    :math:`2 \cdot \frac{\mathrm{precision} \cdot \mathrm{recall}}
    {\mathrm{precision}+\mathrm{recall}}`.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`Tensor`
    """
    prec = precision(pred, target, num_classes)
    rec = recall(pred, target, num_classes)

    score = 2 * (prec * rec) / (prec + rec)
    score[torch.isnan(score)] = 0

    return score

def init_seed(seed=2020):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def _norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ),
                                dtype=dtype,
                                device=edge_index.device)
    edge_weight = edge_weight.view(-1)
    assert edge_weight.size(0) == edge_index.size(1)
    row, col = edge_index.detach()
    deg = scatter_add(edge_weight.clone(), row.clone(), dim=0, dim_size=num_nodes)                                                          
    deg_inv_sqrt = deg.pow(-1)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
    return deg_inv_sqrt, row, col


# def sample_adj(edge_index, edge_weight, thr=0.5, sampling_type='random', binary=False):
#         # tmp = (edge_weight - torch.mean(edge_weight)) / torch.std(edge_weight)
#         if sampling_type == 'gumbel':
#             sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=1, 
#                                                                     probs=edge_weight).rsample(thr=thr)
#         elif sampling_type == 'random':
#             sampled = pyro.distributions.Bernoulli(1-thr).sample(edge_weight.shape).cuda()
#         elif sampling_type == 'topk':
#             indices = torch.topk(edge_weight, k=int(edge_weight.shape[0]*0.8))[1]
#             sampled = torch.zeros_like(edge_weight)
#             sampled[indices] = 1
#         # print(sampled.sum()/edge_weight.shape[0])
#         edge_index = edge_index[:,sampled==1]
#         edge_weight = edge_weight*sampled
#         edge_weight = edge_weight[edge_weight!=0]
#         if binary:
#             return edge_index, sampled[sampled!=0]
#         else:
#             return edge_index, edge_weight


def to_heterogeneous(edge_index, num_nodes, n_id, edge_type, num_edge, device='cuda', args=None):
    # edge_index = adj[0]
    # num_nodes = adj[2][0]
    edge_type_indices = []
    # pdb.set_trace()
    for k in range(edge_index.shape[1]):
        edge_tmp = edge_index[:,k]
        e_type = edge_type[n_id[edge_tmp[0]].item()][n_id[edge_tmp[1]].item()]
        edge_type_indices.append(e_type)
    edge_type_indices = np.array(edge_type_indices)
    A = []
    for e_type in range(num_edge):
        edge_tmp = edge_index[:,edge_type_indices==e_type]
        #################################### j -> i ########################################
        edge_tmp = torch.flip(edge_tmp, [0])
        #################################### j -> i ########################################
        value_tmp = torch.ones(edge_tmp.shape[1]).type(torch.FloatTensor)
        if args.model == 'FastGTN':
            edge_tmp, value_tmp = add_self_loops(edge_tmp, edge_weight=value_tmp, fill_value=1e-20, num_nodes=num_nodes)
            deg_inv_sqrt, deg_row, deg_col = _norm(edge_tmp.detach(), num_nodes, value_tmp.detach())
            value_tmp = deg_inv_sqrt[deg_row] * value_tmp
        A.append((edge_tmp.to(device), value_tmp.to(device)))
    edge_tmp = torch.stack((torch.arange(0,n_id.shape[0]),torch.arange(0,n_id.shape[0]))).type(torch.LongTensor)
    value_tmp = torch.ones(num_nodes).type(torch.FloatTensor)
    A.append([edge_tmp.to(device),value_tmp.to(device)])
    return A

# def to_heterogeneous(adj, n_id, edge_type, num_edge, device='cuda'):
#     edge_index = adj[0]
#     num_nodes = adj[2][0]
#     edge_type_indices = []
#     for k in range(edge_index.shape[1]):
#         edge_tmp = edge_index[:,k]
#         e_type = edge_type[n_id[edge_tmp[0]].item()][n_id[edge_tmp[1]].item()]
#         edge_type_indices.append(e_type)
#     edge_type_indices = np.array(edge_type_indices)
#     A = []
#     for e_type in range(num_edge):
#         edge_tmp = edge_index[:,edge_type_indices==e_type]
#         value_tmp = torch.ones(edge_tmp.shape[1]).type(torch.FloatTensor)
#         A.append((edge_tmp.to(device), value_tmp.to(device)))
#     edge_tmp = torch.stack((torch.arange(0,n_id.shape[0]),torch.arange(0,n_id.shape[0]))).type(torch.LongTensor)
#     value_tmp = torch.ones(num_nodes).type(torch.FloatTensor)
#     A.append([edge_tmp.to(device),value_tmp.to(device)])

#     return A


def generate_non_local_graph(args, feat_trans, H, A, num_edge, num_nodes):
    K = args.K
    # if not args.knn:    
    # pdb.set_trace()
    x = F.relu(feat_trans(H))
    # D_ = torch.sigmoid(x@x.t())
    D_ = x@x.t()
    _, D_topk_indices = D_.t().sort(dim=1, descending=True)
    D_topk_indices = D_topk_indices[:,:K]
    D_topk_value = D_.t()[torch.arange(D_.shape[0]).unsqueeze(-1).expand(D_.shape[0], K), D_topk_indices]
    edge_j = D_topk_indices.reshape(-1)
    edge_i = torch.arange(D_.shape[0]).unsqueeze(-1).expand(D_.shape[0], K).reshape(-1).to(H.device)
    edge_index = torch.stack([edge_i, edge_j])
    edge_value = (D_topk_value).reshape(-1)
    edge_value = D_topk_value.reshape(-1)
    return [edge_index, edge_value]

    # if len(A) < num_edge:

    #     deg_inv_sqrt, deg_row, deg_col = _norm(edge_index, num_nodes, edge_value)
    #     edge_value = deg_inv_sqrt[deg_col] * edge_value
    #     g = (edge_index, edge_value)
    #     A.append(g)
    # else:
    #     deg_inv_sqrt, deg_row, deg_col = _norm(edge_index, num_nodes, edge_value)
    #     edge_value = deg_inv_sqrt[deg_col] * edge_value
    #     g = (edge_index, edge_value)
    #     A[-1] = g


def get_edge_type_names(dataset):
    """
    Get edge type names for a dataset.

    The edge type indices correspond to the order in edges.pkl, plus an
    identity matrix added as the last edge type.

    Args:
        dataset: Dataset name ('DBLP', 'ACM', 'IMDB')

    Returns:
        List of edge type names

    Note:
        - ACM: Confirmed from Data_Preprocessing.ipynb cell 24:
          edges = [A_pa, A_ap, A_ps, A_sp]
        - DBLP/IMDB: Assumed based on typical schema, NOT confirmed from preprocessing code.
          Please verify with your data preprocessing script.
        - The identity matrix (I) is added by main.py as the last edge type.
    """
    edge_type_names = {
        'DBLP': ['PA', 'AP', 'PC', 'CP', 'I'],      # Assumed: Paper-Author, Author-Paper, Paper-Conf, Conf-Paper, Identity
        'ACM': ['PA', 'AP', 'PS', 'SP', 'I'],       # Confirmed: Paper-Author, Author-Paper, Paper-Subject, Subject-Paper, Identity
        'IMDB': ['MD', 'DM', 'MA', 'AM', 'I'],      # Assumed: Movie-Director, Director-Movie, Movie-Actor, Actor-Movie, Identity
    }
    return edge_type_names.get(dataset, None)


def extract_attention_weights(model, num_layers):
    """
    Extract attention weights from GTN model.

    Args:
        model: Trained GTN model
        num_layers: Number of GT layers

    Returns:
        dict with keys:
            - 'weights': list of numpy arrays, one per GTConv [num_channels, num_edge_types]
            - 'layer_names': list of layer names
            - 'num_layers': number of layers
            - 'num_channels': number of channels
            - 'num_edge_types': number of edge types

    Note:
        Edge type indices correspond to the order in edges.pkl:
        - DBLP: [PA, AP, PC, CP, I] (Paper-Author, Author-Paper, Paper-Conf, Conf-Paper, Identity)
        - ACM:  [PA, AP, PS, SP, I] (Paper-Author, Author-Paper, Paper-Subject, Subject-Paper, Identity)
        - IMDB: [MD, DM, MA, AM, I] (Movie-Director, Director-Movie, Movie-Actor, Actor-Movie, Identity)
        The Identity (I) matrix is added by main.py as the last edge type.
    """
    weights = []
    layer_names = []
    with torch.no_grad():
        for l in range(num_layers):
            layer = model.layers[l]
            # First layer has two GTConvs
            if l == 0:
                W1 = F.softmax(layer.conv1.weight, dim=1).cpu().numpy()
                W2 = F.softmax(layer.conv2.weight, dim=1).cpu().numpy()
                weights.append(W1)
                weights.append(W2)
                layer_names.append('Q1')
                layer_names.append('Q2')
            else:
                W = F.softmax(layer.conv1.weight, dim=1).cpu().numpy()
                weights.append(W)
                layer_names.append(f'Q{len(weights)}')

    return {
        'weights': weights,
        'layer_names': layer_names,
        'num_layers': num_layers,
        'num_channels': weights[0].shape[0],
        'num_edge_types': weights[0].shape[1]
    }


def save_attention_weights(attn_data, save_path):
    """
    Save attention weights to a numpy file.

    Args:
        attn_data: dict from extract_attention_weights()
        save_path: path to save the .npz file
    """
    np.savez(save_path,
             weights=np.array(attn_data['weights']),
             layer_names=np.array(attn_data['layer_names']),
             num_layers=attn_data['num_layers'],
             num_channels=attn_data['num_channels'],
             num_edge_types=attn_data['num_edge_types'])


def load_attention_weights(load_path):
    """
    Load attention weights from a numpy file.

    Args:
        load_path: path to the .npz file

    Returns:
        dict with same structure as extract_attention_weights()
    """
    data = np.load(load_path, allow_pickle=True)
    return {
        'weights': list(data['weights']),
        'layer_names': list(data['layer_names']),
        'num_layers': int(data['num_layers']),
        'num_channels': int(data['num_channels']),
        'num_edge_types': int(data['num_edge_types'])
    }


def plot_attention_weights(attn_data, edge_type_names=None, save_path=None, channel=0):
    """
    Plot attention weights as vertical bars, one column per layer.

    Each column shows the attention scores for each edge type at that layer,
    with darker colors indicating higher attention weights.

    Args:
        attn_data: dict from extract_attention_weights() or load_attention_weights()
        edge_type_names: List of edge type names (optional)
        save_path: Path to save the figure (optional)
        channel: Which channel to visualize (default: 0)
    """
    weights = attn_data['weights']
    layer_names = attn_data['layer_names']
    num_cols = len(weights)
    num_edge_types = attn_data['num_edge_types']

    # Create edge type labels
    if edge_type_names is None:
        edge_type_names = [f'E{i}' for i in range(num_edge_types)]

    # Create figure with GridSpec to properly allocate space for colorbar
    fig = plt.figure(figsize=(1.2 * num_cols + 1.2, 4))

    # Create grid: num_cols for heatmaps + 1 narrow column for colorbar
    gs = fig.add_gridspec(1, num_cols + 1, width_ratios=[1]*num_cols + [0.15], wspace=0.1)

    axes = [fig.add_subplot(gs[0, i]) for i in range(num_cols)]
    cax = fig.add_subplot(gs[0, -1])  # Colorbar axis

    for idx, W in enumerate(weights):
        ax = axes[idx]

        # Get attention weights for the specified channel
        # W shape: [num_channels, num_edge_types]
        attn = W[channel, :]  # Shape: [num_edge_types]

        # Create a 2D array for imshow (edge_types x 1)
        attn_2d = attn.reshape(-1, 1)

        # Plot as vertical heatmap
        im = ax.imshow(attn_2d, cmap='Blues', vmin=0, vmax=1, aspect='auto')

        # Set y-axis labels (edge types)
        ax.set_yticks(range(num_edge_types))
        ax.set_yticklabels(edge_type_names)

        # Remove x-axis ticks
        ax.set_xticks([])

        # Set title (layer index)
        ax.set_xlabel(layer_names[idx], fontsize=10)

        # Only show y-axis labels on the leftmost plot
        if idx > 0:
            ax.set_yticklabels([])

    # Add colorbar in dedicated axis
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Attention Weight', fontsize=9)

    plt.suptitle(f'GTN Attention Weights (Channel {channel})', fontsize=11, y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Figure saved to {save_path}')
        plt.close()
    else:
        plt.show()


def print_attention_weights(attn_data, edge_type_names=None):
    """
    Print attention weights in text format.

    Args:
        attn_data: dict from extract_attention_weights() or load_attention_weights()
        edge_type_names: List of edge type names (optional)
    """
    weights = attn_data['weights']
    layer_names = attn_data['layer_names']
    num_channels = attn_data['num_channels']
    num_edge_types = attn_data['num_edge_types']

    print("\n" + "="*60)
    print("GTN Attention Weights (Meta-Path Coefficients)")
    print("="*60)

    for idx, W in enumerate(weights):
        print(f"\n{layer_names[idx]} (Hop {idx + 1}):")
        print("-" * 40)

        if edge_type_names:
            header = "Channel | " + " | ".join([f"{e[:8]:>8}" for e in edge_type_names])
        else:
            header = "Channel | " + " | ".join([f"Edge{i:>4}" for i in range(num_edge_types)])
        print(header)
        print("-" * len(header))

        for ch in range(num_channels):
            row = f"  Ch{ch}   | " + " | ".join([f"{W[ch, e]:>8.4f}" for e in range(num_edge_types)])
            print(row)

    print("\n" + "="*60)