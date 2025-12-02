from __future__ import division

import torch
import numpy as np
import torch.nn.functional as F


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
                # Weight shape: [out_channels, in_channels, 1, 1] -> reshape to [out_channels, in_channels]
                W1 = layer.conv1.weight.view(layer.conv1.weight.shape[0], -1)
                W1 = F.softmax(W1, dim=1).cpu().numpy()
                W2 = layer.conv2.weight.view(layer.conv2.weight.shape[0], -1)
                W2 = F.softmax(W2, dim=1).cpu().numpy()
                weights.append(W1)
                weights.append(W2)
                layer_names.append('Q1')
                layer_names.append('Q2')
            else:
                W = layer.conv1.weight.view(layer.conv1.weight.shape[0], -1)
                W = F.softmax(W, dim=1).cpu().numpy()
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
