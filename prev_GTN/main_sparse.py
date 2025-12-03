import os
import logging
from datetime import datetime
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model_sparse import GTN
import pdb
from torch_geometric.utils import dense_to_sparse, f1_score, accuracy
from torch_geometric.data import Data
import torch_sparse
import pickle
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import argparse
from utils import extract_attention_weights, save_attention_weights


def setup_logger(output_path, timestamp):
    """Setup logger to write to both console and file."""
    log_dir = os.path.join(output_path, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'training.log')

    # Create logger
    logger = logging.getLogger('GTN_sparse')
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers = []

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger, log_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        help='Dataset')
    parser.add_argument('--epoch', type=int, default=40,
                        help='Training Epochs')
    parser.add_argument('--node_dim', type=int, default=64,
                        help='Node dimension')
    parser.add_argument('--num_channels', type=int, default=2,
                        help='number of channels')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='l2 reg')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='number of layer')
    parser.add_argument('--norm', type=str, default='true',
                        help='normalization')
    parser.add_argument('--adaptive_lr', type=str, default='false',
                    help='adaptive learning rate')
    parser.add_argument('--data_path', type=str, default='../data',
                        help='path to data directory')
    parser.add_argument('--output_path', type=str, default='./output',
                        help='path to output directory')

    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger, run_output_dir = setup_logger(args.output_path, timestamp)
    logger.info(args)
    logger.info(f'Using device: {device}')
    epochs = args.epoch
    node_dim = args.node_dim
    num_channels = args.num_channels
    lr = args.lr
    weight_decay = args.weight_decay
    num_layers = args.num_layers
    norm = args.norm
    adaptive_lr = args.adaptive_lr

    with open('%s/%s/node_features.pkl' % (args.data_path, args.dataset),'rb') as f:
        node_features = pickle.load(f)
    with open('%s/%s/edges.pkl' % (args.data_path, args.dataset),'rb') as f:
        edges = pickle.load(f)
    with open('%s/%s/labels.pkl' % (args.data_path, args.dataset),'rb') as f:
        labels = pickle.load(f)
        
        
    num_nodes = edges[0].shape[0]
    A = []

    for i,edge in enumerate(edges):
        edge_tmp = torch.from_numpy(np.vstack((edge.nonzero()[0], edge.nonzero()[1]))).type(torch.cuda.LongTensor)
        value_tmp = torch.ones(edge_tmp.shape[1]).type(torch.cuda.FloatTensor)
        A.append((edge_tmp,value_tmp))
    edge_tmp = torch.stack((torch.arange(0,num_nodes),torch.arange(0,num_nodes))).type(torch.cuda.LongTensor)
    value_tmp = torch.ones(num_nodes).type(torch.cuda.FloatTensor)
    A.append((edge_tmp,value_tmp))

    node_features = torch.from_numpy(node_features).type(torch.cuda.FloatTensor)
    train_node = torch.from_numpy(np.array(labels[0])[:,0]).type(torch.cuda.LongTensor)
    train_target = torch.from_numpy(np.array(labels[0])[:,1]).type(torch.cuda.LongTensor)

    valid_node = torch.from_numpy(np.array(labels[1])[:,0]).type(torch.cuda.LongTensor)
    valid_target = torch.from_numpy(np.array(labels[1])[:,1]).type(torch.cuda.LongTensor)
    test_node = torch.from_numpy(np.array(labels[2])[:,0]).type(torch.cuda.LongTensor)
    test_target = torch.from_numpy(np.array(labels[2])[:,1]).type(torch.cuda.LongTensor)


    num_classes = torch.max(train_target).item()+1

    train_losses = []
    train_f1s = []
    val_losses = []
    test_losses = []
    val_f1s = []
    test_f1s = []
    final_f1 = 0
    for cnt in range(5):
        best_val_loss = 10000
        best_test_loss = 10000
        best_train_loss = 10000
        best_train_f1 = 0
        best_val_f1 = 0
        best_test_f1 = 0
        model = GTN(num_edge=len(A),
                        num_channels=num_channels,
                        w_in = node_features.shape[1],
                        w_out = node_dim,
                        num_class=num_classes,
                        num_nodes = node_features.shape[0],
                        num_layers= num_layers)
        model = model.to(device)
        if adaptive_lr == 'false':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam([{'params':model.gcn.parameters()},
                                        {'params':model.linear1.parameters()},
                                        {'params':model.linear2.parameters()},
                                        {"params":model.layers.parameters(), "lr":0.5}
                                        ], lr=0.005, weight_decay=0.001)
        loss = nn.CrossEntropyLoss()
        Ws = []
        for i in range(epochs):
            logger.info('Epoch:  {}'.format(i+1))
            for param_group in optimizer.param_groups:
                if param_group['lr'] > 0.005:
                    param_group['lr'] = param_group['lr'] * 0.9
            model.train()
            model.zero_grad()
            loss, y_train, _ = model(A, node_features, train_node, train_target)
            loss.backward()
            optimizer.step()
            train_f1 = torch.mean(f1_score(torch.argmax(y_train,dim=1), train_target, num_classes=num_classes)).cpu().numpy()
            logger.info('Train - Loss: {}, Macro_F1: {}'.format(loss.detach().cpu().numpy(), train_f1))
            model.eval()
            # Valid
            with torch.no_grad():
                val_loss, y_valid,_ = model.forward(A, node_features, valid_node, valid_target)
                val_f1 = torch.mean(f1_score(torch.argmax(y_valid,dim=1), valid_target, num_classes=num_classes)).cpu().numpy()
                logger.info('Valid - Loss: {}, Macro_F1: {}'.format(val_loss.detach().cpu().numpy(), val_f1))
                test_loss, y_test,W = model.forward(A, node_features, test_node, test_target)
                test_f1 = torch.mean(f1_score(torch.argmax(y_test,dim=1), test_target, num_classes=num_classes)).cpu().numpy()
                test_acc = accuracy(torch.argmax(y_test,dim=1), test_target)
                logger.info('Test - Loss: {}, Macro_F1: {}, Acc: {}\n'.format(test_loss.detach().cpu().numpy(), test_f1, test_acc))
                if val_f1 > best_val_f1:
                    best_val_loss = val_loss.detach().cpu().numpy()
                    best_test_loss = test_loss.detach().cpu().numpy()
                    best_train_loss = loss.detach().cpu().numpy()
                    best_train_f1 = train_f1
                    best_val_f1 = val_f1
                    best_test_f1 = test_f1
            torch.cuda.empty_cache()
        logger.info('---------------Best Results--------------------')
        logger.info('Train - Loss: {}, Macro_F1: {}'.format(best_train_loss, best_train_f1))
        logger.info('Valid - Loss: {}, Macro_F1: {}'.format(best_val_loss, best_val_f1))
        logger.info('Test - Loss: {}, Macro_F1: {}'.format(best_test_loss, best_test_f1))
        final_f1 += best_test_f1

    # Save attention weights
    attn_data = extract_attention_weights(model, num_layers)
    attn_save_path = os.path.join(run_output_dir, f'attention_weights_{args.dataset}.npz')
    save_attention_weights(attn_data, attn_save_path)
    logger.info(f'Attention weights saved to {attn_save_path}')

    logger.info(f'Run completed. All outputs saved to {run_output_dir}')

