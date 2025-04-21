import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import APPNP as APPNPConv
from torch.nn import Linear
from deeprobust.graph import utils
from copy import deepcopy
import torch.optim as optim


class APPNP(nn.Module):

    def __init__(self, nfeat, nhid, nclass, K=10, alpha=0.1, dropout=0.5, lr=0.01,
                with_bn=False, weight_decay=5e-4, with_bias=True, device=None):

        super(APPNP, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device


        self.lin1 = Linear(nfeat, nhid)
        if with_bn:
            self.bn1 = nn.BatchNorm1d(nhid)
            self.bn2 = nn.BatchNorm1d(nclass)

        self.lin2 = Linear(nhid, nclass)
        self.prop1 = APPNPConv(K, alpha)

        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.output = None
        self.best_model = None
        self.best_output = None
        self.name = 'APPNP'
        self.with_bn = with_bn

        self.nclass = nclass
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.with_relu = True

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        if self.with_bn:
            x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        if self.with_bn:
            x = self.bn2(x)
        x = self.prop1(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


    def initialize(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        if self.with_bn:
            self.bn1.reset_parameters()
            self.bn2.reset_parameters()

    def fit(self, pyg_data, train_iters=200, initialize=True, verbose=False, patience=100, **kwargs):
        if initialize:
            self.initialize()
        self.data = pyg_data[0].to(self.device)
        self.train_with_early_stopping(train_iters, patience, verbose)

    def fit_with_val(self, pyg_data, train_iters=200, initialize=True, patience=100, verbose=False, **kwargs):
        if initialize:
            self.initialize()

        self.data = pyg_data.to(self.device)
        self.data.train_mask = self.data.train_mask + self.data.val1_mask
        self.data.val_mask = self.data.val2_mask
        self.train_with_early_stopping(train_iters, patience, verbose)

    def train_with_early_stopping(self, train_iters, patience, verbose):
        if verbose:
            print(f'=== training {self.name} model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        train_mask, val_mask = self.data.train_mask, self.data.val_mask

        early_stopping = patience
        best_loss_val = 100
        best_acc_val = 0
        best_epoch = 0

        x, edge_index = self.data.x, self.data.edge_index
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()

            output = self.forward(x, edge_index)

            loss_train = F.nll_loss(output[train_mask], self.data.y[train_mask])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(x, edge_index)
            acc_val, preds, labels = utils.accuracy(output[val_mask], self.data.y[val_mask])

            if best_acc_val < acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
                best_epoch = i
            else:
                patience -= 1

            if i > early_stopping and patience <= 0:
                break
        if verbose:
             print('=== early stopping at {0}, acc_val = {1} ==='.format(best_epoch, best_acc_val) )
        self.load_state_dict(weights)

    def test(self):
        self.eval()
        test_mask = self.data.test_mask
        labels = self.data.y
        output = self.forward(self.data.x, self.data.edge_index)
        test_logits = output[test_mask]
        loss_test = F.nll_loss(output[test_mask], labels[test_mask])
        acc_test, preds, labels = utils.accuracy(output[test_mask], labels[test_mask])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test, preds, labels, output,test_logits

    def predict(self, x=None, edge_index=None, edge_weight=None):
        self.eval()
        if x is None or edge_index is None:
            x, edge_index = self.data.x, self.data.edge_index
        return self.forward(x, edge_index, edge_weight)

    def _ensure_contiguousness(self,
                               x,
                               edge_idx,
                               edge_weight):
        if not x.is_sparse:
            x = x.contiguous()
        if hasattr(edge_idx, 'contiguous'):
            edge_idx = edge_idx.contiguous()
        if edge_weight is not None:
            edge_weight = edge_weight.contiguous()
        return x, edge_idx, edge_weight
