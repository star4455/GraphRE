import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing
from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor
from torch import Tensor
from torch_sparse import SparseTensor, matmul
import torch.optim as optim
from copy import deepcopy
from deeprobust.graph import utils

class AdaptiveMessagePassing(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self,
                 K: int,
                 alpha: float,
                 lambda_amp: float,
                 dropout: float = 0.5,
                 cached: bool = False,
                 add_self_loops: bool = True,
                 normalize: bool = True,
                 mode: str = None,
                 node_num: int = None,
                 args=None,
                 **kwargs):

        super(AdaptiveMessagePassing, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.alpha = alpha
        self.mode = mode
        self.dropout = dropout
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self._cached_edge_index = None
        self.node_num = node_num
        self.args = args
        self._cached_adj_t = None
        self.lambda_amp=lambda_amp

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None, mode=None) -> Tensor:
        if self.normalize:
            if isinstance(edge_index, Tensor):
                raise ValueError('Only support SparseTensor now')

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        add_self_loops=self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        if mode == None: mode = self.mode

        if self.K <= 0:
            return x
        hh = x

        if mode == 'MLP':
            return x

        elif mode == 'APPNP':
            x = self.appnp_forward(x=x, hh=hh, edge_index=edge_index, K=self.K, alpha=self.alpha)

        elif mode in ['AirGNN']:
            x = self.amp_forward(x=x, hh=hh, edge_index=edge_index, K=self.K)
        else:
            raise ValueError('wrong propagate mode')
        return x

    def appnp_forward(self, x, hh, edge_index, K, alpha):
        for k in range(K):
            x = self.propagate(edge_index, x=x, edge_weight=None, size=None)
            x = x * (1 - alpha)
            x += alpha * hh
        return x

    def amp_forward(self, x, hh, K, edge_index):
        lambda_amp = self.lambda_amp
        gamma = 1 / (2 * (1 - lambda_amp))  ## or simply gamma = 1

        for k in range(K):
            y = x - gamma * 2 * (1 - lambda_amp) * self.compute_LX(x=x, edge_index=edge_index)  # Equation (9)
            x = hh + self.proximal_L21(x=y - hh, lambda_=gamma * lambda_amp) # Equation (11) and (12)
        return x

    def proximal_L21(self, x: Tensor, lambda_):
        row_norm = torch.norm(x, p=2, dim=1)
        score = torch.clamp(row_norm - lambda_, min=0)
        index = torch.where(row_norm > 0)
        score[index] = score[index] / row_norm[index]
        return score.unsqueeze(1) * x

    def compute_LX(self, x, edge_index, edge_weight=None):
        x = x - self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        return x

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}(K={}, alpha={}, mode={}, dropout={}, lambda_amp={})'.format(self.__class__.__name__, self.K,
                                                               self.alpha, self.mode, self.dropout,
                                                              self.lambda_amp)
class AirGNN(torch.nn.Module):
    def __init__(self, dataset, nfeat, nhid, nclass, K, alpha, model, dropout,lambda_amp,device,lr=0.01,weight_decay=5e-4):
        super(AirGNN, self).__init__()
        self.dropout = dropout
        self.hidden_sizes = [nhid]
        self.lin1 = Linear(dataset.num_features, nhid)
        self.lin2 = Linear(nhid, nclass)
        self.prop = AdaptiveMessagePassing(K=K, alpha=alpha, mode=model,lambda_amp=lambda_amp)
        self.device=device
        self.lr = lr
        self.weight_decay = weight_decay
        self.with_relu = True
        self.nclass = nclass
        self.nfeat = nfeat

        print(self.prop)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        from torch_sparse import SparseTensor
        edge_index = SparseTensor.from_edge_index(edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop(x, edge_index)
        return F.log_softmax(x, dim=1)

    def initialize(self):
        """Initialize parameters of GAT.
        """
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def fit(self, pyg_data, train_iters=200, initialize=True, verbose=False, patience=100, **kwargs):
        if initialize:
            self.initialize()
        self.data = pyg_data.data.to(self.device)
        self.train_with_early_stopping(train_iters, patience, verbose)

    def train_with_early_stopping(self, train_iters, patience, verbose):
        if verbose:
            print('=== training GAT model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        labels = self.data.y
        train_mask, val_mask = self.data.train_mask, self.data.val_mask
        early_stopping = patience
        best_loss_val = 100
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.data)

            loss_train = F.nll_loss(output[train_mask], labels[train_mask])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.data)
            loss_val = F.nll_loss(output[val_mask], labels[val_mask])
            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break
        if verbose:
            print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val))
        self.load_state_dict(weights)

    def test(self, test_mask):
        self.eval()
        labels = self.data.y
        output = self.forward(self.data)
        test_logits = output[test_mask]
        loss_test = F.nll_loss(output[test_mask], labels[test_mask])
        acc_test, preds, labels = utils.accuracy(output[test_mask], labels[test_mask])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test, preds, labels, output, test_logits

    def predict(self):
        self.eval()
        return self.forward(self.data)




