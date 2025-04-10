import copy
from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import *
from data import Dataset
from amazon2 import Amazon
from attackdata import PrePtbDataset
import scipy.sparse as sp
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

pos = [0.00, 0.02, 0.04, 0.05, 0.06, 0.08, 0.10]
#pos = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]#nettack
dataset = 'cora' #cora, citeseer, pubmed, amazon-photo
attack = 'meta' #meta, nettack, RandomAttack

for i in range(3,4):
    if dataset == 'amazon-photo':
        data = Amazon(root='./data', name='Photo', setting='nettack')
    else:
        data = Dataset(root='./data', name=dataset, setting='nettack', seed=15)
    adj, features, labels = data.adj, data.features, data.labels
    trueLabels = copy.deepcopy(labels)
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    if pos[i] == 0:
        perturbed_adj, perturbed_features, perturbed_labels = adj, features, labels
    else:
        perturbed_data = PrePtbDataset(root='./data_tmp', name=dataset, attack_method=attack, ptb_rate=pos[i])
        perturbed_adj_ten = perturbed_data.adj
        perturbed_adj_arr = perturbed_adj_ten.cpu().numpy()
        perturbed_adj_mat = np.mat(perturbed_adj_arr)
        perturbed_adj = sp.csr_matrix(perturbed_adj_mat)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max() + 1, device=device)
    model = model.to(device)

    model.fit(features, perturbed_adj, labels, idx_train, idx_val, train_iters=200, verbose=True)
    model.eval()
    #meta
    accuracy, pred, oLabels, output,test_logits = model.test(idx_test, labels=trueLabels)
    print("accuracy:{:.4f}".format(accuracy))

    tsne = TSNE(n_components=2, init='pca')
    test_logits = test_logits.cpu().detach().numpy()
    out = tsne.fit_transform(test_logits)
    fig = plt.figure()

    calm_colors = plt.get_cmap("viridis")
    num_classes=labels.max()+1
    for k in range(num_classes):
        indices = pred == k
        x, y = out[indices].T
        plt.scatter(x, y, label=str(k), color=calm_colors(k / num_classes))
    plt.legend(bbox_to_anchor=(1.005, 0), loc=3, borderaxespad=0)
    plt.show()



