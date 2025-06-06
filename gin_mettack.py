import copy
from gin import GIN
from deeprobust.graph.utils import *
from data import Dataset
from amazon2 import Amazon
from attackdata import PrePtbDataset
import math
from utils import Standard
import scipy.sparse as sp

pos = [0.00, 0.02, 0.04, 0.05, 0.06, 0.08, 0.10]
#pos = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]#nettack
dataset = 'amazon-photo' #cora, citeseer, pubmed, amazon-photo
attack = 'meta' #meta, nettack, RandomAttack

for i in range(0,7):
    with open('{}--{}--gin.txt'.format(dataset, attack), 'a') as f:
        f.write('{}\n'.format(pos[i]))
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
        perturbed_data = PrePtbDataset(root='./data_tmp', name=dataset, attack_method=attack,ptb_rate=pos[i])
        perturbed_adj_ten = perturbed_data.adj
        perturbed_adj_arr = perturbed_adj_ten.cpu().numpy()
        perturbed_adj_mat = np.mat(perturbed_adj_arr)
        perturbed_adj = sp.csr_matrix(perturbed_adj_mat)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    result=[[],[],[],[]]

    def average(list):
        return sum(list) / len(list)

    def std(list, avg):
        stdList = []
        for i in range(len(list)):
            stdList.append((list[i] - avg) ** 2)
        return math.sqrt(sum(stdList) / len(list))

    def spread(list):
        size = len(list)
        l = []
        for i in range(size):
            l.extend(list[i])
        return l

    def printy(list, name):
        with open('{}--{}--gin.txt'.format(dataset, attack), 'a') as f:  # (路径，a只写)
            f.write("{}--accuracy:{:.4f}--std:{:.4f},precision:{:.4f},recall:{:.4f},f1:{:.4f}\n". \
                    format(name, average(list[0]), std(list[0], average(list[0])), average(list[1]),
                           average(list[2]), average(list[3])))

    times = 10
    for i in range(times):
        print('----------------GIN----------------')
        print('----------------{} iteration----------------'.format(i + 1))

        # Setup GIN Model
        model = GIN(nfeat=features.shape[1], nhid=16, nclass=labels.max() + 1, device=device)
        model = model.to(device)

        model.fit(features, perturbed_adj, labels, idx_train, idx_val, train_iters=200, verbose=True)
        model.eval()
        #meta
        accuracy, pred, oLabels, output,test_logits = model.test(idx_test, labels=trueLabels)#采用
        print("accuracy:{:.4f}".format(accuracy))
        s = Standard(pred, oLabels)
        precision = s.precision()
        recall = s.recall()
        f1 = s.f1()
        result[0].append(accuracy)
        result[1].append(precision)
        result[2].append(recall)
        result[3].append(f1)
        print("precision:{:.4f},recall:{:.4f},f1:{:.4f}".format(precision,recall,f1))

    printy(result, "result")