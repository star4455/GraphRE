import os.path as osp
import warnings
import pickle

class  PrePtbDataset:
    def __init__(self, root, name, attack_method='nettack', ptb_rate=1.0):

        if attack_method == 'mettack' or attack_method == 'metattack':
            attack_method = 'meta'
        self.name = name.lower()
        self.attack_method = attack_method
        self.ptb_rate = ptb_rate
        self.root = osp.expanduser(osp.normpath(root))
        self.data_filename = osp.join(root,
               'gcn_{}_{}_adj_{}.pkl'.format(self.name, self.attack_method, self.ptb_rate))#gai  random attack！！！！！！
        self.target_nodes = None
        self.adj = self.load_data()

    def load_data(self):
        if not osp.exists(self.data_filename):
            self.download_npz()
        print('Loading {} dataset perturbed by {} {}...'.format(self.name, self.ptb_rate, self.attack_method))

        if self.attack_method == 'meta':
            warnings.warn("The pre-attacked graph is perturbed under the data splits provided by ProGNN. So if you are going to verify the attacking performance, you should use the same data splits  (setting='prognn').")
            adj = pickle.load(open(self.data_filename, 'rb'))

        if self.attack_method == 'nettack':
            warnings.warn("The pre-attacked graph is perturbed under the data splits provided by ProGNN. So if you are going to verify the attacking performance, you should use the same data splits  (setting='prognn').")
            adj = pickle.load(open('C://Users/ccyx/Desktop/pkl_gen/gcn/1300gcn_{}_{}_adj_{}.pkl'.format(self.name, self.attack_method, self.ptb_rate),'rb'))
        if self.attack_method == 'RandomAttack':
            warnings.warn("The pre-attacked graph is perturbed under the data splits provided by ProGNN. So if you are going to verify the attacking performance, you should use the same data splits  (setting='prognn').")
            adj = pickle.load(open('C://Users/ccyx/Desktop/pkl_gen/gcn/1300gcn_{}_{}_adj_{}.pkl'.format(self.name, self.attack_method, self.ptb_rate),'rb'))
        return adj



