from collections import namedtuple
from sklearn.preprocessing import MinMaxScaler

from utils import stable_unique
import torch.utils.data as data
import torch
import numpy as np


paths = {
    'awa1': '/mnt/disk1/xlsa17/data/AWA1/AWA1_res101.npz', 
    'awa2': '/mnt/disk1/xlsa17/data/AWA2/AWA2_res101.npz', 
    'apy': '/mnt/disk1/xlsa17/data/APY/APY_res101.npz',
    'sun': '/mnt/disk1/xlsa17/data/SUN/SUN_res101.npz'
}

dset = namedtuple('dset', ['X_tr', 'X_te_seen', 'X_te_unseen', 'S_tr', 'S_te', 
                           'y_tr', 'y_te_seen', 'y_te_unseen', 'nclass'])

udset = namedtuple('udset', ['X_tr', 'X_te_seen', 'X_te_unseen', 
                             'S_tr', 'S_te', 'W_tr' , 'W_te',
                             'y_tr', 'y_te_seen', 'y_te_unseen', 'nclass'])

def preprocessData(dataset='awa', unsupervised=False):
    data = np.load(paths[dataset])

    X_tr = data['X_tr']
    X_te_seen = data['X_te_seen']
    X_te_unseen = data['X_te_unseen'] 

    X_tr = X_tr/np.max(X_tr, axis=1)[:, None]
    X_te_seen = X_te_seen/np.max(X_te_seen, axis=1)[:, None]
    X_te_unseen = X_te_unseen/np.max(X_te_unseen, axis=1)[:, None]
   
    y_tr = np.squeeze(data['y_tr'])
    y_table = np.unique(y_tr)
    y_tr = np.array([np.where(y==y_table)[0][0] for y in y_tr])
    y_te_seen = np.squeeze(data['y_te_seen'])
    y_te_unseen = np.squeeze(data['y_te_unseen'])
   
    S_tr = np.float32(data['S_tr'])
    S_te = np.float32(data['S_te'])

    nclass = S_tr.shape[0]
    S_tr_max, S_te_max = np.max(S_tr, axis=1), np.max(S_te, axis=1)
    S_tr_min, S_te_min = np.min(S_tr, axis=1), np.min(S_te, axis=1)

    S_tr = S_tr - np.min(S_tr, axis=1)[:, None]
    S_tr = S_tr/(S_tr_max - S_tr_min)[:, None]
    S_te = S_te - np.min(S_te, axis=1)[:, None]
    S_te = S_te/(S_te_max - S_te_min)[:, None]

    S_tr = S_tr[np.squeeze(y_tr), :]
    if unsupervised:
       W_tr = np.float32(data['W_tr'])
       W_te = np.float32(data['W_te'])

       W_tr_max, W_te_max = np.max(W_tr, axis=1), np.max(W_te, axis=1)
       W_tr_min, W_te_min = np.min(W_tr, axis=1), np.min(W_te, axis=1)

       W_tr = W_tr - np.min(W_tr, axis=1)[:, None]
       W_tr = W_tr/(W_tr_max - W_tr_min)[:, None]
       W_te = W_te - np.min(W_te, axis=1)[:, None]
       W_te = W_te/(W_te_max - W_te_min)[:, None]

       W_tr = W_tr[np.squeeze(y_tr), :]
       return udset(X_tr, X_te_seen, X_te_unseen, S_tr, S_te, W_tr, W_te,
                    y_tr, y_te_seen, y_te_unseen, nclass)
    return dset(X_tr, X_te_seen, X_te_unseen, S_tr, S_te, 
                y_tr, y_te_seen, y_te_unseen, nclass)

def rowNormalize(x, epsilon=1e-6):
    rowNorm = np.linalg.norm(x, axis=1)[:, None]
    rowNorm[np.where(rowNorm == 0)] = 1

    x = x/rowNorm
    return x

def colNormalize(x, epsilon=1e-6):
    colNorm = np.linalg.norm(x, axis=0)
    colNorm[np.where(colNorm == 0)] = epsilon

    x = x/colNorm
    return x

class db(data.Dataset):
    def __init__(self, attributes, images, labels):
        self.attributes = torch.Tensor(attributes)
        self.images = torch.Tensor(images)
        self.labels = torch.LongTensor(labels.tolist())

        self.length = self.attributes.size(0)

    def __getitem__(self, index):
        return self.attributes[index, :], self.images[index, :],  self.labels[index]

    def __len__(self):
        return self.length


class udb(data.Dataset):
    def __init__(self, attributes, images, labels, word_vector):
        self.attributes = torch.Tensor(attributes)
        self.images = torch.Tensor(images)
        self.labels = torch.LongTensor(labels.tolist())
        self.word_vector = torch.Tensor(word_vector)
        self.length = self.attributes.size(0)

    def __getitem__(self, index):
        return self.attributes[index, :], self.images[index, :], self.word_vector[index, :], self.labels[index]

    def __len__(self):
        return self.length

