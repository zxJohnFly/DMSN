import numpy as np
import torch
import torch.nn as nn
import scipy.io as sio

from torch.autograd import Variable
from dataset import udb, preprocessData, rowNormalize
from config import get, MLP
from utils import *


numpy_rng = np.random.RandomState(1234)
def data_corrupt(x, corruption_level=0.2):
    masks= numpy_rng.binomial(size=x.size(),
                            n=1,
                            p=1-corruption_level)
    return x.mul_(torch.Tensor(masks))

def gzsl_knn(net, dset):
    W_te = Variable(torch.Tensor(dset.W_te)).cuda()
    sp = np.nonzero(dset.y_te == np.min(dset.y_tr)+1)[0][0]
    net.eval()
    pred = net.infer_w2v(W_te)
    Dst = pairDst(dset.X_te, pred.data.cpu().numpy())

    te_cls = stable_unique(dset.y_te)
   
    masks = np.zeros(Dst.shape)
    masks[:,12:] = 1
    accs = []
    for i in np.arange(-1, 3, 0.01):
        pDst = Dst + i*masks
        predict_index = np.argmin(pDst, axis=1)
        l = te_cls[predict_index]

        acc_UT = np.mean(l[:sp] == dset.y_te[:sp])
        acc_ST = np.mean(l[sp:] == dset.y_te[sp:])
        accs.append([acc_UT, acc_ST])
  
    import scipy.io as sio
    sio.savemat('gzsl_apy.mat', {'accs': np.vstack(accs)})

def gzsl_mlp(net, dset, num=500):
    net.eval()

    exemplars = []
    labels = []

    for label, w in enumerate(dset.W_te):
       ws = torch.Tensor(np.tile(w, (num, 1)))
       ws_corrupt = Variable(data_corrupt(ws)).cuda()
       synth = net.infer_w2v(ws_corrupt)
       exemplars.append(synth.data.cpu().numpy())
       labels.append([label]*num)

    exemplars = np.concatenate(exemplars)
    labels = np.concatenate(labels)

    sp = np.nonzero(dset.y_te == np.min(dset.y_tr)+1)[0][0]
    net.eval()
    clf = MLP(exemplars, np.squeeze(labels), 50, 0.0001)
    clf.optimize()
    Dst = clf.softmax(dset.X_te).data.cpu().numpy()
    te_cls = stable_unique(dset.y_te)
    
    predict_index = np.argmax(Dst, axis=1)
    l = te_cls[predict_index]

    acc_UT = np.mean(l[:sp] == dset.y_te[:sp])
    acc_ST = np.mean(l[sp:] == dset.y_te[sp:])
    
    print acc_UT, acc_ST

    masks = np.zeros(Dst.shape)
    masks[:,10:] = 1
    accs = []

    for i in np.arange(-100, 100, 1):
        pDst = Dst - i*masks
        predict_index = np.argmax(pDst, axis=1)
        l = te_cls[predict_index]

        acc_UT = np.mean(l[:sp] == dset.y_te[:sp])
        acc_ST = np.mean(l[sp:] == dset.y_te[sp:])
        accs.append([acc_UT, acc_ST])

    import scipy.io as sio
    sio.savemat('mlp_gzsl_awa.mat', {'accs': np.vstack(accs)})


def mlp(net, dset, num=500):
    net.eval()

    exemplars = []
    labels = []

    for label, w in enumerate(dset.W_te):
       ws = torch.Tensor(np.tile(w, (num, 1)))
       ws_corrupt = Variable(data_corrupt(ws, 0.001)).cuda()
       synth = net.infer_w2v(ws_corrupt)
       exemplars.append(synth.data.cpu().numpy())
       labels.append([label]*num)

    exemplars = np.concatenate(exemplars)
    labels = np.concatenate(labels)
    
    clf = MLP(exemplars, np.squeeze(labels), 60, 0.0001)
    clf.optimize()
    predictedIdx = clf.predict(dset.X_te)
    te_cls = stable_unique(dset.y_te)
    predictedLabel = te_cls[predictedIdx]
   
    acc = np.mean(predictedLabel == dset.y_te)
    print ('S --> V acc: %.4f' % acc)

    return acc

def knn(net, dset):
    W_te = Variable(torch.Tensor(dset.W_te)).cuda()

    net.eval()
    pred = net.infer_w2v(W_te)
    Dst = pairDst(dset.X_te, pred.data.cpu().numpy())
    predictedIdx = np.argmin(Dst, axis=1)
    te_cls = stable_unique(dset.y_te)
    predictedLabel = te_cls[predictedIdx]
   
    acc = np.mean(predictedLabel == dset.y_te)
    print ('S --> V acc: %.4f' % acc)
    
    return Dst

def test(dataset, dset, m):
    model, _  = get(m)
    triAE = model(dset.X_te.shape[1], dset.S_te.shape[1], dset.nclass)
    net = triAE.cuda()
    
    net.load_state_dict(torch.load('./model/%s' % dataset))
    net.eval()
 
    Dst = knn(net, dset)
    sio.savemat('%s_apy_Dst.mat' % dataset, {'Dst': Dst})
#    W_te = Variable(torch.Tensor(dset.W_te)).cuda()
#    attr = net.infer_w2s(W_te) 
#    attr = attr.data.cpu().numpy()
#    accs = []
#    for i in range(1, 20):
#        acc = mlp(net, dset, i*100)
#        accs.append(acc)
#    sio.savemat('%s_accs.mat' % dataset, {'accs': accs})

if __name__ == '__main__':
   dataset = 'apy'
   m = 'triae'
   dset = preprocessData(dataset, unsupervised=True)
   test(dataset, dset, m)
  
