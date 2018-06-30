import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from torch.autograd import Variable
from sklearn.linear_model import RidgeClassifier
from dataset import udb, preprocessData, rowNormalize
from config import get, MLP
from utils import *


numpy_rng = np.random.RandomState(1234)
def data_corrupt(x, corruption_level=0.2):
    masks= numpy_rng.binomial(size=x.size(),
                            n=1,
                            p=1-corruption_level)
    return x.mul_(torch.Tensor(masks))

def cae_loss(w, latent):
    derv = latent*(1-latent)
    return torch.mm(derv**2, torch.sum(Variable(w)**2, dim=1)).sum()
    
def train(dset, model):
   n_epoch = 135 
   batch_size = 128 

   loader = data.DataLoader(dataset=udb(dset.S_tr, dset.X_tr, dset.y_tr, dset.W_tr),
                            batch_size=batch_size,
                            shuffle=True)

   net, recon_loss = get(model)			    
   net = net(dset.X_tr.shape[1], dset.S_tr.shape[1], dset.nclass)
   net = net.cuda()
 
   criter = nn.CrossEntropyLoss()
   optim = torch.optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-6)
   for epoch in xrange(10):
      for i, (S, V, _, L) in enumerate(loader):
         dS = data_corrupt(S, 0.001)
         dS = Variable(dS).cuda()
         S, V, L = Variable(S).cuda(), Variable(V).cuda(), Variable(L).cuda()

         optim.zero_grad()
         latent, pred, out_sv, out_ss = net.SV_net_forward(dS)
         recon_sv_loss = recon_loss(out_sv, V)
         recon_ss_loss = recon_loss(out_ss, S)
#         clf_loss = criter(pred, L)

         loss = recon_sv_loss# + clf_loss*100
         loss.backward()
         optim.step()
         if (i + 1)%50 == 0:
            print ('Epoch [%d/%d] [%d/%d], Loss: %.5f  --recon_sv_loss: %.5f, clf_loss: %.5f'
	 	   % (epoch+1, 40, i+1, len(loader), loss.data[0],
         recon_sv_loss.data[0], 0))#clf_loss.data[0]))
       
#   criter = nn.CrossEntropyLoss()
   optim = torch.optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-6)
   for epoch in xrange(n_epoch):
      for i, (S, V, W, L) in enumerate(loader):
         dW = data_corrupt(W, 0.001)
         dW = Variable(dW).cuda()
         
         dS = data_corrupt(S, 0.001)
         dS = Variable(dS).cuda()

         S, V, W = Variable(S).cuda(), Variable(V).cuda(), Variable(W).cuda()
         L = Variable(L).cuda()

         optim.zero_grad() 
         latent, pred, out_wv, out_ws, out_ww = net(dW, dS)

         # reconstruction error
         recon_wv_loss = recon_loss(out_wv, V)
         recon_ws_loss = recon_loss(out_ws, S)
#         clf_loss = criter(pred, L)
                       
         loss = recon_wv_loss + recon_ws_loss #+ clf_loss*1000
         loss += cae_loss(net.get_WAE_W(), latent)*0.001
         loss.backward()	
    	 optim.step()
         
         if (i + 1)%50 == 0:
              print ('Epoch [%d/%d] [%d/%d], Loss: %.5f  --recon_wv_loss: %.5f, recon_ws_loss: %.5f,  clf_loss: %.5f'
	 	   % (epoch+1, n_epoch, i+1, len(loader), loss.data[0],
         recon_wv_loss.data[0], recon_ws_loss.data[0], 0))#clf_loss.data[0]))

      test(net, dset)
   mlp(net, dset)
   return net

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
    net.train()
    clf = MLP(exemplars, np.squeeze(labels), 60, 0.0001)
    clf.optimize()
    predictedIdx = clf.predict(dset.X_te)
    te_cls = stable_unique(dset.y_te)
    predictedLabel = te_cls[predictedIdx]
   
    acc = np.mean(predictedLabel == dset.y_te)
    print ('S --> V acc: %.4f' % acc)

def test(net, dset):
    W_te = Variable(torch.Tensor(dset.W_te)).cuda()

    net.eval()
    pred = net.infer_w2v(W_te)
    net.train()
    Dst = pairDst(dset.X_te, pred.data.cpu().numpy(), 'cosine')
    predictedIdx = np.argmin(Dst, axis=1)
    te_cls = stable_unique(dset.y_te)
    predictedLabel = te_cls[predictedIdx]
    acc = np.mean(predictedLabel == dset.y_te)
    print ('S --> V acc: %.4f' % acc)

def t(net, dset):
    S_te = Variable(torch.Tensor(dset.S_te)).cuda()

    net.eval()
    pred = net.infer_s2v(S_te)
    net.train()
    Dst = pairDst(dset.X_te, pred.data.cpu().numpy(), 'cosine')
    predictedIdx = np.argmin(Dst, axis=1)
    te_cls = stable_unique(dset.y_te)
    predictedLabel = te_cls[predictedIdx]
    acc = np.mean(predictedLabel == dset.y_te)
    print ('S --> V acc: %.4f' % acc)

if __name__ == '__main__':
   dataset = 'awa2'
   model = 'triae'
   dset = preprocessData(dataset, unsupervised=True)
   net = train(dset, model)
   torch.save(net.state_dict(), './model/%s' % dataset)
