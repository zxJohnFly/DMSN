import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from torch.autograd import Variable
from sklearn.linear_model import RidgeClassifier
from sklearn import svm
from dataset import db, preprocessData, rowNormalize
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
   n_epoch = 300 
   batch_size = 128 
   
   loader = data.DataLoader(dataset=db(dset.S_tr, dset.X_tr, dset.y_tr),
                            batch_size=batch_size,
                            shuffle=True)
			    
	
   model, recon_loss = get(model)
   dualAE = model(dset.X_tr.shape[1], dset.S_tr.shape[1], dset.nclass)

   net = dualAE.cuda()

   print ('-------training V-Net --------')
   V_optim = torch.optim.Adam(net.parameters(), lr=0.0005, weight_decay=1e-5)
   for epoch in xrange(0):
      for i, (_, V, _) in enumerate(loader):
         dV = data_corrupt(V, 0.2)
         dV = Variable(dV).cuda()
         V = Variable(V).cuda()
         
         V_optim.zero_grad()
         latent, out = net.V_net_forward(dV)
         
         loss = recon_loss(out, V)

         if model == 'CAE':
            loss += cae_loss(net.get_VAE_W(), latent)*0.01
         
         loss.backward()
         V_optim.step()

         if (i + 1)%100 == 0:
              print ('Epoch [%d/40] [%d/%d], Loss: %.5f'
				 	   % (epoch+1, i+1, len(loader), loss.data[0]))
         
   print ('-------training S-Net --------')
   S_optim = torch.optim.Adam(net.parameters(), lr=0.00001, weight_decay=1e-5)
   for epoch in xrange(0):
      for i, (S, _, _) in enumerate(loader):
         dS = data_corrupt(S)
         dS = Variable(dS).cuda()
         S = Variable(S).cuda()
          
         S_optim.zero_grad()
         latent, out = net.S_net_forward(dS)
         loss = recon_loss(out, S)

         if model == 'CAE':
            loss += cae_loss(net.get_SAE_W(), latent)
 
         loss.backward()
         S_optim.step()

         if (i + 1)%100 == 0:
              print ('Epoch [%d/40] [%d/%d], Loss: %.5f'
				 	   % (epoch+1, i+1, len(loader), loss.data[0]))

   criter = nn.CrossEntropyLoss()
   optim = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5) #0.0005
   for epoch in xrange(n_epoch):
      for i, (S, V, L) in enumerate(loader):
         dS = data_corrupt(S, 0.01)
         dS = Variable(dS).cuda()
         S, V = Variable(S).cuda(), Variable(V).cuda()
         L = Variable(L).cuda()

         optim.zero_grad()
         latent, pred, out_v, out_s = net(dS)

         # reconstruction error
         recon_v_loss = recon_loss(out_v, V)
         recon_s_loss = recon_loss(out_s, S)
#         clf_loss = criter(pred, L)

         loss = recon_v_loss + recon_s_loss #+ clf_loss*10
         if model == 'CAE':
            loss += cae_loss(net.get_SAE_W(), latent)*0.001
         loss.backward()	
	 optim.step()
         
         if (i + 1)%50 == 0:
              print ('Epoch [%d/%d] [%d/%d], Loss: %.5f  --recon_v_loss: %.5f, recon_s_loss: %.5f, clf_loss: %.5f'
	 	   % (epoch+1, n_epoch, i+1, len(loader), loss.data[0],
         recon_v_loss.data[0], recon_s_loss.data[0], 0.0))#, clf_loss.data[0]))
      if (epoch + 1)%200 == 0:
         optim = torch.optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-5)

      kNN(net, dset)
   return net

def kNN(net, dset):
   S_te = Variable(torch.Tensor(dset.S_te)).cuda()

   net.eval()
   pred_s, pred_v = net.infer_s2v(S_te)
   net.train()
   
   pred_v = pred_v.data.cpu().numpy()

   dst = pairDst(dset.X_te_unseen, pred_v[np.unique(dset.y_te_unseen) - 1, :])
   dst_seen = pairDst(dset.X_te_seen, pred_v)
   dst_unseen = pairDst(dset.X_te_unseen, pred_v)

   te_table = np.unique(dset.y_te_unseen)
   pred = te_table[np.argmin(dst, axis=1)]
   pred_seen = np.argmin(dst_seen, axis=1) + 1
   pred_unseen = np.argmin(dst_unseen, axis=1) + 1

   seen_acc = []
   for y in stable_unique(dset.y_te_seen):
       loc = np.where(dset.y_te_seen == y)[0]
       seen_acc.append(np.mean(pred_seen[loc] == y))
   seen_acc = np.mean(seen_acc)   
   
   unseen_acc = []
   acc = []
   for y in stable_unique(dset.y_te_unseen):
       loc = np.where(dset.y_te_unseen == y)[0]
       unseen_acc.append(np.mean(pred_unseen[loc] == y))
       acc.append(np.mean(pred[loc] == y))
   unseen_acc = np.mean(unseen_acc) 
   acc = np.mean(acc)

   print ('-- zsl -- %.4f' % acc)
   print ('-- gzsl -- tr: %.4f te: %.4f H: %.4f' %
          (seen_acc, unseen_acc, 2*seen_acc*unseen_acc/(seen_acc + unseen_acc)))

def kNN_gzsl(net, dset):
    S_te = Variable(torch.Tensor(dset.S_te)).cuda()
    split_point = np.nonzero(dset.y_te == 1)[0][0]
   
    net.eval()
    pred_s, pred_v = net.infer_s2v(S_te)
    net.train()

    pred_v = pred_v.data.cpu().numpy()
    X_te_U, X_te_S = np.split(dset.X_te, [split_point], axis=0)
    y_te_U, y_te_S = np.split(dset.y_te, [split_point], axis=0)
    pred_v_U, pred_v_S = np.split(pred_v, [10], axis=0)
    
    Dst = pairDst(X_te_U,  pred_v_U)
    predictedIdx = np.argmin(Dst, axis=1)
    te_cls = stable_unique(y_te_U)
    acc_UU = np.mean(te_cls[predictedIdx] == y_te_U)

    Dst = pairDst(X_te_S,  pred_v_S)
    predictedIdx = np.argmin(Dst, axis=1)
    te_cls = stable_unique(y_te_S)
    acc_SS = np.mean(te_cls[predictedIdx] == y_te_S)


    Dst = pairDst(dset.X_te, pred_v)
    predictedIdx = np.argmin(Dst, axis=1)
    te_cls = stable_unique(dset.y_te)
    predictedLabel = te_cls[predictedIdx]

    acc_UT = np.mean(predictedLabel[:split_point] == dset.y_te[:split_point])
    acc_ST = np.mean(predictedLabel[split_point:] == dset.y_te[split_point:])

    print ('S --> V [kNN] [U->U] acc:%.4f [S->S] acc:%.4f [U-->T] acc:%.4f [S-->T] acc:%.4f' % (acc_UU, acc_SS, acc_UT, acc_ST))

def synth_images(net, dset, num=500):
   net.eval()
   
   exemplars = []
   labels = []

   for label, s in enumerate(dset.S_te):
      s = torch.Tensor(np.tile(s, (num, 1)))
      ss = data_corrupt(s).cuda()
      _, pred = net.infer_s2v(s_replica)
      exemplars.append(pred.data.cpu().numpy())
      labels.append([label]*num)

   exemplars = np.concatenate(exemplars)
   labels = np.concatenate(labels)
   
   net.train()
   return exemplars, labels

def mlp(exemplars, labels):
   clf = MLP(exemplars, labels)
   clf.optimize()
   return clf.predict(dset.X_te)

def rrc(exemplars, labels):
   clf = RidgeClassifier(alpha=50, tol=1e-5)
   clf.fit(exemplars, labels)   
   return  clf.predict(dset.X_te)
   
def acc_zsl(predicted_index, dset):
   te_cls = stable_unique(dset.y_te)
   predicted_label = te_cls[predicted_index]
  
   acc = np.mean(predicted_label == dset.y_te)
   print ('S --> V accuracy: %.4f' % acc)

if __name__ == '__main__':
   dataset = 'awa2'
   model = 'CAE'
   dset = preprocessData(dataset, unsupervised=True)
   net = train(dset, model)
    
#   net.eval()
#   S_te = Variable(torch.Tensor(dset.S_te)).cuda()
#   _, pred_v = net.infer_s2v(S_te)
#   pred_v = pred_v.data.cpu().numpy()

#   import scipy.io as sio
#   sio.savemat('gzsl.mat', {'pred_v': pred_v, 'X_te': dset.X_te, 'y_te': dset.y_te})

