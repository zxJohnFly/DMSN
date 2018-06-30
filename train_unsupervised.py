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
   n_epoch = 200

   batch_size = 128 

   loader = data.DataLoader(dataset=udb(dset.S_tr, dset.X_tr, dset.y_tr, dset.W_tr),
                            batch_size=batch_size,
                            shuffle=True)
			    
   model, recon_loss = get(model)	
   triAE = model(dset.X_tr.shape[1], dset.S_tr.shape[1], dset.nclass)
   net = triAE.cuda()
 
   print ('-------training V-Net --------')
   V_optim = torch.optim.Adam(net.parameters(), lr=0.0005, weight_decay=1e-5)
   S_optim = torch.optim.Adam(net.parameters(), lr=0.0005, weight_decay=1e-5)
   W_optim = torch.optim.Adam(net.parameters(), lr=0.0005, weight_decay=1e-5)

   for epoch in xrange(0):
      for i, (S, V, W, _) in enumerate(loader):
         dV = data_corrupt(V)
         dV = Variable(dV).cuda()
         V = Variable(V).cuda()

         dS = data_corrupt(S)
         dS = Variable(dS).cuda()
         S = Variable(S).cuda()
 
         dW = data_corrupt(W)
         dW = Variable(dW).cuda()
         W = Variable(W).cuda()

         V_optim.zero_grad()
         S_optim.zero_grad()
         W_optim.zero_grad()

         latent_v, out_v = net.V_net_forward(dV)
         latent_s, out_s = net.S_net_forward(dS)
         latent_w, out_w = net.W_net_forward(dW)

         V_loss = recon_loss(out_v, V)
         S_loss = recon_loss(out_s, S)
         W_loss = recon_loss(out_w, W)
         
         V_loss.backward()
         V_optim.step()

         S_loss.backward()
         S_optim.step()
 
         W_loss.backward()
         W_optim.step()

         if (i + 1)%100 == 0:
              print ('Epoch [%d/40] [%d/%d], V_Loss: %.5f, S_Loss: %.5f, W_Loss: %.5f'
                      % (epoch+1, i+1, len(loader), V_loss.data[0], S_loss.data[0], W_loss.data[0]))
         
   criter = nn.CrossEntropyLoss()
   optim = torch.optim.Adam(net.parameters(), lr=0.0005, weight_decay=1e-5)
   for epoch in xrange(10):
      for i, (S, V, _, L) in enumerate(loader):
         dS = data_corrupt(S, 0.01)
         dS = Variable(dS).cuda()
         S, V, L = Variable(S).cuda(), Variable(V).cuda(), Variable(L).cuda()

         optim.zero_grad()
         latent, pred, out_sv, out_ss = net.SV_net_forward(dS)
         recon_sw_loss = recon_loss(out_sv, V)
         recon_ss_loss = recon_loss(out_ss, S)
         clf_loss = criter(pred, L)

         loss = recon_sw_loss + clf_loss + cae_loss(net.get_SAE_W(),
                                                    latent)*100
         loss.backward()
         optim.step()
         if (i + 1)%50 == 0:
            print ('Epoch [%d/%d] [%d/%d], Loss: %.5f  --recon_sv_loss: %.5f, clf_loss: %.5f'
	 	   % (epoch+1, 40, i+1, len(loader), loss.data[0],
         recon_sw_loss.data[0], 0))# clf_loss.data[0]))

   criter = nn.CrossEntropyLoss()
   optim = torch.optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-5)
   for epoch in xrange(n_epoch):
      for i, (S, V, W, L) in enumerate(loader):
         dW = data_corrupt(W, 0.01)
         dW = Variable(dW).cuda()
  
         dS = data_corrupt(S, 0.01)
         dS = Variable(dS).cuda()
         S, V, W = Variable(S).cuda(), Variable(V).cuda(), Variable(W).cuda()
         L = Variable(L).cuda()
         
         optim.zero_grad()

         #legacy
         #latent, pred, out_wv, out_ws, out_ww = net(dW)
         latent_w, latent_s, pred_w, pred_s, out_wv, out_ws, out_sv = net(dW, dS)

         # reconstruction error
         recon_wv_loss = recon_loss(out_wv, V)
         recon_ws_loss = recon_loss(out_ws, S)
         recon_sv_loss = recon_loss(out_sv, V)
         clf_loss = (criter(pred_s, L) + criter(pred_s, L))*1
                       
         loss = recon_wv_loss + recon_ws_loss + recon_sv_loss + clf_loss
        # loss += cae_loss(net.get_WAE_W(), latent_w)*0.000001
         loss.backward()
         optim.step()
         
         if (i + 1)%50 == 0:
              print ('Epoch [%d/%d] [%d/%d], Loss: %.5f  --recon_wv_loss: %.5f, recon_ws_loss: %.5f,  clf_loss: %.5f'
	 	   % (epoch+1, n_epoch, i+1, len(loader), loss.data[0],
         recon_wv_loss.data[0], recon_ws_loss.data[0], clf_loss.data[0]))
      if (epoch + 1)%200 == 0:
         optim = torch.optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-5)

      knn(net, dset)
#      if (epoch + 1)%20 == 0:
#         mlp(net, dset)
   return net

   
def mlp(net, dset, num=500):
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
    net.train()

    clf = MLP(exemplars, np.squeeze(labels), 50, 0.0001)
    clf.optimize()
    predictedIdx = clf.predict(dset.X_te)
    te_cls = stable_unique(dset.y_te)
    predictedLabel = te_cls[predictedIdx]
   
    acc = np.mean(predictedLabel == dset.y_te)
    print ('S --> V acc: %.4f' % acc)


def knn_gzsl(net, dset):
    W_te = Variable(torch.Tensor(dset.W_te)).cuda()
    split_point = np.nonzero(dset.y_te == np.min(dset.y_tr)+1)[0][0]
   
    net.eval()
    pred = net.infer_w2v(W_te)
    net.train()

    pred = pred.data.cpu().numpy()
    X_te_U, X_te_S = np.split(dset.X_te, [split_point], axis=0)
    y_te_U, y_te_S = np.split(dset.y_te, [split_point], axis=0)
    pred_v_U, pred_v_S = np.split(pred, [12], axis=0)

    Dst = pairDst(X_te_U, pred_v_U)
    predict_index = np.argmin(Dst, axis=1)
    te_cls = stable_unique(y_te_U)
    acc_UU = np.mean(te_cls[predict_index] == y_te_U)

    Dst = pairDst(X_te_S, pred_v_S)
    predict_index = np.argmin(Dst, axis=1)
    te_cls = stable_unique(y_te_S)
    acc_SS = np.mean(te_cls[predict_index] == y_te_S)

    Dst = pairDst(dset.X_te, pred)
    predict_index = np.argmin(Dst, axis=1)
    te_cls = stable_unique(dset.y_te)
    predict_labels = te_cls[predict_index]

    acc_UT = np.mean(predict_labels[:split_point] == dset.y_te[:split_point])
    acc_ST = np.mean(predict_labels[split_point:] == dset.y_te[split_point:])

    print ('[U->U] %.4f [S->S] %.4f [U->T] %.4f [S->T] %.4f' %
             (acc_UU, acc_SS, acc_UT, acc_ST))

def knn(net, dset):
    W_te = Variable(torch.Tensor(dset.W_te)).cuda()
   
    net.eval()
    pred = net.infer_w2v(W_te)
    net.train()

    pred = pred.data.cpu().numpy()
    dst = pairDst(dset.X_te_unseen, pred[np.unique(dset.y_te_unseen) - 1, :])
    dst_seen = pairDst(dset.X_te_seen, pred)
    dst_unseen = pairDst(dset.X_te_unseen, pred)

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

    print ('-- zsl -- acc: %.4f' % acc)
    print ('ts: %.4f, tr: %.4f, H: %.4f'
           % (unseen_acc, seen_acc, 2*unseen_acc*seen_acc/(unseen_acc + seen_acc)))

if __name__ == '__main__':
   dataset = 'apy'
   model = 'triae'
   dset = preprocessData(dataset, unsupervised=True)
   net = train(dset, model)
   print dataset
#   torch.save(net.state_dict(), './model/%s' % dataset)
