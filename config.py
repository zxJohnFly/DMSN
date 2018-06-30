from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np


mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()
def get(model):
    if model == 'AE' or model == 'CAE':
       return AE, AE_recon_loss
    elif model == 'triae':
       return TripletAE, AE_recon_loss
    elif model == 'ZAAE':
       return ZAAE, mse_loss, bce_loss
    elif model == 'VAE':
       return VAE, VAE_recon_loss

def AE_recon_loss(recon_x, x):
    return bce_loss(recon_x, x)

def VAE_recon_loss(recon_info, x):
    recon_x, mu, logvar = recon_info
    MSE = bce_loss(recon_x, x)
    
    KLD = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= x.size(0)*x.size(1)
    return MSE + KLD
   
class AE(nn.Module):
    def __init__(self, dim1, dim2, nclass):
        super(AE, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        dim3 =  100

        self.VEncoder = nn.Sequential(
              	nn.Linear(dim1, 1000),
	            nn.LeakyReLU(0.1), nn.Dropout(0.2),
	            nn.Linear(1000, 500),
                nn.LeakyReLU(0.1), nn.Dropout(0.2),
                nn.Linear(500, dim3))

        self.VDecoder = nn.Sequential(
		        nn.Linear(dim3, 500),
                nn.LeakyReLU(0.1), nn.Dropout(0.2),
                nn.Linear(500, 1000),
                nn.LeakyReLU(0.1), nn.Dropout(0.2),
                nn.Linear(1000, dim1), nn.Sigmoid())

        self.SEncoder = nn.Sequential(
                nn.Linear(dim2, 200),
                nn.LeakyReLU(0.1), nn.Dropout(0.2),
                nn.Linear(200, dim3))

        self.SDecoder = nn.Sequential(
		nn.Linear(dim3, 200),
                nn.LeakyReLU(0.1), nn.Dropout(0.2),
                nn.Linear(200, dim2), nn.Sigmoid())	

        self.classifier = nn.Linear(dim3, nclass)

        torch.manual_seed(12)
	torch.cuda.manual_seed(12)
	for m in self.modules():
	    if isinstance(m, nn.Linear):
               # Glorot initialisation. gain = 1
	       fan_in, fan_out = m.in_features, m.out_features       
               m.weight.data.normal_(0, np.sqrt(2.0/(fan_in+fan_out)))
               m.bias.data.fill_(0)

    def get_SAE_W(self):
        return self.state_dict()['SEncoder.3.weight']
 
    def get_VAE_W(self):
        return self.state_dict()['VEncoder.6.weight']

    def forward(self, s):
        latent = self.SEncoder(s)
        pred = self.classifier(latent)

	out_v = self.VDecoder(latent)
	out_s = self.SDecoder(latent)
	return latent, pred, out_v, out_s

    def V_forward(self, v):
        latent = self.VEncoder(v)
        pred = self.classifier(latent)

        out_v = self.VDecoder(latent)
        out_s = self.SDecoder(latent)
        return latent, pred, out_v, out_s
       
    def S_net_forward(self, s):
        latent = self.SEncoder(s)
        return latent, self.SDecoder(latent)

    def V_net_forward(self, v):
        latent = self.VEncoder(v)
        return latent, self.VDecoder(latent)

    def infer_v2s(self, v):
        latent = self.VEncoder(v)
        out_s = self.SDecoder(latent)
        out_v = self.VDecoder(latent)
        
        return out_s, out_v

    def infer_s2v(self, s):
        latent = self.SEncoder(s)
        
        out_v = self.VDecoder(latent)
        out_s = self.SDecoder(latent)

        return out_s, out_v

class TripletAE(nn.Module):
    def __init__(self, dim1, dim2, nclass):
        super(TripletAE, self).__init__()
	self.dim1 = dim1
        self.dim2 = dim2

        self.dim3 = 70 if dim2 == 64 else 100
        self.VEncoder = nn.Sequential(
        	nn.Linear(dim1, 500),
	        nn.LeakyReLU(0.1), nn.Dropout(0.2),
	        nn.Linear(500, 200),
                nn.LeakyReLU(0.1), nn.Dropout(0.2),
                nn.Linear(200, self.dim3))

        self.VDecoder = nn.Sequential(
		nn.Linear(self.dim3, 200),
                nn.LeakyReLU(0.1), nn.Dropout(0.2),
                nn.Linear(200, 500),
                nn.LeakyReLU(0.1), nn.Dropout(0.2),
                nn.Linear(500, dim1), nn.Sigmoid())

        self.SEncoder = nn.Sequential(
                nn.Linear(dim2, 200),
                nn.LeakyReLU(0.1), nn.Dropout(0.2),
                nn.Linear(200, self.dim3))

        self.SDecoder = nn.Sequential(
		nn.Linear(self.dim3, 200),
                nn.LeakyReLU(0.1), nn.Dropout(0.2),
                nn.Linear(200, self.dim2), nn.Sigmoid())

        self.WEncoder = nn.Sequential(
                nn.Linear(300, 200),
                nn.LeakyReLU(0.1), nn.Dropout(0.2),
                nn.Linear(200, self.dim3))

        self.WDecoder = nn.Sequential(
		nn.Linear(self.dim3, 200),
                nn.LeakyReLU(0.1), nn.Dropout(0.2),
                nn.Linear(200, 300), nn.Sigmoid())	
        
        self.classifier = nn.Linear(self.dim3, nclass)
        self.sclassifier = nn.Linear(self.dim3, nclass)

        torch.manual_seed(12)
	torch.cuda.manual_seed(12)
	for m in self.modules():
	    if isinstance(m, nn.Linear):
               # Glorot initialisation. gain = 1
	       fan_in, fan_out = m.in_features, m.out_features       
               m.weight.data.normal_(0, np.sqrt(2.0/(fan_in+fan_out)))
               m.bias.data.fill_(0)

    def get_WAE_W(self):
        return self.state_dict()['WEncoder.3.weight']

    def get_SAE_W(self):
        return self.state_dict()['SEncoder.3.weight']
 
    def get_VAE_W(self):
        return self.state_dict()['VEncoder.9.weight']

    def forward(self, w, s):
        latent_w = self.WEncoder(w)
        latent_s = self.SEncoder(s)

        pred_w = self.classifier(latent_w)
        pred_s = self.sclassifier(latent_s)   

        out_wv = self.VDecoder(latent_w)
        out_ws = self.SDecoder(latent_w)
        out_sv = self.VDecoder(latent_s)
        return latent_w, latent_s, pred_w, pred_s, out_wv, out_ws, out_wv

   
    def forward_legacy(self, w):
        latent = self.WEncoder(w)
        pred = self.classifier(latent)
   
        out_wv = self.VDecoder(latent)
        out_ws = self.SDecoder(latent)
        out_ww = self.WDecoder(latent)
        return latent, pred, out_wv, out_ws, out_ww

    def SV_net_forward(self, x):
        latent = self.SEncoder(x)
        pred = self.sclassifier(latent)
 
        out_ss = self.SDecoder(latent)
        out_sv = self.VDecoder(latent)
        return latent, pred, out_sv, out_ss

    def S_net_forward(self, s):
        latent = self.SEncoder(s)
        return latent, self.SDecoder(latent)

    def V_net_forward(self, v):
        latent = self.VEncoder(v)
        return latent, self.VDecoder(latent)

    def W_net_forward(self, w):
        latent = self.WEncoder(w)
        return latent, self.WDecoder(latent)
       
    def infer_w2v(self, w):
        latent = self.WEncoder(w)
        out = self.VDecoder(latent)
        return out

    def infer_s2v(self, s):
        latent = self.SEncoder(s)
        out = self.VDecoder(latent)
        return out
  
    def infer_w2s(self, w):
        latent = self.WEncoder(w)
        out = self.SDecoder(latent)
        return out

class ZAAE(nn.Module):
    def __init__(self, dim1, dim2, nclass):
        super(ZAAE, self).__init__()
	self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = 100

        self.VEncoder = nn.Sequential(
        	nn.Linear(self.dim1, 2000),
	        nn.LeakyReLU(0.01), nn.Dropout(0.5),
                nn.Linear(2000, 1000),
                nn.LeakyReLU(0.01), nn.Dropout(0.5),
                nn.Linear(1000, self.dim3))

	self.VDecoder = nn.Sequential(
                nn.Linear(self.dim3, 1000),
                nn.LeakyReLU(0.01), nn.Dropout(0.5),
                nn.Linear(1000, 2000),
                nn.LeakyReLU(0.01), nn.Dropout(0.5),
                nn.Linear(2000, self.dim1), nn.Sigmoid())

        self.SEncoder = nn.Sequential(
                nn.Linear(dim2, 200),
                nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(200, self.dim3))

        self.SDecoder = nn.Sequential(
		nn.Linear(self.dim3, 200),
                nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(200, self.dim2), nn.Sigmoid())

        self.WEncoder = nn.Sequential(
                nn.Linear(300, 200),
                nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(200, self.dim3))

        self.WDecoder = nn.Sequential(
		nn.Linear(self.dim3, 200),
                nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(200, 300), nn.Sigmoid())	
        
        self.classifier = nn.Linear(self.dim3, nclass)
        self.sclassifier = nn.Linear(self.dim3, nclass)

        torch.manual_seed(12)
	torch.cuda.manual_seed(12)
	for m in self.modules():
	    if isinstance(m, nn.Linear):
               # Glorot initialisation. gain = 1
	       fan_in, fan_out = m.in_features, m.out_features       
               m.weight.data.normal_(0, np.sqrt(2.0/(fan_in+fan_out)))
               m.bias.data.fill_(0)

    def get_WAE_W(self):
        return self.state_dict()['WEncoder.3.weight']

    def get_SAE_W(self):
        return self.state_dict()['SEncoder.3.weight']
 
    def get_VAE_W(self):
        return self.state_dict()['VEncoder.6.weight']

    def forward_legacy(self, w, s):
        latent_w = self.WEncoder(w)
        latent_s = self.SEncoder(s)

        pred_w = self.classifier(latent_w)
        pred_s = self.sclassifier(latent_s)   

        out_wv = self.VDecoder(latent_w)
        out_ws = self.SDecoder(latent_w)
        out_sv = self.VDecoder(latent_s)
        return pred_w, pred_s, out_wv, out_ws, out_wv


    def forward(self, w):
        latent = self.WEncoder(w)
        pred = self.classifier(latent)
   
        out_wv = self.VDecoder(latent)
        out_ws = self.SDecoder(latent)
        out_ww = self.WDecoder(latent)
        return latent, pred, out_wv, out_ws, out_ww

    def SV_net_forward(self, x):
        latent = self.SEncoder(x)
        pred = self.sclassifier(latent)
 
        out_ss = self.SDecoder(latent)
        out_sv = self.VDecoder(latent)
        return latent, pred, out_sv, out_ss

    def S_net_forward(self, s):
        latent = self.SEncoder(s)
        return latent, self.SDecoder(latent)

    def V_net_forward(self, v):
        latent = self.VEncoder(v)
        return latent, self.VDecoder(latent)

    def W_net_forward(self, w):
        latent = self.WEncoder(w)
        return latent, self.WDecoder(latent)
       
    def infer_w2v(self, w):
        latent = self.WEncoder(w)
        out = self.VDecoder(latent)
        return out

    def infer_s2v(self, s):
        latent = self.SEncoder(s)
        out = self.VDecoder(latent)
        return out
   
    def infer_w2s(self, w):
        latent = self.WEncoder(w)
        out = self.SDecoder(latent)
        return out


class VAE(nn.Module):
    def __init__(self, dim1, dim2):
        super(VAE, self).__init__()

 	self.VEncoder = nn.Sequential(
        	nn.Linear(dim1, 500),
	        nn.ReLU(True), nn.Dropout(0.2),
	        nn.Linear(500, 250),
                nn.ReLU(True), nn.Dropout(0.2))
        self.VE1 = nn.Linear(250, 50)
        self.VE2 = nn.Linear(250, 50)

	self.VDecoder = nn.Sequential(
		nn.Linear(50, 250),
		nn.ReLU(True), nn.Dropout(0.2),
		nn.Linear(250, 500),
                nn.ReLU(True), nn.Dropout(0.2),
                nn.Linear(500, dim1))

        self.SEncoder = nn.Sequential(
                nn.Linear(dim2, 100),
                nn.ReLU(True), nn.Dropout(0.2))
        self.SE1 = nn.Linear(100, 50)
	self.SE2 = nn.Linear(100, 50)

        self.SDecoder = nn.Sequential(
		nn.Linear(50, 100),
                nn.ReLU(True), nn.Dropout(0.2),
                nn.Linear(100, dim2))	

	for m in self.modules():
	    if isinstance(m, nn.Linear):
               # Glorot initialisation. gain = 1
	       fan_in, fan_out = m.in_features, m.out_features       
               m.weight.data.normal_(0, np.sqrt(2.0/(fan_in+fan_out)))

    def reparameterize(self, mu, logvar):
        if self.training:
           std = logvar.mul(0.5).exp_()
           eps = Variable(std.data.new(std.size()).normal_())
           return eps.mul(std).add_(mu)
        else:
           return mu

    def _vencoder(self, x):
        h = self.VEncoder(x)
        mu, logvar = self.VE1(h), self.VE2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def _sencoder(self, x):
        h = self.SEncoder(x)
        mu, logvar = self.SE1(h), self.SE2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def _vdecoder(self, z):
        return self.VDecoder(z)

    def _sdecoder(self, z):
        return self.SDecoder(z)

    def V_net_forward(self, v):
        z, mu, logvar = self._vencoder(v)
        return self._vdecoder(z), mu, logvar

    def S_net_forward(self, s):
        z, mu, logvar = self._sencoder(s)
        return self._sdecoder(z), mu, logvar
        
    def forward(self, v, s):
        z_s, mu_s, logvar_s = self._sencoder(s)
        z_v, mu_v, logvar_v = self._vencoder(v)

        out_s = self._sdecoder(z_s)
        out_v = self._vdecoder(z_s)
        return z_v, z_s, (out_v, mu_v, logvar_v), (out_s, mu_s, logvar_s)

    def infer_v2s(self, v):
        latent, _, _ = self._vencoder(v)
        out = self._sdeencoder(latetn)
        return out

    def infer_s2v(self, s):
        latent, _, _ = self._sencoder(s)
        out = self._vdecoder(latent)
        return out
       
class MLP(object):
    def __init__(self, exemplars, labels, epochs, lr):
        nclass = np.max(labels)+1

        self.loader = data.DataLoader(dataset=simple_db(exemplars, labels),
                                      batch_size=128,
                                      shuffle=True)

        self.net = simple_nn(exemplars.shape[1], nclass).cuda()
        self.loss = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.epochs = epochs

    def optimize(self):
        for epoch in xrange(self.epochs):
            for x, y in self.loader:
                x, y = Variable(x).cuda(), Variable(y).cuda()

                self.optim.zero_grad()
                out = self.net(x)
                loss = self.loss(out, y)
            
                loss.backward()
                self.optim.step()

    def predict(self, x):
        x = Variable(torch.Tensor(x)).cuda()
        y = self.net(x)
        return np.argmax(y.data.cpu().numpy(), axis=1)

    def softmax(self, x):
        return self.net(Variable(torch.Tensor(x)).cuda())

class simple_nn(nn.Module):
    def __init__(self, n_input, n_output):
       super(simple_nn, self).__init__()
       self.net = nn.Sequential(
                     nn.Linear(n_input, n_output))
  
    def forward(self, x):
       return self.net(x)

class simple_db(data.Dataset):
    def __init__(self, x, y):
        self.x = torch.Tensor(x)
        self.y = torch.LongTensor(y.tolist())
 
        self.length = self.y.size(0)

    def __getitem__(self, index):
        return self.x[index, :], self.y[index]

    def __len__(self):
        return self.length
