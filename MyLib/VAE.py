'''
    implement VAE
    author:kyonio

'''

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
nChannel = 32

class one_layer(nn.Module):
    def __init__(self):
        super(one_layer,self).__init__()
        self.fc = nn.Linear(nChannel,138)
    def forward(self,z):
        return self.fc(z)


class VAE(nn.Module):
    def __init__(self):
        super(VAE,self).__init__()

        self.Normalize = nn.BatchNorm2d
        self.fc1 = nn.Linear(nChannel*8,nChannel)
        self.fc = nn.Linear(nChannel,138)
        self.encoder_conv = nn.Sequential(

            nn.Conv2d(1, nChannel, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            self.Normalize(nChannel),

            nn.Conv2d(nChannel, nChannel*2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            self.Normalize(nChannel*2),

            nn.Conv2d(nChannel*2, nChannel*4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            self.Normalize(nChannel*4),

            nn.Conv2d(nChannel*4, nChannel*8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            self.Normalize(nChannel*8)

        )

        self.encoder_dense = nn.Sequential(
            nn.Linear(nChannel*8*16, nChannel*8),
            nn.ReLU()
        )

        self.decoder_dense = nn.Sequential(
            nn.Linear(nChannel,nChannel*8),
            nn.ReLU(),
            nn.Linear(nChannel*8,nChannel*8*16),
            nn.ReLU()
        )

        self.decoder_conv = nn.Sequential(

            
            
            nn.ConvTranspose2d(nChannel*8, nChannel*4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            self.Normalize(nChannel*4),

            nn.ConvTranspose2d(nChannel*4, nChannel*2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            self.Normalize(nChannel*2),

            nn.ConvTranspose2d(nChannel*2, nChannel, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            self.Normalize(nChannel),

            nn.ConvTranspose2d(nChannel, 1, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            self.Normalize(1),
            nn.Sigmoid()
        )

    def encoder(self,x):
        h1 = self.encoder_conv(x)
        flat = h1.view(-1,nChannel*8*16)
        hidden = self.encoder_dense(flat)
        z_mean = self.fc1(hidden)
        z_logvar = self.fc1(hidden)

        return z_mean, z_logvar

    def reparameterize(self,mu,logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def decoder(self,z):
        z1 = self.decoder_dense(z)
        z1 = z1.view(-1,nChannel*8,4,4)
        z2 = self.decoder_conv(z1)

        return z2.view(-1,64*64)

        

    def forward(self,x):
        mu,logvar = self.encoder(x)
        z = self.reparameterize(mu,logvar)
        return self.decoder(z),mu,logvar,z



def DC_VAE(**kwargs):
    model = VAE()
    return model

def Linear_model():
    model = one_layer()
    return model
