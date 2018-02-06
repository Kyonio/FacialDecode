'''
    implement VAE
    author:kyonio

'''

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
nChannel = 32

layer_names = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
               'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
               'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
               'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
               'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5']
default_content_layers = ['relu1_2', 'relu2_2', 'relu3_4']


class one_layer(nn.Module):
    def __init__(self):
        super(one_layer,self).__init__()
        self.fc = nn.Linear(nChannel,138)
    def forward(self,z):
        return self.fc(z)

class _VGG(nn.Module):
    '''
    Classic pre-trained VGG19 model.
    Its forward call returns a list of the activations from
    the predefined content layers.
    '''

    def __init__(self):
        super(_VGG, self).__init__()

        features = models.vgg19(pretrained=True).features

        self.features = nn.Sequential()
        for i, module in enumerate(features):
            name = layer_names[i]
            self.features.add_module(name, module)

    def forward(self, input):
        batch_size = 128
        all_outputs = []
        output = input
        for name, module in self.features.named_children():
            
            output = module(output)
            if name in default_content_layers:
                all_outputs.append(output.view(batch_size, -1))
        return all_outputs
    
    
    
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
        '''
        self.upsample_decoder = nn.Sequential(
        
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(nChannel*8,nChannel*4,3,padding=1),
            nn.LeakyReLU(0.2),
            self.Normalize(nChannel*4,1e-3),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(nChannel*4, nChannel*2, 3, padding=1),
            nn.LeakyReLU(0.2),
            self.Normalize(128,1e-3),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(nChannel*2, nChannel, 3, padding=1),
            nn.LeakyReLU(0.2),
            self.Normalize(64, 1e-3),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(64, 1, 3, padding=1)
            
            
        
        
        )
    '''
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

def pre_trained_vgg():
    model = _VGG()
    return model

