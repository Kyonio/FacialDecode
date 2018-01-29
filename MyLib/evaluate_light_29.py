from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import average_precision_score
from light_cnn import LightCNN_29Layers
#from logger import Logger
from load_celebA import celeba_dataset
import pdb

def load_part_model(model,pretrained_dict):
    '''
    this function provides a way to load part model(checkpoint) in pytorch
    example:
    checkpoint = torch.load('/path/to/checkpoint')
    model = Neural_Network()
    
    the checkpoint and the model must belong to the same class
    then:
    model = load_part_model(model,checkpoint)
    '''
    model_dict = model.state_dict()
    pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model

def train(epoch):
    Light_CNN_29.train()
    train_loss = 0
    batch_size = 128
    for batch_idx, data in enumerate(train_loader):
        #transform the data
        #pdb.set_trace()
        image = Variable(data['image'].float().view(-1,1,128,128))
        target = Variable(data['label'])
        if torch.cuda.is_available():
            image = image.cuda()
            target = target.cuda()
        #pdb.set_trace()
        optimizer.zero_grad()
        #get get output from network
        _, output = Light_CNN_29(image)

        loss = BCEWithLogitsLoss(output,target)
	
        predict = nn.functional.sigmoid(output)>0.5
        r = (predict==target.byte())
        acc = r.float().sum().data[0]
        acc = float(acc) / (40*batch_size)
	
        
        #acc = average_precision_score(target.data.cpu().numpy(),output.data.cpu().numpy())
        train_loss += loss.data[0]
        loss.backward()
        optimizer.step()
        if batch_idx % batch_size == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f} \t Acc: {:.6f}'.format(
                epoch, batch_idx*len(image), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]/len(image), acc
		))
    print ('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))
            
def test(epoch):
    test_loss = 0
    #train_loss = 0
    batch_size = 128
    for batch_idx, data in enumerate(test_loader):
        #transform the data
        #pdb.set_trace()
        image = Variable(data['image'].float().view(-1,1,128,128))
        target = Variable(data['label'])
        if torch.cuda.is_available():
            image = image.cuda()
            target = target.cuda()
        #pdb.set_trace()
        #optimizer.zero_grad()
        #get get output from network
        _, output = Light_CNN_29(image)

        loss = BCEWithLogitsLoss(output,target)
	
        predict = nn.functional.sigmoid(output)>0.5
        r = (predict==target.byte())
        acc = r.float().sum().data[0]
        acc = float(acc) / (40*batch_size)
	
        
        #acc = average_precision_score(target.data.cpu().numpy(),output.data.cpu().numpy())
        test_loss += loss.data[0]
        #loss.backward()
        #optimizer.step()
    print ('====> Epoch: {} Average loss : {:.4f} Acc : {:.4f}'.format(
        epoch, test_loss/ len(test_loader.dataset), acc
    ))


        


if __name__ == '__main__':
    #pdb.set_trace()    
    #init the model as a 29 Layers one and print model
    Light_CNN_29 = LightCNN_29Layers()
    Light_CNN_29.eval()

    #pre_trained_model is trained on GPU
    if torch.cuda.is_available():
        Light_CNN_29 = nn.DataParallel(Light_CNN_29).cuda()

    else:
        print ('cuda is not available, cannot load the model')
    #pdb.set_trace()
    #load pre_trained_modenl
    pretrained_dict = torch.load('/home/ubuntu/wkhan/dataset/LightCNN_29Layers_checkpoint.pth.tar')
    Light_CNN_29 = load_part_model(Light_CNN_29,pretrained_dict)

    #choose the optimizer
    for para in list(Light_CNN_29.parameters())[:-2]:
        para.requires_grad = False
    #pdb.set_trace()
    optimizer = optim.Adam(params=[Light_CNN_29.module.fc3.weight,Light_CNN_29.module.fc3.bias],lr = 2e-4)

    #load the dataset
    dataloader, train_loader, test_loader = celeba_dataset()
    
    num_epochs = 10
    #train the model
    BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
    for epoch in range(num_epochs):
        #pdb.set_trace()
        train(epoch)
        test(epoch)

        






