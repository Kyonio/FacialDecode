from __future__ import print_function, division
import torch
#import pandas as import pd
from skimage import io,transform,color
import matplotlib.pyplot as plt
import pdb
from torch.utils.data import Dataset,DataLoader,sampler
from torchvision import transforms, utils
import random
import os
import numpy as np
from math import floor

class Rescale(object):
    
    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        self.output_size = output_size
    
    def __call__(self,sample):
        image = sample['image']

        h,w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h>w:
                new_h, new_w = self.output_size*h/w, self.output_size
            else:
                new_h, new_w = self.output_size,self.output_size*w/h
        else:
            new_h,new_w = self.output_size
        
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image,(new_h,new_w))

        return {'image':img}

class ToGray(object):
    
    def __call__(self,sample):
        image = sample['image']
        image = color.rgb2gray(image)
        return {'image':image}


class ToTensor(object):

    def __call__(self,sample):
        image = sample['image']
        #image = image.transpose((2,0,1))
        return {'image':torch.from_numpy(image)}    




def read_label(label_path):
    with open(label_path) as myfile:
        data = myfile.readlines()

    dict_label = {}
    for line in data:
        if '.jpg' in line:
            l = []
            for t in line[10:].split():
                try:
                    l.append(float(t))
                except:
                    pass
            dict_label[line[:10]] = l
    return dict_label



def CellActi(cell_path):
    i = 0
    for file_name in os.listdir(cell_path):
        file_name = os.path.join(cell_path,file_name)
        #pdb.set_trace()
        file = np.loadtxt(file_name)
        if i>0:
            file = np.concatenate((pre_file,file))
        pre_file = file
        i+=1
        
    file = file.reshape(138,-1)
    
    image_attr_cell = np.transpose(file)
    cell_dict = {}
    for i in range(2162):
        cell_dict[str(i+1)+'.png'] = image_attr_cell[i]
    
    return cell_dict
    



class FacialDataset(Dataset):
    
    def __init__(self,root_dir,cell_dict,transform=None):
        #super(FacialDataset,self).__init__()
        
        
        self.data_files = os.listdir(root_dir)
        self.data_files.sort(key= lambda x: float(x.strip('.png')))
        self.root_dir = root_dir
        self.transform = transform
        self.cell_dict = cell_dict
        
    def __getitem__(self,idx):
        img_name = os.path.join(self.root_dir,self.data_files[idx])
        #pdb.set_trace()
        image = io.imread(img_name)
        cell_acti = self.cell_dict[self.data_files[idx]]
        cell_acti = torch.FloatTensor(cell_acti)
        sample = {'image':image}
        if self.transform:
            sample = self.transform(sample)
        sample['cell'] = cell_acti
        #sample['label'] = self.data_files[idx]
        return sample

    def __len__(self):
        return len(self.data_files)



class celebADataset(Dataset):
    def __init__(self,root_dir,dict_label,transform=None):
        self.data_files = os.listdir(root_dir)
        self.root_dir = root_dir
        self.transform = transform
        self.dict_label = dict_label
    
    def __getitem__(self,idx):
        img_name = os.path.join(self.root_dir,self.data_files[idx])
        image = io.imread(img_name)
        attr = self.dict_label[self.data_files[idx]]
        attr = torch.FloatTensor(attr)
        sample = {'image' : image}
        if self.transform:
            sample = self.transform(sample)
        sample['label'] = attr
        return sample

    def __len__(self):
        return len(self.data_files)


def split_dataset(dataset,test_size=0.8,shuffle=False,random_seed=0):
    length = len(dataset)
    indices = list(range(1,length))
    #pdb.set_trace()
    if shuffle == True:
        random.seed(random_seed)
        random.shuffle(indices)
    
    if type(test_size) is float:
        split = int(test_size*length)
    elif type(test_size) is int:
        split = test_size
    
    else:
        raise ValueError('%s should be an int or a float' %str)
    
    return indices[:split], indices[split:]

#pdb.set_trace()


def facial_dataset():
    dset = FacialDataset(root_dir='/home/ubuntu/wkhan/VAE/dataset/new_2k_face/',
                        cell_dict=CellActi('/home/ubuntu/wkhan/VAE/dataset/population_response/'),
                        transform = transforms.Compose([
                         Rescale((64,64)),
                         ToTensor()
                         ]))
    train_idx, test_idx = split_dataset(dset,0.8,shuffle=True)
    train_sampler = sampler.SubsetRandomSampler(train_idx)
    test_sampler = sampler.SubsetRandomSampler(test_idx)
    
    dataloader = DataLoader(dset,batch_size=128,shuffle=False)
    train_loader = DataLoader(dset,batch_size=128,sampler=train_sampler)
    test_loader = DataLoader(dset,batch_size=128,sampler=test_sampler)
    
    return dataloader,train_loader,test_loader



def celeba_dataset():
    dset = celebADataset(root_dir='/home/ubuntu/wkhan/VAE/dataset/img_align_celeba/',
                     dict_label = read_label('/home/ubuntu/wkhan/VAE/dataset/list_attr_celeba.txt'),
                     transform = transforms.Compose([
                         Rescale((64,64)),
                         ToGray(),
                         ToTensor()
                         ]))
    train_idx, test_idx = split_dataset(dset,0.8,shuffle=True)
    train_sampler = sampler.SubsetRandomSampler(train_idx)
    test_sampler = sampler.SubsetRandomSampler(test_idx)

    dataloader = DataLoader(dset, batch_size = 128, shuffle=False)
    train_loader = DataLoader(dset, batch_size=128, sampler=train_sampler)
    test_loader = DataLoader(dset, batch_size=128, sampler=test_sampler)

    return dataloader,train_loader,test_loader




'''
dset = celebADataset(root_dir='/home/hwk/dataset/img_align_celeba/',
                     dict_label = read_label('/home/hwk/dataset/list_attr_celeba.txt'),
                     transform = transforms.Compose([
                         Rescale((128,128)),
                         ToGray(),
                         ToTensor()
                         ]))
train_idx, test_idx = split_dataset(dset,0.8,shuffle=True)
train_sampler = sampler.SubsetRandomSampler(train_idx)
test_sampler = sampler.SubsetRandomSampler(test_idx)

dataloader = DataLoader(dset, batch_size = 128, shuffle=False)
train_loader = DataLoader(dset, batch_size=128, sampler=train_sampler)
test_loader = DataLoader(dset, batch_size=128, sampler=test_sampler)
for i_batch, sample_batched in enumerate(dataloader):
    print (i_batch, sample_batched['image'].size())
'''
