'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os 
import sys 
import numpy as np 
import random 
import torch
import torch.nn as nn
from models import *
from torchvision.models import resnet50, alexnet
import torchvision
import torchvision.transforms as transforms


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def get_testloader(dataset):
    if dataset =='imagenet':
        path = './images_val'
        image_size = 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)
        transform = transforms.Compose([transforms.Resize(size=(image_size,image_size)),
                                        transforms.ToTensor(),
                                        normalize])
        dataset = torchvision.datasets.ImageFolder(root=path, transform=transform)
        testloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
         
    elif dataset == 'cifar10':
        image_size = 32
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        tranfrom_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        ])
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=tranfrom_test)
        testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=2)
    return testloader, image_size, mean, std

def get_model(dataset, model):
    if dataset == 'imagenet':
        model = eval('{}(pretrained=True)'.format(model))
        model = model.cuda()
        model.eval()
    elif dataset == 'cifar10':
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/%s.t7'%model)
        model = checkpoint['net']
        model = model.cuda() 
        model.eval() 
    return model 





def group(distance, radius, confidence):
    '''
    distance: distance pairs NxN matrix 
    radius: group radius
    '''
    candidate = np.arange(len(distance))
    group = []
    i = 0
    while len(candidate) != 0:
        i=i+1
        # print (candidate)
        best_ind = np.argmax(confidence)
        index_best = candidate[best_ind]
        cluster = np.where(distance[index_best]<=radius)[0]
        candidate = np.delete(candidate, cluster, axis=-1)
        confidence = np.delete(confidence, cluster, axis=-1)
        distance = np.delete(distance, cluster, axis=-1)
        group.append(index_best)
    # print (len(group))
    # print (len(distance))
    return group


class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath): 
        self.file = None
        self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume: 
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()

    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            if isinstance(num, str):
                self.file.write("{}".format(num))
            else:
                self.file.write("{0:.1f}".format(num))
            self.file.write('\t\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()
    def close(self):
        if self.file is not None:
            self.file.close()


class BatchDeNormalize(object):
    def __init__(self, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t in tensor:
            t.mul_(self.std.reshape(shape=(3, 1, 1))).add_(self.mean.reshape(shape=(3, 1, 1)))
        return tensor