import os
import sys
import re
import datetime
import numpy
import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from cutout import Cutout

def get_network(args):

    if args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    
    elif args.net == 'simplevit':
        from models.simplevit import simple_vit
        net = simple_vit()
        
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True,method = 'none',n_holes = 1,length = 16):
    
    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    if method == 'cutout':
        transform_train.transforms.append(Cutout(n_holes=n_holes, length=length))
        print("yes")
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def compute_mean_std(cifar100_dataset):
    
    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):

    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):

        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders  #有个时间只是先创建文件夹拿来放 不一定有pth  但有的话也只会有一个
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):

    weight_files_ = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = [w for w in weight_files_ if 'best' in w]
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[0]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    print(weight_file)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[0])

    return resume_epoch

def best_acc_weights(weights_folder):
    #print("aaaaaaaaa",weights_folder)
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''
    #print("aaaaaaaaaaa",files)
    regex_str = r'([0-9]+)-(regular|best)'
    best_files = [w for w in files if 'best' in w and re.search(regex_str, w).groups()[1] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[0]))
    return best_files[-1]