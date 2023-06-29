#!/usr/bin/env	python3
""" train network using pytorch
"""
import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights
import cv2

save_path = r"images/"

def train(epoch,method):
    start = time.time()
    net.train()
    k = 1
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()
        optimizer.zero_grad()
        def method_criterion(criterion, pred, y_a, y_b, lam):
            return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

        if method == 'mixup':
            def mixup(x, y, alpha=1, use_cuda=True):
                if alpha > 0:
                    lam = np.random.beta(alpha, alpha)
                else:
                    lam = 1
                batch_size = x.size()[0]
                if use_cuda:
                    index = torch.randperm(batch_size).cuda()
                else:
                    index = torch.randperm(batch_size)
                mixed_x = lam * x + (1 - lam) * x[index, :]
                y_a, y_b = y, y[index]
                return mixed_x, y_a, y_b, lam
            inputs, targets_a, targets_b, lam = mixup(images, labels,
                                                                args.alpha, use_cuda=True)
            #下面代码是用来挑选三张示范图片  选好就可以注释掉了
            '''
            if k <= 3:
                #import pdb;pdb.set_trace()
                save_img = np.asarray(inputs[5].permute(1,2,0).cpu())
                save_img = 1.0 / (1 + np.exp(-1 * save_img))
                save_img = np.round(save_img * 255)
                cv2.imwrite(save_path + 'mixup_{}.png'.format(k), save_img)
                print(k)
                k += 1
            '''
            outputs = net(inputs)
            loss = method_criterion(loss_function, outputs, targets_a, targets_b, lam)

        elif method == 'cutmix':
            def rand_bbox(size, lam):
                W = size[2]
                H = size[3]
                cut_rat = np.sqrt(1. - lam)
                cut_w = np.int(W * cut_rat)
                cut_h = np.int(H * cut_rat)
                # uniform
                cx = np.random.randint(W)
                cy = np.random.randint(H)
                bbx1 = np.clip(cx - cut_w // 2, 0, W)
                bby1 = np.clip(cy - cut_h // 2, 0, H)
                bbx2 = np.clip(cx + cut_w // 2, 0, W)
                bby2 = np.clip(cy + cut_h // 2, 0, H)
                return bbx1, bby1, bbx2, bby2

            def cutmix(x, y, alpha=1, use_cuda=True):
                '''Returns mixed inputs, pairs of targets, and lambda'''
                # r = np.random.rand(1)   and r < args.cutmix_prob
                if alpha > 0:
                    lam = np.random.beta(alpha, alpha)
                else:
                    lam = 1
                batch_size = x.size()[0]
                if use_cuda:
                    index = torch.randperm(batch_size).cuda()
                else:
                    index = torch.randperm(batch_size)
                y_a, y_b = y, y[index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
                x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
                return x, y_a, y_b, lam

            inputs, targets_a, targets_b, lam = cutmix(images, labels,
                                                      args.alpha, use_cuda=True)
            #       inputs, targets_a, targets_b = map(Variable, (inputs,targets_a, targets_b))
            #下面代码是用来挑选三张示范图片
            '''
            if k <= 3:
                #import pdb;pdb.set_trace()
                save_img = np.asarray(inputs[5].permute(1,2,0).cpu())
                save_img = 1.0 / (1 + np.exp(-1 * save_img))
                save_img = np.round(save_img * 255)
                cv2.imwrite(save_path + 'cutmix_{}.png'.format(k), save_img)
                print(k)
                k += 1
            '''
            outputs = net(inputs)
            loss = method_criterion(loss_function, outputs, targets_a, targets_b, lam)

        elif method == 'cutout' or args.method == 'baseline':
            #下面代码是用来挑选三张示范图片
            '''
            if k <= 3:
                #import pdb;pdb.set_trace()
                save_img = np.asarray(images[5].permute(1,2,0).cpu())
                save_img = 1.0 / (1 + np.exp(-1 * save_img))
                save_img = np.round(save_img * 255)
                cv2.imwrite(save_path + 'cutout_{}.png'.format(k), save_img)
                print(k)
                k += 1
            '''
            outputs = net(images)
            loss = loss_function(outputs, labels)

        loss.backward()
        optimizer.step()
        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1
        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)
        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar100_training_loader.dataset)
        ))
        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)
        if epoch <= args.warm:
            warmup_scheduler.step()

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)
    finish = time.time()
    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(epoch=0, tb=True):
    start = time.time()
    net.eval()
    test_loss = 0.0 # cost function error
    correct = 0.0
    for (images, labels) in cifar100_test_loader:
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
    finish = time.time()
    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset),
        finish - start
    ))
    print()
    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)
    return correct.float() / len(cifar100_test_loader.dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-method', default='baseline',
                        choices=['cutmix', 'cutout', 'mixup', 'baseline'])
    parser.add_argument('-alpha', default=1, type=float,
                        help='mixup interpolation coefficient (default: 1)')
    parser.add_argument('-cutmix_prob', default=0.5, type=float, help='probility of CutMix')
    parser.add_argument('-n_holes', type=int, default=1,
                        help='number of holes to cut out from image')
    parser.add_argument('-length', type=int, default=16,
                        help='length of the holes')
    args = parser.parse_args()
    net = get_network(args)

    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True,
        method = args.method,
        n_holes = args.n_holes,
        length = args.length

    )
    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net, args.method), fmt=settings.DATE_FORMAT)   #因为最近一次create的一定是最佳的
        #print("撒啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊",recent_folder)
        if not recent_folder:
            raise Exception('no recent folder were found')
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, args.method, recent_folder)
    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, args.method, settings.TIME_NOW)
    #print("啊啊cccc啊啊啊啊啊啊啊啊",checkpoint_path)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 3, 32, 32)
    if args.gpu:
        input_tensor = input_tensor.cuda()
    writer.add_graph(net, input_tensor)

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{epoch}-{type}.pth')
    #print("对了吗",checkpoint_path)
    
    best_acc = 0.0
    if args.resume: 
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, args.method, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, args.method, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, args.method, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, args.method, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))
        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, args.method, recent_folder))

    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)
        if args.resume:
            if epoch <= resume_epoch:
                continue
        train(epoch,args.method)
        acc = eval_training(epoch)

        #start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path.format(epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:  #就是保存下来的 但是压根没用
            weights_path = checkpoint_path.format(epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
    writer.close()
