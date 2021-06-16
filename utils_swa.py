import os
import time 
import torch
import random
import shutil
import numpy as np  
import torch.nn as nn 
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from advertorch.attacks import LinfPGDAttack, L2PGDAttack
from advertorch.context import ctx_noparamgrad
from advertorch.utils import NormalizeByChannelMeanStd
# from datasets import *
# from models.preactivate_resnet import *
# from models.vgg import *
# from models.wideresnet import *

__all__ = ['save_checkpoint', 'setup_dataset_models', 'setup_dataset_models_standard', 'setup_seed', 'moving_average', 'bn_update', 'print_args',
            'train_epoch', 'train_epoch_adv', 'train_epoch_adv_dual_teacher',
            'test', 'test_adv']

def save_checkpoint(state, is_SA_best, is_RA_best, is_SA_best_swa, is_RA_best_swa, save_path, filename='checkpoint.pth.tar'):
    
    filepath = os.path.join(save_path, filename)
    torch.save(state, filepath)

    if is_SA_best_swa:
        shutil.copyfile(filepath, os.path.join(save_path, 'model_SWA_SA_best.pth.tar'))
    if is_RA_best_swa:
        shutil.copyfile(filepath, os.path.join(save_path, 'model_SWA_RA_best.pth.tar'))
    if is_SA_best:
        shutil.copyfile(filepath, os.path.join(save_path, 'model_SA_best.pth.tar'))
    if is_RA_best:
        shutil.copyfile(filepath, os.path.join(save_path, 'model_RA_best.pth.tar'))


# prepare dataset and models
# def setup_dataset_models(args):

#     # prepare dataset
#     if args.dataset == 'cifar10':
#         classes = 10
#         dataset_normalization = NormalizeByChannelMeanStd(
#             mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
#         train_loader, val_loader, test_loader = cifar10_dataloaders(batch_size = args.batch_size, data_dir = args.data)
    
#     elif args.dataset == 'cifar100':
#         classes = 100
#         dataset_normalization = NormalizeByChannelMeanStd(
#             mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
#         train_loader, val_loader, test_loader = cifar100_dataloaders(batch_size = args.batch_size, data_dir = args.data)
    
#     elif args.dataset == 'tinyimagenet':
#         classes = 200
#         dataset_normalization = NormalizeByChannelMeanStd(
#             mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
#         train_loader, val_loader, test_loader = tiny_imagenet_dataloaders(batch_size = args.batch_size, data_dir = args.data)
    
#     else:
#         raise ValueError("Unknown Dataset")

#     #prepare model

#     if args.arch == 'resnet18':
#         model = ResNet18(num_classes = classes)
#         model.normalize = dataset_normalization

#         if args.swa:
#             swa_model = ResNet18(num_classes = classes)
#             swa_model.normalize = dataset_normalization
#         else:
#             swa_model = None

#         if args.lwf:
#             teacher1 = ResNet18(num_classes = classes)
#             teacher1.normalize = dataset_normalization
#             teacher2 = ResNet18(num_classes = classes)
#             teacher2.normalize = dataset_normalization
#         else:
#             teacher1 = None
#             teacher2 = None 

#     elif args.arch == 'wideresnet':
#         model = WideResNet(args.depth_factor, classes, widen_factor=args.width_factor, dropRate=0.0)
#         model.normalize = dataset_normalization

#         if args.swa:
#             swa_model = WideResNet(args.depth_factor, classes, widen_factor=args.width_factor, dropRate=0.0)
#             swa_model.normalize = dataset_normalization
#         else:
#             swa_model = None

#         if args.lwf:
#             teacher1 = WideResNet(args.depth_factor, classes, widen_factor=args.width_factor, dropRate=0.0)
#             teacher1.normalize = dataset_normalization
#             teacher2 = WideResNet(args.depth_factor, classes, widen_factor=args.width_factor, dropRate=0.0)
#             teacher2.normalize = dataset_normalization
#         else:
#             teacher1 = None
#             teacher2 = None 

#     elif args.arch == 'vgg16':
#         model = vgg16_bn(num_classes = 10)
#         model.normalize = dataset_normalization

#         if args.swa:
#             swa_model = vgg16_bn(num_classes = 10)
#             swa_model.normalize = dataset_normalization
#         else:
#             swa_model = None

#         if args.lwf:
#             teacher1 = vgg16_bn(num_classes = 10)
#             teacher1.normalize = dataset_normalization
#             teacher2 = vgg16_bn(num_classes = 10)
#             teacher2.normalize = dataset_normalization
#         else:
#             teacher1 = None
#             teacher2 = None 

#     else:
#         raise ValueError("Unknown Model")   
    
#     return train_loader, val_loader, test_loader, model, swa_model, teacher1, teacher2

# def setup_dataset_models_standard(args):

#     # prepare dataset
#     if args.dataset == 'cifar10':
#         classes = 10
#         dataset_normalization = NormalizeByChannelMeanStd(
#             mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
#         train_loader, val_loader, test_loader = cifar10_dataloaders(batch_size = args.batch_size, data_dir = args.data)
    
#     elif args.dataset == 'cifar100':
#         classes = 100
#         dataset_normalization = NormalizeByChannelMeanStd(
#             mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
#         train_loader, val_loader, test_loader = cifar100_dataloaders(batch_size = args.batch_size, data_dir = args.data)
    
#     elif args.dataset == 'tinyimagenet':
#         classes = 200
#         dataset_normalization = NormalizeByChannelMeanStd(
#             mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
#         train_loader, val_loader, test_loader = tiny_imagenet_dataloaders(batch_size = args.batch_size, data_dir = args.data)
    
#     else:
#         raise ValueError("Unknown Dataset")

#     #prepare model

#     if args.arch == 'resnet18':
#         model = ResNet18(num_classes = classes)
#         model.normalize = dataset_normalization

#     elif args.arch == 'wideresnet':
#         model = WideResNet(args.depth_factor, classes, widen_factor=args.width_factor, dropRate=0.0)
#         model.normalize = dataset_normalization

#     elif args.arch == 'vgg16':
#         model = vgg16_bn(num_classes = 10)
#         model.normalize = dataset_normalization

#     else:
#         raise ValueError("Unknown Model")   
    
#     return train_loader, val_loader, test_loader, model

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def setup_seed(seed): 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 

# knowledge distillation loss function
def loss_fn_kd(scores, target_scores, T=2.):
    """Compute knowledge-distillation (KD) loss given [scores] and [target_scores].

    Both [scores] and [target_scores] should be tensors, although [target_scores] should be repackaged.
    'Hyperparameter': temperature"""

    device = scores.device

    log_scores_norm = F.log_softmax(scores / T, dim=1)
    targets_norm = F.softmax(target_scores / T, dim=1)

    # if [scores] and [target_scores] do not have equal size, append 0's to [targets_norm]
    if not scores.size(1) == target_scores.size(1):
        print('size does not match')

    n = scores.size(1)
    if n>target_scores.size(1):
        n_batch = scores.size(0)
        zeros_to_add = torch.zeros(n_batch, n-target_scores.size(1))
        zeros_to_add = zeros_to_add.to(device)
        targets_norm = torch.cat([targets_norm.detach(), zeros_to_add], dim=1)

    # Calculate distillation loss (see e.g., Li and Hoiem, 2017)
    KD_loss_unnorm = -(targets_norm * log_scores_norm)
    KD_loss_unnorm = KD_loss_unnorm.sum(dim=1)                      #--> sum over classes
    KD_loss_unnorm = KD_loss_unnorm.mean()                          #--> average over batch

    # normalize
    KD_loss = KD_loss_unnorm * T**2

    return KD_loss

def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha

def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True

def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]

def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)

def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum

def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]

def bn_update(loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.

        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input, _ in loader:
        input = input.cuda()
        b = input.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))
