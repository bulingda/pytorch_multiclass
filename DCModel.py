"""训练"""
import torch.nn as nn
from torchvision import models
import time
import copy
import glob
import argparse
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
import torch.nn.functional as F
import numpy as np


from DCDataset import DCDataset
plt.ion()
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

class_names = ['mixed_type', 'no', 'pigmented_type', 'structural_type', 'vascular_type']

# data_path = r'A:\picture\blackeye\dark_circle'


def args_parsers():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--data_path', type=str, default=r'A:\picture\blackeye\dark_circle', dest='dp', help='')
    parser.add_argument('--batch_size', type=int, default=128, dest='bs', help='input batch size for training(default:1)')
    parser.add_argument('--epochs', type=int, default=30, dest='epochs', help='number of epochs to train(defaut:20)')
    parser.add_argument('--lr', type=float, default=0.001, dest='lr', help='learning rate(default:0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, dest='momentum', help='Adam momentum(default:0.9)')
    parser.add_argument('--step_size', type=int, default=7, dest='step_size', help='StepLR step_size(default:7)')
    parser.add_argument('--num_worked', type=int, default=4, dest='nw', help='num_worked of dataset(default:4)')
    return parser.parse_args()


# 支持多分类和二分类
class FocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. '
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)

        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class pmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained_model = models.resnet18(pretrained=True)
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        dim = self.pretrained_model.fc.in_features
        self.pretrained_model.fc = nn.Linear(dim, 5)

    def forward(self, img):
        feat = self.pretrained_model(img)
        return feat


def train_model(dataloaders, model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders):
                if is_cuda:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if i % 100 == 0:
                    print(loss.item())
            # print(len(dataloaders.dataset))
            # print(running_loss, running_corrects.item())
            epoch_loss = running_loss / len(dataloaders.dataset)
            epoch_acc = running_corrects.item()/ len(dataloaders.dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        scheduler.step()
        # load best model weights
        model.load_state_dict(best_model_wts)
        torch.save({
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer_ft.state_dict(),
            'best_acc': best_acc
        }, r'model_Adam_Focalloss_steplr_bs{}_lr{}_acc{:.6f}.pt'.format(args.bs, args.lr, best_acc))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return model


# def imshow(inp, title=None):
#     """Imshow for Tensor."""
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)  # pause a bit so that plots are updated
#

# # Get a batch of training data
# inputs, classes = next(iter(dataloaders['train']))
#
# # Make a grid from batch
# out = torchvision.utils.make_grid(inputs)
#
# imshow(out, title=[class_names[x] for x in classes])

#
# def visualize_model(dataloaders, model, num_images=6):
#     was_training = model.training
#     model.eval()
#     images_so_far = 0
#     fig = plt.figure()
#
#     with torch.no_grad():
#         for i, (inputs, labels) in enumerate(dataloaders['val']):
#             if is_cuda:
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)
#
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#
#             for j in range(inputs.size()[0]):
#                 images_so_far += 1
#                 ax = plt.subplot(num_images//2, 2, images_so_far)
#                 ax.axis('off')
#                 ax.set_title('predicted: {}'.format(class_names[preds[j]]))
#                 imshow(inputs.cpu().data[j])
#
#                 if images_so_far == num_images:
#                     model.train(mode=was_training)
#                     return
#         model.train(mode=was_training)


if __name__ == '__main__':
    args = args_parsers()
    transformers = transforms.Compose([
        transforms.Resize((128, 128)),
        # transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        # transforms.RandomVerticalFlip(p=0.5),
        # transforms.RandomRotation(degrees=20, expand=True),  # 正负20度之间旋转，填满整张图片
        # transforms.ColorJitter(0.3, 0.3, 0.3),
        # transforms.RandomGrayscale(0.1),
        # transforms.RandomAffine(degrees=30),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = DCDataset(args.dp, transformers)

    weights = torch.FloatTensor([73*13*42, 170*42*13, 170*73*13, 170*73*42, 73*13*42])
    sampler = WeightedRandomSampler(weights, num_samples=10, replacement=True)
    dataloader = DataLoader(dataset, args.bs, num_workers=args.nw, shuffle=True, drop_last=True, sampler=sampler)  # , sampler
    pre_model = pmodel()

    is_cuda = False
    if is_cuda:
        pre_model = pre_model.to(device)

    # criterion = nn.CrossEntropyLoss()
    criterion = FocalLoss(5, alpha=[1, 2, 4, 16, 1], gamma=2)
    # nn.F

    optimizer_ft = torch.optim.SGD(pre_model.parameters(), lr=args.lr, momentum=args.momentum)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=args.step_size, gamma=0.1)

    pretrained_model = train_model(dataloader, pre_model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=args.epochs)

