import torch
from qhoptim.pyt import QHAdam
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR, MultiStepLR
from torch.utils.data import WeightedRandomSampler
import numpy as np
import re
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def define_optimizer(model, args):
    if args.optimizer.startswith('adam'):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)
    elif args.optimizer.startswith('rmsprop'):
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()),
                                        lr=args.lr,
                                        weight_decay=args.weight_decay)
    elif args.optimizer.startswith('sgd'):
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=args.lr,
                                    weight_decay=args.weight_decay)
    elif args.optimizer.startswith('qhadam'):
        optimizer = QHAdam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.lr,
                           nus=[0.7, 1.0],
                           betas=[0.995, 0.999],
                           weight_decay=args.weight_decay)
    else:
        raise ValueError('Optimizer not supported')
    print('Optimizer: ', optimizer)
    return optimizer


def define_lr_scheduler(args, optimizer, loader):
    scheduler = None
    if args.scheduler == 'Cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=len(loader), eta_min=args.lr_end)
    elif args.scheduler == 'Cyclic':
        scheduler = CyclicLR(optimizer, base_lr=args.lr_end, max_lr=args.lr, step_size_up=len(loader) * 3 // 4,
                             step_size_down=len(loader) - len(loader) * 3 // 4)
    elif args.scheduler == 'MStep':
        scheduler = MultiStepLR(optimizer=optimizer, milestones=[5, 10, 15, 20])
    return scheduler


def get_class_weights(labels):
    labels = np.array(labels)
    weight = torch.tensor([1/np.sum(labels == i) for i in np.unique(labels)])
    weight = weight / weight.sum()
    weight = weight[torch.tensor(list(labels))]
    return weight


def define_sampler(args, weights):
    sampler_ = None
    shuffle_ = True
    if args.sampler == 'wrs':
        sampler_ = WeightedRandomSampler(weights, len(weights))
        shuffle_ = False

    return sampler_, shuffle_


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def plot_confusion_matrix(y_true, y_pred, classes=None,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(y_true, y_pred)
    if classes is None:
        classes = list(np.linspace(0, max(y_true), max(y_true)+1))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.axis('tight')
