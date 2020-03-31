import torch.utils.data as data
import numpy as np
import pandas as pd
import torch
import cv2
from albumentations import (CLAHE, RandomRotate90, ShiftScaleRotate, Blur,  IAAAdditiveGaussianNoise, GaussNoise,
                            MotionBlur, MedianBlur, RandomBrightnessContrast, IAASharpen, IAAEmboss, Flip, OneOf,
                            Compose, Resize, Normalize)


labels_dict = {'leaf_rust': 0,
               'stem_rust': 1,
               'healthy_wheat': 2}


def strong_aug(p=1):
    return Compose([
        OneOf([
            RandomRotate90(p=1),
            Flip(p=1),
        ], p=1),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=1, value=0, border_mode=2),
        OneOf([
            IAAAdditiveGaussianNoise(p=0.7),
            GaussNoise(p=0.7),
        ], p=1),
        OneOf([
            MotionBlur(p=0.7),
            MedianBlur(blur_limit=3, p=0.7),
            Blur(blur_limit=3, p=0.7),
        ], p=1),
        RandomBrightnessContrast(p=0.5),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(p=0.7),
        ], p=1)
    ], p=p)


def light_aug(p=1):
    return Compose([
        OneOf([
            RandomRotate90(p=0.7),
            Flip(p=0.7),
        ], p=1),
        ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=45, p=0.7, value=0, border_mode=2),
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.08, p=0.7)
    ], p=p)


def preproc(size=380):
    return Resize(size, size, p=1)


class WheatRust(data.Dataset):
    def __init__(self, names, size=380, augmentation='none', mode='default', pweight=0, lweight=False):
        super(WheatRust, self).__init__()

        self.mode = mode
        self.names = names
        self.augmentation = augmentation
        # self.labels = pd.read_csv('Data/origin_labels_w.csv', index_col=0)
        self.labels = pd.read_csv('Data/labels_cluster.csv', index_col=0)
        self.plabels = pd.read_csv('Data/confidant_pseudolabels.csv')
        self.test_dir = 'Data/test/'
        self.train_dir = 'Data/train_resized/'
        self.size = size
        self.pweight = pweight
        self.lweight = lweight

    def __getitem__(self, index):
        name = self.names[index]

        if self.mode == 'default':
            imdir = self.train_dir
            # c = self.labels[self.labels['ID'] == name]['class'].item()
            c = self.labels[self.labels['name'] == name]['label'].item() - 1
            if self.lweight:
                weight = self.labels[self.labels['ID'] == name]['weights'].item()
            else:
                weight = 1.0
            label = [0, 0, 0]
            label[c] = 1
        elif self.mode == 'pseudo':
            imdir = self.test_dir
            label = self.plabels[self.plabels['ID'] == name][['leaf_rust', 'stem_rust', 'healthy_wheat']].values[0].tolist()
            weight = self.pweight
        elif self.mode == 'binary':
            imdir = self.train_dir
            c = self.labels[self.labels['ID'] == name]['class'].item()
            if self.lweight:
                weight = self.labels[self.labels['ID'] == name]['weights'].item()
            else:
                weight = 1.0
            if c < 2:
                c = 0
            elif c == 2:
                c = 1
            label = [0, 0]
            label[c] = 1
        elif self.mode == 'binary_rust':
            imdir = self.train_dir
            # c = self.labels[self.labels['ID'] == name]['class'].item()
            c = self.labels[self.labels['name'] == name]['label'].item() - 1
            if self.lweight:
                weight = self.labels[self.labels['ID'] == name]['weights'].item()
            else:
                weight = 1.0
            label = [0, 0]
            if c < 2:
                label[c] = 1
        img = cv2.imread(imdir + name)

        if self.augmentation != 'none':
            if self.augmentation == 'strong':
                aug = strong_aug(p=1)
            elif self.augmentation == 'light':
                aug = light_aug(p=1)
            scope = aug(image=img)
            img = scope['image']

        transform = preproc(self.size)
        scope = transform(image=img)
        img = scope['image']
        img = np.moveaxis(np.array(img), 2, 0)
        img = img - np.mean(img)
        img = img / np.std(img)


        return torch.tensor(img, dtype=torch.float), \
               torch.tensor(label, dtype=torch.float), \
               torch.tensor(weight, dtype=torch.float)

    def __len__(self):
        return int(len(self.names))


class TestSet(data.Dataset):
    def __init__(self, names, size=380):
        super(TestSet, self).__init__()

        self.names = names
        self.test_dir = 'Data/test/'
        self.size = size

    def __getitem__(self, index):
        name = self.names[index]
        img = cv2.imread(self.test_dir + name)
        transform = preproc(self.size)
        scope = transform(image=img)
        img = scope['image']
        img = np.moveaxis(np.array(img), 2, 0)
        img = img - np.min(img)
        img = img / np.max(img)

        return torch.tensor(img, dtype=torch.float)

    def __len__(self):
        return int(len(self.names))
