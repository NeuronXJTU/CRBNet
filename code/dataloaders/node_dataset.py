import torch
import random
import itertools
import numpy as np
from PIL import Image
from scipy import ndimage
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from scipy.ndimage.interpolation import zoom
import cv2

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample

class BaseDataSets(Dataset):
    def __init__(self, base_dir= None, split= 'train', num= None, transform= None, data_name= None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.data_name = data_name

        if self.split == 'train':
            with open(self._base_dir + '/train_fold1.txt', 'r') as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('\n', '')
                                for item in self.sample_list]

        elif self.split == 'val':
            with open(self._base_dir + '/test_fold1.txt', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '')
                                for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            # case_temp = case.replace('.png', '.jpg')
            case_temp = case.replace('.png', '.png')
            image = Image.open(self._base_dir + "/imgs/{}".format(case_temp)).convert('L')
            image = np.array(image) / 255.0
            label = Image.open(self._base_dir + "/masks/{}".format(case)).convert('L')
            label = np.array(label)
            label[label > 0] = 1
        else:
            # case_temp = case.replace('.png', '.jpg')
            case_temp = case.replace('.png', '.png')
            image = Image.open(self._base_dir + "/imgs/{}".format(case_temp)).convert('L')
            image = np.array(image) / 255.0
            label = Image.open(self._base_dir + "/masks/{}".format(case)).convert('L')
            label = np.array(label)
            label[label > 0]=1

        sample = {'image': image, 'label': label.astype(np.uint8)}
        
        if self.transform:
            sample = self.transform(sample)

        sample["idx"] = idx
        return sample