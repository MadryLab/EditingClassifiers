import torch.utils.data as data
from torch.utils.data import Dataset
from torchvision import transforms
from robustness.tools.folder import has_file_allowed_extension, IMG_EXTENSIONS, default_loader

from PIL import Image

import os
import os.path
import sys


def make_dataset(dir, img_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    
    for img_idx, target in img_to_idx.items():
        d = os.path.join(dir, str(target))
        assert os.path.isdir(d)
        
        path = os.path.join(d, f'img_{img_idx:05d}.png')
        if not os.path.exists(path):
            path = os.path.join(d, f'img_{img_idx}.png')
            assert os.path.exists(path)
            
        item = (path, img_idx)
        images.append(item)

    return images


class DatasetFolder(data.Dataset):

    def __init__(self, root, loader, extensions, transform=None,
                 target_transform=None, img_mapping=None):

        samples = make_dataset(root, img_mapping)

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class ImageFolder(DatasetFolder):
    def __init__(self, root, transform=None, 
                 target_transform=None,
                 loader=default_loader,
                 img_mapping=None):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform,
                                          img_mapping=img_mapping)