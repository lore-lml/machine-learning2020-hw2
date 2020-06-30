import os
import os.path
import pandas as pd

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class CaltechSubset(Dataset):

    def __init__(self, images, labels, transforms):
        assert len(images) == len(labels)
        self.images = list(images)
        self.labels = list(labels)
        self.transforms = transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img, label = self.images[index], self.labels[index]

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label



class Caltech(VisionDataset):
    def __init__(self, root, src='train', transform=None, eval_transform=None):
        super(Caltech, self).__init__(root, transform=transform)

        self.root = root
        self.split = src # This defines the src you are going to use
                           # (src files are called 'train.txt' and 'test.txt')
        self.transform = transform
        self.eval_transform = eval_transform

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''

        split_path = "Caltech101/train.txt" if src == "train" else "Caltech101/test.txt"
        self.dir_images = []
        classes = []
        images = []
        with open(split_path) as reader:
            for line in reader:
                if not line.startswith("BACKGROUND_Google"):
                    classes.append(line.split("/")[0])
                    image_path = os.path.join(self.root, line[:-1])
                    self.images.append(pil_loader(image_path))

        self.class_names = {k: i for i, k in enumerate(sorted(set(self.labels)))}
        labels_int = [self.class_names[v] for v in self.labels]

        self.df = pd.DataFrame({
            'image': pd.Series(list(images)),
            'label': labels_int,
        })

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        # Provide a way to access image and label via index
        # Image should be a PIL Image
        # label can be int
        image, label = self.df.loc[index, 'image'], self.df.loc[index, 'label']

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        # Provide a way to get the length (number of elements) of the dataset
        return self.df.shape[0]

    def get_labels(self):
        return np.array(self.df.loc[:, 'label'])

    def get_train_validation_set(self, train_size=.5):
        train_idx, val_idx = train_test_split(np.arange(self.__len__()), train_size=train_size, stratify=self.df.loc[:, 'label'])
        train_subset = CaltechSubset(self.df.loc[train_idx, 'image'], self.df.loc[train_idx, 'label'], self.transform)
        val_subset = CaltechSubset(self.df.loc[val_idx, 'image'], self.df.loc[val_idx, 'label'], self.eval_transform)
        return train_subset, val_subset