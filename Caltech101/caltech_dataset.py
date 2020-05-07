from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys
from random import randint
import numpy as np

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def random_split_indices(dataset, train_size=.5):
    validation_size = int(dataset.__len__() * (1-train_size))
    train_indices = list(range(dataset.__len__()))
    validation_indices = []
    for i in range(validation_size):
        ind = randint(0, len(train_indices))
        validation_indices.append(train_indices.pop(ind))

    return train_indices, validation_indices


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.root = root
        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')
        self.transform = transform
        self.target_transform = target_transform

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''

        split_path = "Caltech101/train.txt" if split == "train" else "Caltech101/test.txt"
        self.dir_images = []
        self.labels = []
        self.images = []
        with open(split_path) as reader:
            for line in reader:
                if not line.startswith("BACKGROUND_Google"):
                    self.labels.append(line.split("/")[0])
                    image_path = os.path.join(self.root, line[:-1])
                    self.images.append(pil_loader(image_path))

        self.class_names = {k: i for i, k in enumerate(sorted(set(self.labels)))}
        self.labels_int = [self.class_names[v] for v in self.labels]

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
        image, label = self.images[index], self.labels_int[index]

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
        return len(self.labels)

    def get_labels(self):
        return np.array(self.labels_int)
