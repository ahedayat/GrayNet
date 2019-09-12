import random
import torch

import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import DataLoader, Dataset


class GrayLoader(Dataset):
    def __init__(self, root_path, data_transform=None, target_transform=None, shuffle_data=True):
        self.data_path = '{}/images'.format(root_path)
        self.file_names = [line.rstrip('\n')
                           for line in open('{}/file_names.txt'.format(root_path))]

        self.data_transform = data_transform
        self.target_transform = target_transform
        self.make_gray = transforms.Grayscale(num_output_channels=1)

        if shuffle_data == True:
            random.shuffle(self.file_names)

    def __getitem__(self, ix):
        file_name = self.file_names[ix]
        colored_image = Image.open('{}/{}'.format(self.data_path, file_name))
        gray_image = Image.open('{}/{}'.format(self.data_path, file_name))
        if self.data_transform is not None:
            colored_image = self.data_transform(colored_image)

        if self.target_transform is not None:
            gray_image = self.target_transform(gray_image)

        gray_image = self.make_gray(gray_image)

        colored_image = np.array(colored_image)
        gray_image = np.array(gray_image)

        colored_image = torch.tensor(colored_image, dtype=torch.float)
        gray_image = torch.tensor(gray_image, dtype=torch.float)

        return colored_image, gray_image

    def __len__(self):
        return len(self.file_names)
