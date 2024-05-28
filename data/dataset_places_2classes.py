import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from math import floor
from random import shuffle


def get_transform(input_size=224):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

class PlacesDataset(Dataset):
    def __init__(self,
                 dataset_npy: str,
                 mode: str = 'train',
                 perc_train: float = 1.0):

        self.data = {}
        self.labels = {}
        self.classes = {
            'bedroom': 0,
            'childs_room': 1,
        }
        self.idx_to_class = {str(idx): value for idx, value in enumerate(self.classes.keys())}
        self.mode = mode
        self.transform = get_transform()
        
        reader = np.load(dataset_npy)
        for [img_path, label, group] in reader:
            # label == classe idx
            if label in self.data.keys():
                self.data[label].append(img_path)
                self.labels[label].append([label, group])
            else:
                self.data[label] = [img_path]
                self.labels[label] = [[label, group]]
        if mode == 'train':
            for label in self.data.keys():
                self.data[label] = np.asarray(self.data[label][:floor(perc_train*len(self.data[label]))])
                self.labels[label] = np.asarray(self.labels[label][:floor(perc_train*len(self.labels[label]))])
        if mode == 'val':
            for label in self.data.keys():
                self.data[label] = np.asarray(self.data[label][floor(perc_train*len(self.data[label])):])
                self.labels[label] = np.asarray(self.labels[label][floor(perc_train*len(self.labels[label])):])

        new_data = []
        assert self.data['0'].shape[0] == self.labels['0'].shape[0] 
        assert self.data['1'].shape[0] == self.data['1'].shape[0]
        
        part1 = np.column_stack([self.data['0'],self.labels['0']])
        part2 = np.column_stack([self.data['1'],self.labels['1']])
        self.data = np.concatenate([part1, part2])
        print(self.data.shape)
        
    def __getitem__(self, index: int):
        img_path, label, group = self.data[index]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        label = torch.tensor(int(label))
        group = torch.tensor(int(group))
        return image, label, group, img_path

    def __len__(self):
        return len(self.data)
