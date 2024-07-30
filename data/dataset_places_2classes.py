import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
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
        self.n_confounders = 2
        self.n_groups = 2
        self.n_classes = 2

        reader = np.load(dataset_npy)
        if self.mode in ['train', 'validation']:
            for [img_path, label, group] in reader:
                # label == classe idx
                if label in self.data.keys():
                    self.data[label].append(img_path)
                    self.labels[label].append([int(label), int(group)])
                else:
                    self.data[label] = [img_path]
                    self.labels[label] = [[int(label), int(group)]]
            
            if mode == 'train':
                for label in self.data.keys():
                    self.data[label] = self.data[label][:floor(perc_train*len(self.data[label]))]
                    self.labels[label] = self.labels[label][:floor(perc_train*len(self.labels[label]))]
            if mode == 'validation':
                for label in self.data.keys():
                    self.data[label] = self.data[label][floor(perc_train*len(self.data[label])):]
                    self.labels[label] = self.labels[label][floor(perc_train*len(self.labels[label])):]
        else:
            for [img_path, label] in reader:
                # label == classe idx
                if label in self.data.keys():
                    self.data[label].append(img_path)
                    self.labels[label].append(int(label))
                else:
                    self.data[label] = [img_path]
                    self.labels[label] = [int(label)]

        for label in self.data.keys():
            self.data[label] =  np.asarray(self.data[label])
            self.labels[label] =  np.asarray(self.labels[label])
        
        assert self.data['0'].shape[0] == self.labels['0'].shape[0] 
        assert self.data['1'].shape[0] == self.labels['1'].shape[0]        
        
        part1 = np.column_stack([self.data['0'], self.labels['0']])
        part2 = np.column_stack([self.data['1'], self.labels['1']])
        self.data = np.concatenate([part1, part2])

        if self.mode in ['train', 'validation']:
            group_array = [int(row[2]) for row in self.data]
            y_array = [int(row[1]) for row in self.data]
            self._group_array = torch.LongTensor(group_array)
            self._group_counts = (torch.arange(self.n_groups).unsqueeze(1)==self._group_array).sum(1).float()
            
            self._y_array = torch.LongTensor(y_array)
            self._y_counts = (torch.arange(self.n_classes).unsqueeze(1)==self._y_array).sum(1).float()
        uniques, count = np.unique(self.data[:, 1], return_counts=True)
        print(f"Mode: {self.mode} |  shape: {self.data.shape}")
        print(f"label counts: {uniques, count}")
        print(f"group counts: {self._group_counts}")
    
    def __getitem__(self, index: int):
        if self.mode in ['train', 'validation']:
            img_path, label, group = self.data[index]
        else:
            img_path, label = self.data[index]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        label = torch.tensor(int(label))
        if self.mode in ['train', 'validation']:
            group = torch.tensor(int(group))
            return image, label, group, img_path
        else:
            return image, label, img_path
    
    def group_counts(self):
        return self._group_counts

    def class_counts(self):
        return self._y_counts
        
    def __len__(self):
        return len(self.data)

    def get_loader(self, train, batch_size, n_workers):
        if not train:
            shuffle = False
        else:
            shuffle = True
        loader = DataLoader(
            self,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=n_workers
        )
        return loader
