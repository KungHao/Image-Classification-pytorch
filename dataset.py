import os
import torch
import random
import pandas as pd
from PIL import Image
from torchvision import datasets, transforms as T
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

class Cartoon(Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.mode = mode
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.dataset = {}
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name
        
        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        # Dataset has 7757 images.
        if self.mode == 'train':
            lines = lines[:-1500]       # train set contains x - 1500 images
        if self.mode == 'val':
            lines = lines[-1500:-500]   # val set contains 1000 images
        if self.mode == 'test':
            lines = lines[-500:]        # test set contains 500 images

        imgs, labels = [], []
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            imgs.append(filename)
            values = split[1:]

            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                labels.append(values[idx] == '1')
        
        self.dataset = {'imgs': imgs, 'labels': labels}

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        
        filename, label = self.dataset[index]
        label = [label]
        image = Image.open(os.path.join(self.image_dir, filename)).convert('RGB')

        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return len(self.dataset)

def get_class_weights(dataset):
    class_weights = []

    df = pd.DataFrame(dataset)
    # print(df['labels'].value_counts())
    countF = df['labels'].value_counts().values[0]
    countT = df['labels'].value_counts().values[1]

    class_weights = [1/countF, 1/countT]
    return class_weights

def random_sampler(dataset):
    class_weights = get_class_weights()
    sampl_weights = [0] * len(dataset)
    for i, (_, label) in enumerate(dataset):
        class_weight = class_weights[int(label)]
        sampl_weights[i] = class_weight
    sampler = WeightedRandomSampler(sampl_weights, num_samples=len(sampl_weights), replacement=True)

    return sampler

def get_loader(image_dir, attr_path, selected_attrs, crop_size=378, image_size=224, 
               batch_size=16):
    """Build and return a data loader."""

    dataloader = {}
    transform = []
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))

    val_transform = T.Compose(transform)       # make val loader before transform is inserted
    val_dataset = Cartoon(image_dir, attr_path, selected_attrs, val_transform, mode='val')
    sampler = random_sampler(val_dataset)
    dataloader['val'] = DataLoader(val_dataset, batch_size=batch_size, sampler=sampler, num_workers=1)

    test_transform = T.Compose(transform)
    test_dataset = Cartoon(image_dir, attr_path, selected_attrs, test_transform, mode='test')
    sampler = random_sampler(test_dataset)
    dataloader['test'] = DataLoader(test_dataset, batch_size=1, sampler=sampler, num_workers=1)
    
    transform.insert(0, T.RandomHorizontalFlip())
    train_transform = T.Compose(transform)
    train_dataset = Cartoon(image_dir, attr_path, selected_attrs, train_transform, mode='train')
    sampler = random_sampler(train_dataset)
    dataloader['train'] = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=1)

    return dataloader

def image_folder(image_dir, transform, batch_size): 
    "Pytorch imageFolder"
    dataset = datasets.ImageFolder(image_dir, transform)
    length = len(dataset)
    train_len = int(length*0.8)
    val_len = int(length*0.1)
    test_len = length - train_len - val_len
    train, val, test = torch.utils.data.random_split(dataset, [train_len, val_len, test_len])
    dataloader = {}
    dataloader['train'] = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    dataloader['val'] = torch.utils.data.DataLoader(val, batch_size, shuffle=True)
    dataloader['test'] = torch.utils.data.DataLoader(test, 1, shuffle=False)

    return dataloader