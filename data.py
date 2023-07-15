import os
import tqdm
import json
import torch
import numpy as np
from PIL import Image
from data_transform import get_transform
from torch.utils.data import DataLoader, Dataset


def get_train_name_label_pairs():
    with open('./data/data.json', 'r') as f:
        name_label_pairs = json.load(f)
    return name_label_pairs


def get_val_name_label_pairs():
    mapping = np.load('./data/ID/ImageNet/row_max.npy')
    mapping_dict = dict()
    for i in range(mapping.shape[0]):
        mapping_dict[i] = mapping[i]

    txt_path = './data/ID/ImageNet/val_gt.txt'
    data_path = './data/ID/ImageNet/val/'

    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    name_label_pairs = list()
    for line in lines:
        split_line = line.strip().split(' ')
        abs_img_name = data_path + split_line[0]
        label = mapping_dict[int(split_line[1]) - 1]
        name_label_pairs.append((abs_img_name, label))
    return name_label_pairs


class MyDataset(Dataset):
    def __init__(self, names, transform, id=True):
        self.names = names
        self.transform = transform
        self.label = id

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        if self.label:
            img_name, label = self.names[index]
            img = Image.open(img_name).convert('RGB')
            img = self.transform(img)
            return img, int(label)
        else:
            img_name = self.names[index]
            img = Image.open(img_name).convert('RGB')
            img = self.transform(img)
            return img
            

def data_pipeline(batch_size):
    train_pairs = get_train_name_label_pairs()
    val_pairs = get_val_name_label_pairs()
    train_transform, val_transform = get_transform()
    train_set = MyDataset(train_pairs, transform=train_transform)
    val_set = MyDataset(val_pairs, transform=val_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=6)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader


def get_train_val_dataset(train_num=None):
    train_pairs = get_train_name_label_pairs()
    val_pairs = get_val_name_label_pairs()
    train_transform, val_transform = get_transform()
    if train_num is not None:
        np.random.shuffle(train_pairs)
        train_set = MyDataset(train_pairs[:train_num], transform=train_transform)
    else:
        train_set = MyDataset(train_pairs, transform=train_transform)
    val_set = MyDataset(val_pairs, transform=val_transform)
    return train_set, val_set


def get_imagenet_ood_dataset(ood_type):
    if ood_type == 'iNaturalist':
        ood_path = './data/OOD/As_ImageNet_OOD/iNaturalist/'
        ood_names = [ood_path + img_name for img_name in os.listdir(ood_path)]
    elif ood_type == 'Places':
        ood_path = './data/OOD/As_ImageNet_OOD/Places/'
        ood_names = [ood_path + img_name for img_name in os.listdir(ood_path)]
    elif ood_type == 'SUN':
        ood_path = './data/OOD/As_ImageNet_OOD/SUN/'
        ood_names = [ood_path + img_name for img_name in os.listdir(ood_path)]
    elif ood_type == 'Texture':
        ood_path = './data/OOD/As_CIFAR10_OOD/classicOOD/Texture/images/'
        ood_names = list()
        for folder in os.listdir(ood_path):
            abs_folder = ood_path + folder
            for img_name in os.listdir(abs_folder):
                ood_names.append(abs_folder + '/' + img_name)
    
    _, val_transform = get_transform()
    ood_dataset = MyDataset(ood_names, val_transform, id=False)
    return ood_dataset


# def replace_img_name():
#     with open('./data/data.json', 'r') as f:
#         name_label_pairs = json.load(f)
    
#     relative_img_anns = list()
#     for img_ann in tqdm.tqdm(name_label_pairs):
#         img_name, img_label = img_ann
#         relative_name = '.' + img_name.split('/mnt/bn/benchmark-dataset/BayesAug')[-1]
#         relative_img_anns.append([relative_name, img_label])
    
#     print(len(relative_img_anns))
    
#     with open('./data/data.json', 'w') as f:
#         json.dump(relative_img_anns, f)
    

if __name__ == '__main__':
    batch_size = 64

    # checking ID dataset
    train_loader, val_loader = data_pipeline(batch_size)
    for x, y in tqdm.tqdm(train_loader):
        print(x.shape, y.shape, torch.min(x), torch.max(x))
    
    for x, y in tqdm.tqdm(val_loader):
        print(x.shape, y.shape, torch.min(x), torch.max(x))

    # check OOD dataset
    # ood_set = get_imagenet_ood_dataset(ood_type='Texture')
    # ood_set = get_imagenet_ood_dataset(ood_type='iNaturalist')
    # ood_set = get_imagenet_ood_dataset(ood_type='Places')
    ood_set = get_imagenet_ood_dataset(ood_type='SUN')
    ood_loader = DataLoader(ood_set, batch_size=batch_size, shuffle=True, num_workers=3)
    for x in tqdm.tqdm(ood_loader):
        print(x.shape, torch.min(x), torch.max(x))