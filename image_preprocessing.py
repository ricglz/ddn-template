"""Module for preprocess the dataset"""
from glob import glob
from os import mkdir, path, remove
from pathlib import Path
from shutil import move

from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

from torch import Tensor
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torchvision.utils as t_utils

from pytorch_lightning import seed_everything

from constant import PROJECT_NAME

data_path = path.join('.', PROJECT_NAME)


training_path = path.join(data_path, 'train')
test_path = path.join(data_path, 'test')
validation_path = path.join(data_path, 'val')

classes = ImageFolder(test_path).classes

resize = T.Resize((224, 224))

def save_image(img: Tensor, label: str, index: int, prefix: str):
    '''
    Saves tensor image into the following filepath template:

    project_path/class/prefix_index.jpg
    '''
    klass = classes[label]
    img_path = path.join(training_path, klass, f'{prefix}_{index:08}.png')
    t_utils.save_image(img, img_path)

def resize_dataset(ds_path: str):
    '''Resizes the dataset into the normally used image size'''
    print(f'Resizing {ds_path}')
    dataset = ImageFolder(ds_path, T.Compose([resize, T.ToTensor()]))
    for index, (img_path, _) in enumerate(tqdm(dataset.imgs)):
        t_utils.save_image(dataset[index][0], img_path)

def clear_balanced():
    '''Removes images created for balancing a under-represented class'''
    balance_imgs = glob(
        f'{training_path}/**/balance*.png', recursive=True)
    for balance_img in tqdm(balance_imgs):
        remove(balance_img)

def get_mayor_count(dataset: ImageFolder):
    '''Gets the count of the most populated class'''
    targets = np.array(dataset.targets)
    get_count = lambda target: np.count_nonzero(targets == target)
    class_counts = [get_count(idx) for idx in range(len(dataset.classes))]
    return max(class_counts)

def balance_dataset(dataset_path: str = training_path):
    '''Balances the given dataset'''
    print('Balancing dataset')
    transforms = T.Compose([
      resize,
      T.ColorJitter(brightness=0.25, contrast=0.25),
      T.ToTensor(),
    ])
    dataset = ImageFolder(dataset_path, transforms)
    mayor_count = get_mayor_count(dataset)
    targets = np.array(dataset.targets)
    get_count = lambda target: np.count_nonzero(targets == target)
    for klass in range(len(dataset.classes)):
        class_count = get_count(klass)
        images_to_save = mayor_count - class_count
        if images_to_save == 0:
            continue
        class_indexes = np.where(targets == klass)[0]
        replace = len(class_indexes) < images_to_save
        random_chosen_indexes = np.random.choice(class_indexes, images_to_save, replace=replace)
        for save_img_index, index in enumerate(tqdm(random_chosen_indexes)):
            img, label = dataset[index]
            save_image(img, label, save_img_index, 'balance')

def half_the_data():
    '''Half the data only in training path because validation is more important'''
    transforms = T.Compose([resize, T.ToTensor()])
    train_ds = ImageFolder(training_path, transforms)
    files = list(map(lambda a: a[0], train_ds.samples))
    _, erase_files = train_test_split(
        files, test_size=0.5, shuffle=True, stratify=train_ds.targets)
    for file_to_erase in tqdm(erase_files):
        remove(file_to_erase)

def split_training_dataset():
    '''Splits the training dataset in two: train and val'''
    print('Splitting dataset')
    if not path.exists(validation_path):
        mkdir(validation_path)
        for klass in classes:
            mkdir(path.join(validation_path, klass))
    train_ds = ImageFolder(training_path)
    targets = train_ds.targets
    _, valid_idx= train_test_split(
        range(len(targets)), test_size=0.2, shuffle=True, stratify=targets)
    for idx in tqdm(valid_idx):
        img_path, label = train_ds.imgs[idx]
        filename = Path(img_path).name
        klass = classes[label]
        new_path = path.join(validation_path, klass, filename)
        move(img_path, new_path)

def count_dataset(dataset_path: str):
    '''Count amount of images in the dataset of each path'''
    print(f'Counting data inside {dataset_path}')
    dataset = ImageFolder(dataset_path)
    targets = np.array(dataset.targets)
    total = 0
    for klass, class_label in enumerate(classes):
        class_count = np.count_nonzero(targets == klass)
        total += class_count
        print(f'{class_label}: {class_count}')
    print(f'Total: {total}')

def main():
    '''Main function'''
    seed_everything(42)
    resize_dataset(training_path)
    resize_dataset(test_path)
    clear_balanced()
    balance_dataset(training_path)
    split_training_dataset()
    count_dataset(training_path)
    count_dataset(validation_path)
    count_dataset(test_path)

if __name__ == "__main__":
    main()
