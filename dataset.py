import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from albumentations import Compose, CLAHE, Flip, RandomRotate90, RandomBrightnessContrast, ShiftScaleRotate, RGBShift, Resize, Normalize
from albumentations.pytorch import ToTensor

import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer

DATA_PATH = Path('data')

LAND = ['agriculture', 'artisinal_mine', 'bare_ground', 'blooming', 'blow_down', 'conventional_mine',
        'cultivation', 'habitation', 'primary', 'road', 'selective_logging', 'slash_burn', 'water']

WEATHER = ['clear', 'cloudy', 'haze', 'partly_cloudy']

mlb = MultiLabelBinarizer().fit([WEATHER, LAND])

class PlanetDataset(Dataset):
    def __init__(self, csv_path, img_folder, ext, transform, val=False):
        self.img_folder = img_folder
        self.ext = ext
        self.transform = transform
        
        # prototype
        # if val:
        #     self.csv = pd.read_csv(csv_path)[:100]
        # else:
        #     self.csv = pd.read_csv(csv_path)[:2000]
        self.csv = pd.read_csv(csv_path)

        self.x_train = self.csv['image_name']
        self.y_train = mlb.transform(self.csv['tags'].str.split()).astype(np.float32)
    
    def __len__(self):
        return self.csv.shape[0]
    
    def __getitem__(self, idx):
        # img = Image.open(f'{self.img_folder}/{self.x_train[idx]}.{self.ext}')
        # img = img.convert('RGB')
        img = cv2.imread(str(f'{self.img_folder}/{self.x_train[idx]}.{self.ext}'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = self.transform(image=img)['image']
        y = self.y_train[idx]
        
        return (x, y)


def get_data(img_size, batch_size):
    CSV_PATH = DATA_PATH/'train_v2.csv'
    IMG_FOLDER = DATA_PATH/'train-jpg'
    EXT = 'jpg'
    SZ = img_size
    BS = batch_size
    MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

    # torch transforms
    # transform = transforms.Compose([
    #     transforms.Resize(SZ),
    #     transforms.ToTensor(),
    #     transforms.Normalize(MEAN, STD)
    # ])

    transform = {
        'train': Compose([
            Resize(height=SZ, width=SZ),
            CLAHE(clip_limit=1.0, p=0.25),
            Flip(p=0.5),
            RandomRotate90(p=0.5),
            RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0),
            RGBShift(p=0.25),
            Normalize(mean=MEAN, std=STD),
            ToTensor()
        ]),

        'val': Compose([
            Resize(height=SZ, width=SZ),
            Flip(p=0.5),
            RandomRotate90(p=0.5),
            Normalize(mean=MEAN, std=STD),
            ToTensor()
        ])
    }

    train_ds = PlanetDataset(CSV_PATH/'train.csv', IMG_FOLDER, EXT, transform['train'])
    val_ds = PlanetDataset(CSV_PATH/'val.csv', IMG_FOLDER, EXT, transform['val'], val=True)
    train_dl = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=BS * 2, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    
    # Show the details in the console
    print(f'''Train DS: {train_ds.img_folder}   \t \
              Ext: {train_ds.ext}               \t \
              x_train: {train_ds.x_train.shape} \t \
              y_train: {train_ds.y_train.shape} \t''')

    print(f'''Validation DS: {val_ds.img_folder} \t \
              Ext: {val_ds.ext}                  \t \
              x_train: {val_ds.x_train.shape}    \t \
              y_train: {val_ds.y_train.shape}    \t''')

    return (train_dl, val_dl)


class PlanetDatasetTest(Dataset):
    def __init__(self, img_folder, transform):
        self.img_folder = img_folder
        self.transform = transform
        self.filenames = [path.name for path in Path(img_folder).iterdir()]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # img = Image.open(f'{self.img_folder}/{self.filenames[idx]}')
        # img = img.convert('RGB')
        # x = self.transform(img)
        img = cv2.imread(str(f'{self.img_folder}/{self.filenames[idx]}'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)['image']

        return (img, self.filenames[idx].split('.')[0])


def get_data_test(img_size, batch_size):
    IMG_FOLDER = DATA_PATH/'test-jpg'
    SZ = img_size
    BS = batch_size
    MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

    # transform = transforms.Compose([
    #     transforms.Resize(SZ),
    #     transforms.ToTensor(),
    #     transforms.Normalize(MEAN, STD)
    # ])

    transform = {
        'test': Compose([
            Resize(height=SZ, width=SZ),
            Normalize(mean=MEAN, std=STD),
            ToTensor()
        ])
    }

    test_ds = PlanetDatasetTest(IMG_FOLDER, transform['test'])
    test_dl = DataLoader(test_ds, batch_size=BS, shuffle=False,
                         num_workers=4, pin_memory=True, drop_last=False)

    # Show the details in the console
    print(f'''Test DS: {test_ds.img_folder}              \t \
              Ext: jpg                                   \t \
              Number of Images: {len(test_ds.filenames)} \t''')

    return (mlb, test_dl)
