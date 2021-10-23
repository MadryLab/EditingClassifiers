import os
import torch as ch
from torchvision import transforms
import numpy as np
from robustness import datasets
from PIL import Image

def tile_image(img):
    tiled = Image.new('RGB', (800,800), "black")
    pixels = img.load()
    pixels_tiled = tiled.load()
    for i in range(800):
        for j in range(800):
            pixels_tiled[i,j] = pixels[i % 256,j % 256]
    return tiled

def get_dataset(dataset_name, dataset_path, 
                batch_size=32, workers=8):
    assert dataset_name in ['ImageNet', 'Places365']
    if dataset_name == 'ImageNet':
        dataset = datasets.ImageNet(dataset_path)
    else:
        dataset = datasets.Places365(dataset_path)

    train_loader, val_loader = dataset.make_loaders(batch_size=32, workers=8)
    return dataset, train_loader, val_loader

def tile_image(img):
    tiled = Image.new('RGB', (800,800), "black")
    pixels = img.load()
    pixels_tiled = tiled.load()
    for i in range(800):
        for j in range(800):
            pixels_tiled[i,j] = pixels[i % 256,j % 256]
    return tiled

def get_vehicles_on_snow_data(dataset_name, class_dict, dataset_path='./data/'):
    
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256 if dataset_name == 'Places' else 224),
            transforms.ToTensor(),
        ])
    
    TRAIN_PATH = f'{dataset_path}/train_snow/road:15:05:2021_17:39:17.pt'
    train_data = ch.load(TRAIN_PATH)
    train_imgs, train_masks, train_labels = train_data['imgs'], train_data['masks'], train_data['labels']
    
    pattern_img_path = f'{dataset_path}/train_snow/snow_texture.jpg'
    pattern_img = transform(Image.open(pattern_img_path))[:3, :, :]
    modified_imgs = train_imgs * (1-train_masks) + pattern_img.unsqueeze(0) * train_masks
    
    train_data = {'imgs': train_imgs,
                 'modified_imgs': modified_imgs,
                 'masks': train_masks,
                 'labels': train_labels
                 }

    TEST_PATH = f'{dataset_path}/test_snow'
    test_data = get_scraped_data(TEST_PATH, dataset_name, class_dict, transform)
    
    return train_data, test_data

def get_scraped_data(data_path, dataset_name, class_dict, transform):
    data = {}

    print("Test data stats...")
    for c in os.listdir(data_path):
        l = []
        for f in os.listdir(os.path.join(data_path, c)):
            img_path = os.path.join(data_path, c, f)
            img = Image.open(img_path)
            l.append(transform(img)[:3, :, :])
        valid_classes = [k for k, v in class_dict.items() if c.split('_')[0] in v.replace(' ', '')]
        assert len(valid_classes) == 1
        valid_classes = valid_classes[0]
        data[valid_classes] = ch.stack(l)
        
        print(f'ImageNet class: {class_dict[valid_classes]}; # Images: {len(data[valid_classes])} \n')
    return data