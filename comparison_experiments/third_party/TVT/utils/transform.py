# coding=utf-8
from __future__ import absolute_import, division, print_function
from torchvision import transforms
from data.data_list_image import Normalize

def get_transform(dataset, img_size):
    # Official code uses data/ilsvrc_2012_mean.npy (VISDA per-pixel mean file),
    # which is unavailable on our server; use equivalent ImageNet per-channel
    # means instead (Normalize subtracts the mean only, no std division).
    IM_MEAN = [0.485, 0.456, 0.406]
    if dataset in ['svhn2mnist', 'usps2mnist', 'mnist2usps']:
        transform_source = transforms.Compose([
                transforms.RandomResizedCrop((img_size, img_size), scale=(0.75, 1.2)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

        transform_target = transforms.Compose([
                transforms.RandomResizedCrop((img_size, img_size), scale=(0.75, 1.2)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

        transform_test = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
    elif dataset in ['visda17', 'office-home']:
        transform_source = transforms.Compose([
                transforms.Resize((img_size+32, img_size+32)),
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                Normalize(mean=IM_MEAN)
        ])
        transform_test = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                Normalize(mean=IM_MEAN)
            ])
    else:
        transform_source = transforms.Compose([
                transforms.Resize((img_size+32, img_size+32)),
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                Normalize(mean=IM_MEAN)
        ])

        transform_test = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                Normalize(mean=IM_MEAN)
            ])

    return transform_source, transform_source, transform_test





