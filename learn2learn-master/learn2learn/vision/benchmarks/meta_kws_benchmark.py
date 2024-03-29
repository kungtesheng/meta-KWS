#!/usr/bin/env python3

import random
import learn2learn as l2l

from torchvision import transforms
from PIL.Image import LANCZOS


def meta_kws_tasksets(
    train_ways,
    train_samples,
    test_ways,
    test_samples,
    root,
    mode,
    **kwargs
):
    """
    Benchmark definition for Omniglot.
    """
    data_transforms = transforms.Compose([
        #transforms.Resize(28, interpolation=LANCZOS),
        transforms.ToPILImage(),
        transforms.ToTensor(),
        lambda x: 1.0 - x,
    ])
    kws = l2l.vision.datasets.Meta_kws(
        root=root,
        transform=data_transforms,
        download=False,
        mode = mode
    )
    dataset = l2l.data.MetaDataset(kws)
    train_dataset = dataset
    validation_datatset = dataset
    test_dataset = dataset

    classes = list(range(35))
    random.shuffle(classes)
    train_transforms = [
        l2l.data.transforms.FusedNWaysKShots(dataset,
                                             n=train_ways,
                                             k=train_samples,
                                             filter_labels=classes[:20]),
        l2l.data.transforms.LoadData(dataset),
        l2l.data.transforms.RemapLabels(dataset),
        l2l.data.transforms.ConsecutiveLabels(dataset),
        #l2l.vision.transforms.RandomClassRotation(dataset, [0.0, 90.0, 180.0, 270.0])
    ]
    validation_transforms = [
        l2l.data.transforms.FusedNWaysKShots(dataset,
                                             n=test_ways,
                                             k=test_samples,
                                             filter_labels=classes[20:30]),
        l2l.data.transforms.LoadData(dataset),
        l2l.data.transforms.RemapLabels(dataset),
        l2l.data.transforms.ConsecutiveLabels(dataset),
        #l2l.vision.transforms.RandomClassRotation(dataset, [0.0, 90.0, 180.0, 270.0])
    ]
    test_transforms = [
        l2l.data.transforms.FusedNWaysKShots(dataset,
                                             n=test_ways,
                                             k=test_samples,
                                             filter_labels=classes[30:]),
        l2l.data.transforms.LoadData(dataset),
        l2l.data.transforms.RemapLabels(dataset),
        l2l.data.transforms.ConsecutiveLabels(dataset),
        #l2l.vision.transforms.RandomClassRotation(dataset, [0.0, 90.0, 180.0, 270.0])
    ]

    _datasets = (train_dataset, validation_datatset, test_dataset)
    _transforms = (train_transforms, validation_transforms, test_transforms)
    return _datasets, _transforms
