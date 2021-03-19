#!/usr/bin/env python3

import os
from torch.utils.data import Dataset, ConcatDataset
import glob
import torchaudio
import torch.nn.functional as nnf
import torch

class data_info:
    def __init__(self, root, label):
        self.data_root = root
        self.data_label = label



class Meta_kws(Dataset):
    """

    [[Source]]()

    **Description**

    This class provides an interface to the Omniglot dataset.

    The Omniglot dataset was introduced by Lake et al., 2015.
    Omniglot consists of 1623 character classes from 50 different alphabets, each containing 20 samples.
    While the original dataset is separated in background and evaluation sets,
    this class concatenates both sets and leaves to the user the choice of classes splitting
    as was done in Ravi and Larochelle, 2017.
    The background and evaluation splits are available in the `torchvision` package.

    **References**

    1. Lake et al. 2015. “Human-Level Concept Learning through Probabilistic Program Induction.” Science.
    2. Ravi and Larochelle. 2017. “Optimization as a Model for Few-Shot Learning.” ICLR.

    **Arguments**

    * **root** (str) - Path to download the data.
    * **transform** (Transform, *optional*, default=None) - Input pre-processing.
    * **target_transform** (Transform, *optional*, default=None) - Target pre-processing.
    * **download** (bool, *optional*, default=False) - Whether to download the dataset.

    **Example**
    ~~~python
    omniglot = l2l.vision.datasets.FullOmniglot(root='./data',
                                                transform=transforms.Compose([
                                                    transforms.Resize(28, interpolation=LANCZOS),
                                                    transforms.ToTensor(),
                                                    lambda x: 1.0 - x,
                                                ]),
                                                download=True)
    omniglot = l2l.data.MetaDataset(omniglot)
    ~~~

    """

    def __init__(self, root, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        
        self.data_count = 0
        class_len = 0
        class_tem = []
        self.dataset =[]

        for name_class in glob.glob(self.root+'/*'):
            class_tem.append(name_class)
            class_len += 1

        for name in glob.glob(self.root+'/*/*'):
            data_tem = name
            root_tem = data_tem[:data_tem.rfind('/')]
            for class_count in range(class_len):
                if root_tem == class_tem[class_count]:
                    self.dataset.append( data_info(data_tem,class_count) )
                    data_tem = ''
                    break
            self.data_count += 1
        # # Set up both the background and eval dataset
        # omni_background = Omniglot(self.root, background=True, download=download)
        # # Eval labels also start from 0.
        # # It's important to add 964 to label values in eval so they don't overwrite background dataset.
        # omni_evaluation = Omniglot(self.root,
        #                            background=False,
        #                            download=download,
        #                            target_transform=lambda x: x + len(omni_background._characters))

        # self.dataset = ConcatDataset((omni_background, omni_evaluation))
        # self._bookkeeping_path = os.path.join(self.root, 'meta_kws-bookkeeping.pkl')
        


    def __len__(self):
        return self.data_count

    def __getitem__(self, item):

        waveform, sample_rate = torchaudio.load(self.dataset[item].data_root)
        specgram = torchaudio.transforms.Spectrogram(power = 2)(waveform).sqrt()
        label = self.dataset[item].data_label
        specgram = nnf.interpolate( specgram , size = (81), mode = 'nearest' )
        specgram = specgram.permute(1,2,0)
        specgram = nnf.interpolate(specgram , size = (3), mode = 'nearest')
        specgram = specgram.permute(2,0,1)
        print(specgram.shape)
        if self.transform:
            specgram = self.transform(specgram)
        if self.target_transform:
            label = self.target_transform(label)
        return specgram, label
