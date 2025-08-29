import os

import torch
from torch.utils.data import Dataset, Subset

import glob
import numpy as np

from einops import rearrange

#Adapted from PANGAEA https://github.com/VMarsocci/pangaea-bench

class RawGeoFMDataset(Dataset):
    """Base class for all datasets."""

    def __init__(
        self,
        split: str,
        dataset_name: str,
        multi_modal: bool,
        multi_temporal: int,
        root_path: str,
        classes: list,
        num_classes: int,
        ignore_index: int,
        img_size: int,
        bands: dict[str, list[str]],
        distribution: list[int],
        data_mean: dict[str, list[str]],
        data_std: dict[str, list[str]],
        data_min: dict[str, list[str]],
        data_max: dict[str, list[str]],
        download_url: str,
        auto_download: bool,
    ):
        """Initializes the dataset.

        Args:
            split (str): split of the dataset (train, val, test)
            dataset_name (str): dataset name
            multi_modal (bool): whether the dataset is multi_modal
            multi_temporal (int): number of temporal frames
            root_path (str): root path of the dataset
            classes (list): dataset classes names
            num_classes (int): number of classes
            ignore_index (int): index to ignore
            img_size (int): dataset's image size
            bands (dict[str, list[str]]): bands of the dataset
            distribution (list[int]): class distribution.
            data_mean (dict[str, list[str]]): mean for each band for each modality.
            Dictionary with keys as the modality and values as the list of means.
            e.g. {"s2": [b1_mean, ..., bn_mean], "s1": [b1_mean, ..., bn_mean]}
            data_std (dict[str, list[str]]): str for each band for each modality.
            Dictionary with keys as the modality and values as the list of stds.
            e.g. {"s2": [b1_std, ..., bn_std], "s1": [b1_std, ..., bn_std]}
            data_min (dict[str, list[str]]): min for each band for each modality.
            Dictionary with keys as the modality and values as the list of mins.
            e.g. {"s2": [b1_min, ..., bn_min], "s1": [b1_min, ..., bn_min]}
            data_max (dict[str, list[str]]): max for each band for each modality.
            Dictionary with keys as the modality and values as the list of maxs.
            e.g. {"s2": [b1_max, ..., bn_max], "s1": [b1_max, ..., bn_max]}
            download_url (str): url to download the dataset.
            auto_download (bool): whether to download the dataset automatically.
        """
        self.split = split
        self.dataset_name = dataset_name
        self.multi_modal = multi_modal
        self.multi_temporal = multi_temporal
        self.root_path = root_path
        self.classes = classes
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.img_size = img_size
        self.bands = bands
        self.distribution = distribution
        self.data_mean = data_mean
        self.data_std = data_std
        self.data_min = data_min
        self.data_max = data_max
        self.download_url = download_url
        self.auto_download = auto_download

        if not os.path.exists(self.root_path):
            self.download(self)

    def __len__(self) -> int:
        """Returns the length of the dataset.

        Raises:
            NotImplementedError: raise if the method is not implemented

        Returns:
            int: length of the dataset
        """
        raise NotImplementedError

    def __getitem__(self, i: int) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Returns the i-th item of the dataset.

        Args:
            i (int): index of the item

        Raises:
            NotImplementedError: raise if the method is not implemented

        Returns:
            dict[str, torch.Tensor | dict[str, torch.Tensor]]: output dictionary follwing the format
            {"image":
                {
                "optical": torch.Tensor of shape (C T H W) (where T=1 if single-temporal dataset),
                 "sar": torch.Tensor of shape (C T H W) (where T=1 if single-temporal dataset),
                 },
            "target": torch.Tensor of shape (H W) of type torch.int64 for segmentation, torch.float for
            regression datasets.,
             "metadata": dict}.
        """
        raise NotImplementedError

    @staticmethod
    def download(self) -> None:
        """Download the dataset.

        Raises:
            NotImplementedError: raise if the method is not implemented
        """
        raise NotImplementedError

class PairedModalityLC(RawGeoFMDataset):

    year = '2022'
    months  = [
        str(i).rjust(2,'0') for i in range(1,13,1)
    ]
    modalities = [
        's1',
        's2',
        'hls',
        'l8',
        'l9'
    ]
    def __init__(
        self,
        split: str,
        dataset_name: str,
        multi_modal: bool,
        multi_temporal: int,
        root_path: str,
        classes: list,
        num_classes: int,
        ignore_index: int,
        img_size: int,
        bands: dict[str, list[str]],
        distribution: list[int],
        data_mean: dict[str, list[str]],
        data_std: dict[str, list[str]],
        data_min: dict[str, list[str]],
        data_max: dict[str, list[str]],
        download_url: str,
        auto_download: bool,
    ):
        super(PairedModalityLC, self).__init__(
            split=split,
            dataset_name=dataset_name,
            multi_modal=multi_modal,
            multi_temporal=multi_temporal,
            root_path=root_path,
            classes=classes,
            num_classes=num_classes,
            ignore_index=ignore_index,
            img_size=img_size,
            bands=bands,
            distribution=distribution,
            data_mean=data_mean,
            data_std=data_std,
            data_min=data_min,
            data_max=data_max,
            download_url=download_url,
            auto_download=auto_download,
        )

        self.root_path = root_path
        self.classes = classes
        self.split = split

        self.data_mean = data_mean
        self.data_std = data_std
        self.data_min = data_min
        self.data_max = data_max
        self.classes = classes
        self.img_size = img_size
        self.distribution = distribution
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.download_url = download_url
        self.auto_download = auto_download


        self.files = []
        for date in self.months:
            dir = glob.glob(os.path.join(root_path,self.year,date,'Scene_*'))
            self.files = self.files + dir

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self,index):

        file = self.files[index]

        meta = file.split('/')

        scene_id = meta[-2].split('_')[-1]
        date = meta[-4] + '-' + meta[-3]

        hls = torch.from_numpy(np.load(os.path.join(file,'hls_image.npy')).astype(np.float32))
        s2 = torch.from_numpy(np.load(os.path.join(file,'s2_image.npy')).astype(np.float32))
        l8 = torch.from_numpy(np.load(os.path.join(file,'l8_image.npy')).astype(np.float32))
        l9 = torch.from_numpy(np.load(os.path.join(file,'l9_image.npy')).astype(np.float32))
        s1 = torch.from_numpy(np.load(os.path.join(file,'s1_image.npy')).astype(np.float32))

        target = torch.from_numpy(np.load(os.path.join(file,'lc.npy')).astype(np.int64)).long()
        output = {
            'image':{
                'hls':hls,
                's2':s2,
                'l8':l8,
                'l9':l9,
                's1':s1
            },
            'target':target,
            'metadata':{
                'scene':scene_id
            }
        }

        return output

class RawGeoFMDataset(Dataset):
    """Base class for all datasets."""

    def __init__(
        self,
        split: str,
        dataset_name: str,
        multi_modal: bool,
        multi_temporal: int,
        root_path: str,
        classes: list,
        num_classes: int,
        ignore_index: int,
        img_size: int,
        bands: dict[str, list[str]],
        distribution: list[int],
        data_mean: dict[str, list[str]],
        data_std: dict[str, list[str]],
        data_min: dict[str, list[str]],
        data_max: dict[str, list[str]],
        download_url: str,
        auto_download: bool,
    ):
        """Initializes the dataset.

        Args:
            split (str): split of the dataset (train, val, test)
            dataset_name (str): dataset name
            multi_modal (bool): whether the dataset is multi_modal
            multi_temporal (int): number of temporal frames
            root_path (str): root path of the dataset
            classes (list): dataset classes names
            num_classes (int): number of classes
            ignore_index (int): index to ignore
            img_size (int): dataset's image size
            bands (dict[str, list[str]]): bands of the dataset
            distribution (list[int]): class distribution.
            data_mean (dict[str, list[str]]): mean for each band for each modality.
            Dictionary with keys as the modality and values as the list of means.
            e.g. {"s2": [b1_mean, ..., bn_mean], "s1": [b1_mean, ..., bn_mean]}
            data_std (dict[str, list[str]]): str for each band for each modality.
            Dictionary with keys as the modality and values as the list of stds.
            e.g. {"s2": [b1_std, ..., bn_std], "s1": [b1_std, ..., bn_std]}
            data_min (dict[str, list[str]]): min for each band for each modality.
            Dictionary with keys as the modality and values as the list of mins.
            e.g. {"s2": [b1_min, ..., bn_min], "s1": [b1_min, ..., bn_min]}
            data_max (dict[str, list[str]]): max for each band for each modality.
            Dictionary with keys as the modality and values as the list of maxs.
            e.g. {"s2": [b1_max, ..., bn_max], "s1": [b1_max, ..., bn_max]}
            download_url (str): url to download the dataset.
            auto_download (bool): whether to download the dataset automatically.
        """
        self.split = split
        self.dataset_name = dataset_name
        self.multi_modal = multi_modal
        self.multi_temporal = multi_temporal
        self.root_path = root_path
        self.classes = classes
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.img_size = img_size
        self.bands = bands
        self.distribution = distribution
        self.data_mean = data_mean
        self.data_std = data_std
        self.data_min = data_min
        self.data_max = data_max
        self.download_url = download_url
        self.auto_download = auto_download

        if not os.path.exists(self.root_path):
            self.download(self)

    def __len__(self) -> int:
        """Returns the length of the dataset.

        Raises:
            NotImplementedError: raise if the method is not implemented

        Returns:
            int: length of the dataset
        """
        raise NotImplementedError

    def __getitem__(self, i: int) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Returns the i-th item of the dataset.

        Args:
            i (int): index of the item

        Raises:
            NotImplementedError: raise if the method is not implemented

        Returns:
            dict[str, torch.Tensor | dict[str, torch.Tensor]]: output dictionary follwing the format
            {"image":
                {
                "optical": torch.Tensor of shape (C T H W) (where T=1 if single-temporal dataset),
                 "sar": torch.Tensor of shape (C T H W) (where T=1 if single-temporal dataset),
                 },
            "target": torch.Tensor of shape (H W) of type torch.int64 for segmentation, torch.float for
            regression datasets.,
             "metadata": dict}.
        """
        raise NotImplementedError

    @staticmethod
    def download(self) -> None:
        """Download the dataset.

        Raises:
            NotImplementedError: raise if the method is not implemented
        """
        raise NotImplementedError