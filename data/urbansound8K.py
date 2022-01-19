import math
import pandas as pd
import random
import torch
import torchaudio
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from scipy.stats import entropy
import librosa
import torchvision

from util import scale
from util.transformations import *


class UrbanSoundDataset(Dataset):

    """
    Class for the UrbanSound8K dataset
    Args:
        folder_list ([int]): list of folders to use in the dataset
        resample(int, optional): resampling rate.
        n_mels(int, optional): number Mel bands, or “bins”, that the Mel scale will be broken up into.
        mimic_bmw (bool, optional): mimic the imbalance of bmw dataset. Defaults to False.
        train_set (UrbanSoundDataset, optional): avoid overlapping between train and test sets. Defaults to None.
        spectr_form(bool, optional): convert the audio signal to a Mel spectrogram signal.train
        train(bool, optional): True if
    """

    def __init__(
        self,
        resample,
        split="train",
        mimic_bmw=False,
        train_set=None,
        spectr_form=False,
        max_class_samples=None,
        n_mels=128,
        spectr_channels=3,
        window_size=1024,
        hop_size=512,
        waveform_aug=False,
        spec_aug=False,
        random_scale=1.25,
        random_crop_size=192000,
        n_freq_masks=2,
        width_freq_masks=32,
        n_time_masks=1,
        width_time_masks=32,
    ):
        assert (mimic_bmw and train_set is None) or (not mimic_bmw)
        assert split in ["train", "test", "overfit"]
        if split == "overfit":
            folder_list = [1]
        elif split == "train":
            folder_list = list(range(1, 10))
            self.train = True
        elif split == "test":
            folder_list = list(range(1, 11))
            self.train = False

        # datasets paths:
        self.file_path = r"./data/datasets/UrbanSound8K/audio"
        csv_path = r"./data/datasets/UrbanSound8K/metadata/UrbanSound8K.csv"
        csv_data = pd.read_csv(csv_path)

        self.classes = [
            "air_conditioner",
            "car_horn",
            "children_playing",
            "dog_bark",
            "drilling",
            "engine_idling",
            "gun_shot",
            "jackhammer",
            "siren",
            "street_music",
        ]

        # initialize lists to hold file names, labels, and folder numbers
        self.file_names = []
        self.labels = []
        self.folders = []

        self.resample = resample
        self.spectr_form = spectr_form
        self.n_mels = n_mels
        self.spectr_channels = spectr_channels
        self.window_size = window_size
        self.hop_size = hop_size
        self.waveform_aug = waveform_aug
        self.spec_aug = spec_aug
        self.random_scale = random_scale
        self.random_crop_size = random_crop_size
        self.n_freq_masks = n_freq_masks
        self.width_freq_masks = width_freq_masks
        self.n_time_masks = n_time_masks
        self.width_time_masks = width_time_masks

        # loop through the csv entries and only add entries from folders in the folder list
        if mimic_bmw:
            self.mimic_bmw(csv_data, folder_list)
        else:
            for i in range(0, len(csv_data)):
                if train_set is not None:
                    if csv_data.iloc[i, 0] in train_set.file_names:
                        continue
                if csv_data.iloc[i, 5] in folder_list:
                    self.file_names.append(csv_data.iloc[i, 0])
                    self.labels.append(csv_data.iloc[i, 6])
                    self.folders.append(csv_data.iloc[i, 5])
                    if split == "overfit" and len(self.file_names) >= 64:
                        break
        self.labels = np.array(self.labels)

        if self.train:
            self.wave_transforms = torchvision.transforms.Compose(
                [
                    ToTensor1D(),
                    RandomScale(max_scale=self.random_scale),
                    # RandomPadding(out_len=int(self.resample*8.5), train=True),  # 8.5 seconds
                    RandomCrop(out_len=self.random_crop_size),
                ]
            )

            self.spec_transforms = torchvision.transforms.Compose(
                [
                    FrequencyMask(
                        max_width=self.width_freq_masks, numbers=self.n_freq_masks
                    ),
                    TimeMask(
                        max_width=self.width_time_masks, numbers=self.n_time_masks
                    ),
                ]
            )

        else:  # for test
            self.wave_transforms = torchvision.transforms.Compose(
                [
                    ToTensor1D(),
                    # RandomPadding(out_len=int(self.resample*8.5), train=False),  # 8.5 seconds
                    # RandomCrop(out_len=self.random_crop_size),
                ]
            )

            self.spec_transforms = torchvision.transforms.Compose([])

    def __getitem__(self, index):
        # format the file path and load the file
        path = (
            Path(self.file_path)
            / ("fold" + str(self.folders[index]))
            / self.file_names[index]
        )
        # load returns a tensor with the sound data and the sampling frequency (44.1kHz for UrbanSound8K)
        waveform, sample_rate = torchaudio.load(str(path))
        # convert waveform to mono
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        # downsample
        downsample = torchaudio.transforms.Resample(
            sample_rate, self.resample, resampling_method="sinc_interpolation"
        )

        # Remove silent sections
        # waveform is 2D Tensor
        start = waveform.nonzero()[:, 1].min()
        end = waveform.nonzero()[:, 1].max()
        waveform = waveform[:, start : end + 1]

        waveform = downsample(waveform)
        # Pad waveform to size 4 sec * sample rate
        _, n_values = waveform.shape
        missing_values = self.resample * 4 - n_values
        waveform = torch.nn.functional.pad(
            waveform,
            (math.floor(missing_values / 2), math.ceil(missing_values / 2)),
            "constant",
            0,
        )

        # apply transforms on wave form
        if self.waveform_aug:
            waveform = self.wave_transforms(waveform)
        else:
            waveform = waveform.squeeze()

        if not (self.spectr_form):
            return waveform.squeeze(), self.labels[index]
        else:
            spectogram = librosa.feature.melspectrogram(
                waveform.numpy(),
                sr=self.resample,
                n_mels=self.n_mels,
                n_fft=self.window_size,
                hop_length=self.hop_size,
            )
            log_spectogram = librosa.power_to_db(spectogram, ref=np.max)

            to_tensor = torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()]
            )
            log_spectogram = to_tensor(log_spectogram)

            # applying transforms on spectrogram form
            if self.spec_aug:
                log_spectogram = self.spec_transforms(log_spectogram)

            # creating 3 channels by copying log_s 3 times
            if self.spectr_channels == 3:
                melspectrogram = torch.cat(
                    (log_spectogram, log_spectogram, log_spectogram), dim=0
                )
            else:
                melspectrogram = log_spectogram

            return melspectrogram, self.labels[index]

    def __len__(self):
        return len(self.file_names)

    def mimic_bmw(self, csv_data, folder_list):
        """
        This function guarantees that bmw dataset and UrbanSound8k have the same balance.
        Args:
            csv_data (pandas.DataFrame): metadata of UrbanSound8k
            folder_list ([int]): list of folders to use in the dataset
        """
        bmw_classes = [313, 72, 135, 270, 250, 50]
        n = sum(bmw_classes)
        # compute bmw dataset shannon entropy and balance
        entropy_bmw = entropy(bmw_classes)
        balance_bmw = entropy_bmw / math.log(len(bmw_classes))

        # compute UrbanSound8k's entropy given the same balance as bmw dataset
        entropy_us8k = balance_bmw * math.log(10)

        # set the number of classes instances manually and compare the entropy and balance
        us8k_classes = [int(x * 6 / 10) for x in bmw_classes]
        us8k_classes += [266, 63, 45, 63]
        entropy_us8k = entropy(us8k_classes)
        balance_us8k = entropy_us8k / math.log(len(us8k_classes))

        # create train set according to the number of classes in us8k_classes
        appended_classes = [0] * 10
        indice_list = list(range(0, len(csv_data)))
        random.shuffle(indice_list)
        for i in indice_list:
            label = csv_data.iloc[i, 6]
            if (csv_data.iloc[i, 5] in folder_list) and (
                appended_classes[label] < us8k_classes[label]
            ):
                self.file_names.append(csv_data.iloc[i, 0])
                self.labels.append(label)
                self.folders.append(csv_data.iloc[i, 5])
                appended_classes[label] += 1
