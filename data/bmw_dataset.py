import math
import librosa
import torch
import torchaudio
import torchvision
import random
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from util.transformations import *


class BmwDataset(Dataset):
    """
    Class for the BMW Dataset
    Args:
        dataset_path(str): path to the BmwDataset csv file
        resample(int, optional): resampling rate.
        n_mels(int, optional): number Mel bands, or “bins”, that the Mel scale will be broken up into.
        train_set (BmwDataset, optional): avoid overlapping between train and test sets. Defaults to None.
        spectr_form(bool, optional): convert the audio signal to a Mel spectrogram signal.
    """

    def __init__(
        self,
        test_percent,
        resample,
        train,
        max_class_samples=None,
        spectr_form=False,
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
        # get variables from cfg:
        dataset_path = r"./data/datasets/BMW/Brake_Noise_Jan20_trim"
        self.dataset_path = Path(dataset_path)

        self.classes = [
            "Hubknarzen",
            "Knarzen",
            "No brake noise",
            "Quietschen",
            "Scheibenknacken",
            "Schrummknarzen",
        ]

        # initialize lists to hold file names and labels
        self.file_names = []
        self.class_names = []
        self.labels = []

        self.resample = resample
        self.train = train
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

        for i, folder in enumerate(self.classes):
            # Join the two strings in order to form the full filepath.
            class_path = self.dataset_path / folder
            files = [x.name for x in sorted(class_path.glob("*.wav"))]

            if max_class_samples:
                assert max_class_samples < len(
                    files
                ), f"max_class_samples is bigger than the number of samples in class {folder}."

                random.seed(42)
                random.shuffle(files)
                if train:
                    files = files[:max_class_samples]
                else:
                    files = files[max_class_samples:]
            self.file_names += files
            self.class_names += [folder] * len(files)
            self.labels += [i] * len(files)

        if max_class_samples is None:
            idx_train, idx_test = train_test_split(
                np.arange(len(self.labels)),
                test_size=test_percent,
                stratify=self.labels,
                random_state=1,
            )
            if self.train:
                self.file_names = [self.file_names[x] for x in idx_train]
                self.class_names = [self.class_names[x] for x in idx_train]
                self.labels = [self.labels[x] for x in idx_train]
            else:
                self.file_names = [self.file_names[x] for x in idx_test]
                self.class_names = [self.class_names[x] for x in idx_test]
                self.labels = [self.labels[x] for x in idx_test]
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
        path = self.dataset_path / self.class_names[index] / self.file_names[index]
        # load returns a tensor with the sound data and the sampling frequency
        # BMW waveforms are already mono
        waveform, sample_rate = torchaudio.load(str(path))

        # Remove silent sections
        # waveform is 2D Tensor
        start = waveform.nonzero()[:, 1].min()
        end = waveform.nonzero()[:, 1].max()
        waveform = waveform[:, start : end + 1]

        # downsample
        downsample = torchaudio.transforms.Resample(
            sample_rate, self.resample, resampling_method="sinc_interpolation"
        )
        waveform = downsample(waveform)

        # Pad waveform to size 8.5 sec * sample rate
        _, n_values = waveform.shape
        missing_values = self.resample * 8.5 - n_values
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

        if not self.spectr_form:
            return waveform, self.labels[index]
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
