import torch
import statistics
import torchaudio
import matplotlib.pyplot as plt
from data import BmwDataset
from tqdm import tqdm


class BmwDatasetCustom(BmwDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        # format the file path and load the file
        path = self.dataset_path / self.class_names[index] / self.file_names[index]
        # load returns a tensor with the sound data and the sampling frequency
        waveform_mono, sample_rate = torchaudio.load(str(path))
        return waveform_mono, sample_rate


def get_bmw_stats():
    dataset = BmwDatasetCustom()
    duration = []
    for i, item in tqdm(enumerate(dataset)):
        waveform_mono, sample_rate = item
        # print("duration: ", torch.numel(waveform_mono) / sample_rate)
        duration.append(torch.numel(waveform_mono) / sample_rate)

    print("max: ", max(duration))
    print("min: ", min(duration))
    print("mean:", statistics.mean(duration))
    print("median:", statistics.median(duration))
    # fig, ax = plt.subplots()
    # ax.bar([i for i in range(len(duration))], duration)
    plt.hist(duration, bins=50)
    plt.show(block=False)
