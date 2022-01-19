import matplotlib
import torchaudio
from data import UrbanSoundDataset, BmwDataset
from util import plot_spectrogram, plot_waveform
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torch.utils.data import DataLoader


def test_urban_dataset(cfg):
    """
    Test UrbanSoundDataset implementation
        and plot waveform and spectogram of one example
    """
    train_set = UrbanSoundDataset(
        split="train",
        resample=32000,
        mimic_bmw=True,
        spectr_form=True,
    )
    test_set = UrbanSoundDataset(
        split="test",
        resample=32000,
        train_set=train_set,
        spectr_form=True,
    )

    # Test Urban Dataset implementation
    print("Train set size: " + str(len(train_set)))
    print("Test set size: " + str(len(test_set)))
    item = train_set[0]
    print(item[0][0].shape)
    print(item[0][0])
    plot_spectrogram(item[0][0])
    input("Press <ENTER> to continue")

    # plot_waveform(item[0], 8000)
    # input("Press <ENTER> to continue")
    # mel_specgram = torchaudio.transforms.MelSpectrogram(8000)(item[0])
    # plot_spectrogram(
    #     mel_specgram[0], title="MelSpectrogram - torchaudio", ylabel="mel freq"
    # )
    # input("Press <ENTER> to continue")


def test_bmw_dataset():
    """
    Test BmwDataset implementation
        and plot waveform and spectogram of one example
    """
    train_set = BmwDataset(
        test_percent=0.2,
        resample=8000,
        spectr_form=False,
        n_mels=128,
        train=True,
        max_class_samples=None,
    )
    train_set_spec = BmwDataset(
        test_percent=0.2,
        resample=8000,
        spectr_form=True,
        n_mels=128,
        train=True,
        max_class_samples=None,
    )

    # Test Urban Dataset implementation
    print("Train set size: " + str(len(train_set)))
    print("Test set size: " + str(len(train_set_spec)))

    item = train_set_spec[0]
    print(item[0].shape)
    print(item[0])
    plot_spectrogram(item[0])
    input("Press <ENTER> to continue")

    # # Spectrogram extractor
    # spectrogram_extractor = Spectrogram(
    #     n_fft=400,
    #     hop_length=200,
    #     win_length=400,
    #     window="hann",
    #     center=True,
    #     pad_mode="reflect",
    #     freeze_parameters=True,
    # )

    # # Logmel feature extractor
    # logmel_extractor = LogmelFilterBank(
    #     sr=8000,
    #     n_fft=400,
    #     n_mels=128,
    #     fmin=0,
    #     fmax=8000,
    #     ref=1.0,
    #     amin=1e-10,
    #     top_db=None,
    #     freeze_parameters=True,
    # )
    # trainloader = DataLoader(
    #     train_set,
    #     batch_size=16,
    #     num_workers=0,
    #     pin_memory=True,
    # )

    # for i, batch in enumerate(trainloader):
    #     x = spectrogram_extractor(batch[0])  # (batch_size, 1, time_steps, freq_bins)
    #     x = logmel_extractor(x)
    #     print(x.shape)
    #     break

    # plot_spectrogram(x[0, 0, :, :].T)
    # input("Press <ENTER> to continue")
