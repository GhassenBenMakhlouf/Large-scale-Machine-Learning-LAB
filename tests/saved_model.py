import torch
from torch.utils.data import DataLoader
from data import UrbanSoundDataset
from models.toy_model_mel import ToyModelMel
from util import worker_init_fn


def test_loaded_model():
    US8K_AUDIO_PATH = r"./data/datasets/UrbanSound8K/audio"
    US8K_METADATA_PATH = r"./data/datasets/UrbanSound8K/metadata/UrbanSound8K.csv"

    train_set = UrbanSoundDataset(
        US8K_METADATA_PATH,
        US8K_AUDIO_PATH,
        list(range(1, 10)),
        mimic_bmw=True,
        spectr_form=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = (
        {
            "num_workers": 1,
            "pin_memory": True,
            "worker_init_fn": worker_init_fn,
        }
        if device == "cuda"
        else {}
    )
    train_dataloader = DataLoader(train_set, batch_size=1, shuffle=True, **kwargs)
    item = next(iter(train_dataloader))
    input = item[0]
    label = item[1]

    model = ToyModelMel()
    model.load_state_dict(torch.load("toy_example.pth"))

    out = model(input)
    print(f"output {out}")
    print(f"label {label}")
