from unittest.mock import MagicMock

import pytest
import torch
import yaml

from desed_task.utils import ExponentialWarmup
from encoder import ManyHotEncoder
from model import CRNN
from trainer import SED
from utils import classes_labels


@pytest.fixture
def sed_model():
    with open("tests/test_params.yaml", "r") as f:
        config = yaml.safe_load(f)

    encoder = ManyHotEncoder(
        list(classes_labels.keys()),
        audio_len=config["data"]["audio_max_len"],
        frame_len=config["feats"]["n_filters"],
        frame_hop=config["feats"]["hop_length"],
        net_pooling=config["data"]["net_subsample"],
        fs=config["data"]["fs"],
    )
    crnn = CRNN(**config["net"])
    opt = torch.optim.Adam(crnn.parameters(), config["opt"]["lr"], betas=(0.9, 0.999))
    exp_steps = config["training"]["n_epochs_warmup"] * 10
    exp_scheduler = {
        "scheduler": ExponentialWarmup(opt, config["opt"]["lr"], exp_steps),
        "interval": "step",
    }
    sed = SED(config, encoder=encoder, sed=crnn, scheduler=exp_scheduler, opt=opt)
    return sed


def test_init(sed_model):
    assert isinstance(sed_model, SED), "Initialization failed, SED object not created"


def test_train_dataloader(sed_model):
    sed_model.train_data = MagicMock()
    sed_model.hparams["training"]["num_workers"] = 1
    loader = sed_model.train_dataloader()
    assert loader is not None, "Train dataloader not instantiated"


def test_val_dataloader(sed_model):
    sed_model.valid_data = MagicMock()
    sed_model.hparams["training"]["batch_size_val"] = 2
    sed_model.hparams["training"]["num_workers"] = 1
    loader = sed_model.val_dataloader()
    assert loader is not None, "Validation dataloader not instantiated"


def test_test_dataloader(sed_model):
    sed_model.test_data = MagicMock()
    sed_model.hparams["training"]["batch_size_val"] = 2
    sed_model.hparams["training"]["num_workers"] = 1
    loader = sed_model.test_dataloader()
    assert loader is not None, "Test dataloader not instantiated"


def test_forward(sed_model):
    input_data = torch.rand((16000))
    output = sed_model.forward(input_data)
    assert output is not None, "Forward method failed to produce output"


def test_training_step(sed_model):
    batch = (torch.rand((10, 160000)), torch.rand((10, 10, 618)), None)
    batch_indx = 0
    loss = sed_model.training_step(batch, batch_indx)
    assert loss is not None, "Training step did not return a loss"


def test_validation_step(sed_model):
    batch = (
        torch.rand((10, 160000)),
        torch.rand((10, 10)),
        None,
        ["file1.wav", "file2.wav"],
    )
    batch_indx = 0
    sed_model.validation_step(batch, batch_indx)


def test_configure_optimizers(sed_model):
    optimizers, schedulers = sed_model.configure_optimizers()
    assert (
        optimizers is not None and schedulers is not None
    ), "Optimizers and/or schedulers not configured properly"


if __name__ == "__main__":
    pytest.main([__file__])
