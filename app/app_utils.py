from pathlib import Path
from typing import Tuple

import dcase_util
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sed_vis
import sed_vis.visualization
import torch
import yaml

from desed_task.dataio.datasets import StronglyAnnotatedSet
from encoder import ManyHotEncoder
from model import CRNN
from trainer import SED
from utils import batched_decode_preds, classes_labels


def load_config():
    """
    Loads the configuration parameters from a yaml file.

    Returns:
        dict: The configuration parameters.
    """
    with open("../params.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config


def initialize_sed(config: dict) -> Tuple[SED, ManyHotEncoder]:
    """
    Initializes the Sound Event Detection (SED) model and the encoder.

    Parameters:
        config (dict): The configuration parameters.

    Returns:
        tuple: The initialized SED model and encoder.
    """
    encoder = ManyHotEncoder(
        list(classes_labels.keys()),
        audio_len=config["data"]["audio_max_len"],
        frame_len=config["feats"]["n_filters"],
        frame_hop=config["feats"]["hop_length"],
        net_pooling=config["data"]["net_subsample"],
        fs=config["data"]["fs"],
    )

    sed = SED(config, encoder=encoder, sed=CRNN(**config["net"]))
    return sed, encoder


def load_model(
    sed: SED, model_path: str = "../dvclive/artifacts/epoch=68-step=7314.ckpt"
) -> SED:
    """
    Loads a pretrained SED model from a checkpoint file.

    Parameters:
        sed (SED): The SED model to load the weights into.
        model_path (str): The path to the checkpoint file.

    Returns:
        SED: The SED model with loaded weights.
    """
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))["state_dict"]
    sed.load_state_dict(state_dict)
    return sed


def create_dataset(config: dict, encoder: ManyHotEncoder) -> StronglyAnnotatedSet:
    """
    Creates a dataset for testing.

    Parameters:
            config (dict): The configuration parameters.
            encoder (ManyHotEncoder): The encoder for the dataset.

    Returns:
            StronglyAnnotatedSet: The created dataset.
    """
    tsv_entries_strong = pd.read_csv(config["data"]["test_tsv"], sep="\t")
    dataset_strong = StronglyAnnotatedSet(
        audio_folder=config["data"]["test_folder"],
        tsv_entries=tsv_entries_strong,
        encoder=encoder,
        return_filename=True,
        pad_to=config["data"]["audio_max_len"],
    )
    return dataset_strong


def make_prediction(
    sed: SED,
    dataset_strong: StronglyAnnotatedSet,
    config: dict,
    encoder: ManyHotEncoder,
    idx: int,
    print_df: bool = False,
    model: str = "student",
) -> Tuple[
    sed_vis.visualization.EventListVisualizer, dcase_util.containers.AudioContainer
]:
    """
    Makes a prediction on a sample from the dataset.

    Parameters:
        sed (SED): The SED model to use for prediction.
        dataset_strong (StronglyAnnotatedSet): The dataset to use.
        config (dict): The configuration parameters.
        encoder (ManyHotEncoder): The encoder for the dataset.
        idx (int): The index of the sample in the dataset to predict.
        print_df (bool, optional): Whether to print the dataframes. Defaults to False.

    Returns:
        tuple: The visualizer for the event list and the audio container.
    """
    preds = batched_decode_preds(
        sed.forward(dataset_strong[idx][0], model=model),
        dataset_strong[idx][3],
        encoder,
    )
    preds[2][0.5][["onset", "offset", "event_label"]].to_csv(
        "preds.tsv", index=False, sep="\t"
    )

    if print_df:
        print("--------Preds--------")
        print(preds[2][0.5][["onset", "offset", "event_label"]])

    ref_df = pd.read_csv(config["data"]["test_tsv"], sep="\t")
    ref_df = (
        ref_df[ref_df["filename"] == dataset_strong[idx][3].split("/")[-1]]
        .reset_index()
        .drop(columns=["index"])
    )
    ref_df.drop("filename", axis=1).to_csv("ref.tsv", index=False, sep="\t")

    if print_df:
        print("--------Ref--------")
        print(ref_df)

    audio_container = dcase_util.containers.AudioContainer().load(
        f"/mnt/d/DESED_dataset/audio/test/public/{dataset_strong[idx][3].split('/')[-1]}"
    )
    audio_container.data = np.mean(audio_container.data, axis=0)

    # Load event lists
    reference_event_list = dcase_util.containers.MetaDataContainer().load("ref.tsv")
    estimated_event_list = dcase_util.containers.MetaDataContainer().load("preds.tsv")

    event_lists = {
        "reference": reference_event_list,
        "estimated": estimated_event_list,
    }

    vis = sed_vis.visualization.EventListVisualizer(
        event_lists=event_lists,
        audio_signal=audio_container.data,
        sampling_rate=audio_container.fs,
        publication_mode=True,
    )
    return vis, audio_container


def inference(
    idx: int,
    model_path: str = "../dvclive/artifacts/epoch=68-step=7314.ckpt",
    print_df: bool = False,
    **kwargs: str,
) -> Tuple[sed_vis.visualization.EventListVisualizer, np.ndarray, int]:
    """
    Performs inference on a sample from the dataset.

    Parameters:
        idx (int): The index of the sample in the dataset to infer.
        model_path (str, optional): The path to the checkpoint file. Defaults to "../dvclive/artifacts/epoch=68-step=7314.ckpt".
        print_df (bool, optional): Whether to print the dataframes. Defaults to False.

    Returns:
        tuple: The visualizer for the event list, the audio data, and the sampling rate.
    """
    config = load_config()
    sed, encoder = initialize_sed(config)
    sed = load_model(sed, model_path)
    dataset_strong = create_dataset(config, encoder)
    vis, audio_container = make_prediction(
        sed, dataset_strong, config, encoder, idx, print_df, **kwargs
    )
    return vis, audio_container.data, audio_container.fs


def id_filename_mapping(idx: int, dataset_strong: StronglyAnnotatedSet) -> Path:
    return Path(dataset_strong[idx][3].split("/")[-1])


def plot_scores(idx: int) -> plt.Axes:
    config = load_config()
    _, encoder = initialize_sed(config)
    dataset_strong = create_dataset(config, encoder)

    paths = [
        Path(
            "/home/armak/Python_projects_WSL/Master_SED/exp/metrics_test/scenario1/scores"
        )
        / id_filename_mapping(idx, dataset_strong).with_suffix(".tsv"),
        Path(
            "/home/armak/Python_projects_WSL/Master_SED/exp/metrics_test/scenario2/scores"
        )
        / id_filename_mapping(idx, dataset_strong).with_suffix(".tsv"),
    ]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    titles = ["PSDS 1", "PSDS 2"]

    for path_id, path in enumerate(paths):
        df = pd.read_csv(path, sep="\t")
        df = df.drop(columns=["onset", "offset"])

        df.plot(ax=axes[path_id])
        axes[path_id].grid()
        axes[path_id].set_title(titles[path_id])
        axes[path_id].set_ylabel("Probability")
        axes[path_id].set_xlabel("Frame")
    return fig


def make_prediction_on_audio(
    sed: SED,
    audio: torch.Tensor,
    config: dict,
    encoder: ManyHotEncoder,
    filename: str,
    print_df: bool = False,
    model: str = "student",
    median_filter: int = 7
) -> Tuple[
    sed_vis.visualization.EventListVisualizer, dcase_util.containers.AudioContainer
]:
    """
    Makes a prediction on a provided audio tensor.

    Parameters:
        sed (SED): The SED model to use for prediction.
        audio (torch.Tensor): The audio data as a tensor.
        config (dict): The configuration parameters.
        encoder (ManyHotEncoder): The encoder for the dataset.
        filename (str): The filename associated with the audio tensor.
        print_df (bool, optional): Whether to print the dataframes. Defaults to False.

    Returns:
        tuple: The visualizer for the event list and the audio container.
    """
    preds = batched_decode_preds(
        sed.forward(audio, model=model),
        filename,
        encoder,
        median_filter=median_filter
    )
    preds[2][0.5][["onset", "offset", "event_label"]].to_csv(
        "preds.tsv", index=False, sep="\t"
    )

    if print_df:
        print("--------Preds--------")
        print(preds[2][0.5][["onset", "offset", "event_label"]])

    ref_df = pd.read_csv(config["data"]["test_tsv"], sep="\t")
    ref_df = (
        ref_df[ref_df["filename"] == filename].reset_index().drop(columns=["index"])
    )
    ref_df.drop("filename", axis=1).to_csv("ref.tsv", index=False, sep="\t")

    if print_df:
        print("--------Ref--------")
        print(ref_df)

    # Create an AudioContainer from the audio tensor
    audio_container = dcase_util.containers.AudioContainer(
        data=audio.numpy(),
        fs=config["data"]["fs"],
    )

    # Load event lists
    reference_event_list = dcase_util.containers.MetaDataContainer().load("ref.tsv")
    estimated_event_list = dcase_util.containers.MetaDataContainer().load("preds.tsv")

    event_lists = {
        "reference": reference_event_list,
        "estimated": estimated_event_list,
    }

    vis = sed_vis.visualization.EventListVisualizer(
        event_lists=event_lists,
        audio_signal=audio_container.data,
        sampling_rate=audio_container.fs,
        publication_mode=True,
    )
    return vis, audio_container
