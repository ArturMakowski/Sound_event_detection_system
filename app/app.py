import base64

import sys

import dcase_util
import numpy as np
import pandas as pd
import sed_vis
import streamlit as st
import torch
import yaml

sys.path.append("../")
from desed_task.dataio.datasets import StronglyAnnotatedSet
from encoder import ManyHotEncoder
from model import CRNN
from trainer import SED
from utils import batched_decode_preds, classes_labels


def inference(
    idx, model_path="../dvclive/artifacts/epoch=68-step=7314.ckpt", print_df=False
):
    with open("../params.yaml", "r") as f:
        config = yaml.safe_load(f)

    encoder = ManyHotEncoder(
        list(classes_labels.keys()),
        audio_len=config["data"]["audio_max_len"],
        frame_len=config["feats"]["n_filters"],
        frame_hop=config["feats"]["hop_length"],
        net_pooling=config["data"]["net_subsample"],
        fs=config["data"]["fs"],
    )

    sed = SED(config, encoder=encoder, sed=CRNN(**config["net"]))
    # print(sed.state_dict().keys())

    state_dict = torch.load(model_path, map_location=torch.device("cpu"))["state_dict"]
    sed.load_state_dict(state_dict)

    tsv_entries_strong = pd.read_csv(config["data"]["test_tsv"], sep="\t")
    dataset_strong = StronglyAnnotatedSet(
        audio_folder=config["data"]["test_folder"],
        tsv_entries=tsv_entries_strong,
        encoder=encoder,
        return_filename=True,
        pad_to=config["data"]["audio_max_len"],
    )

    preds = batched_decode_preds(
        sed(dataset_strong[idx][0]), dataset_strong[idx][3], encoder
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
        f"/mnt/d/DESED_dataset/audio/validation/validation_16k/{dataset_strong[idx][3].split('/')[-1]}"
    )
    audio_container.data = np.mean(audio_container.data, axis=0)

    # Load event lists
    reference_event_list = dcase_util.containers.MetaDataContainer().load("ref.tsv")
    estimated_event_list = dcase_util.containers.MetaDataContainer().load("preds.tsv")

    event_lists = {"reference": reference_event_list, "estimated": estimated_event_list}

    vis = sed_vis.visualization.EventListVisualizer(
        event_lists=event_lists,
        audio_signal=audio_container.data,
        sampling_rate=audio_container.fs,
        publication_mode=True,
    )

    return vis, audio_container.data, audio_container.fs


def main():
    st.title("SED Visualization App")

    idx = st.number_input("Enter the index:", value=5)

    try:
        vis, audio, fs = inference(
            idx, model_path="../dvclive/artifacts/epoch=58-step=12449.ckpt"
        )
    except IndexError:
        st.error("Invalid index. Please enter a valid index.")
        return

    vis.save(filename="vis.png")

    with open("vis.png", "rb") as img_file:
        b64_string = base64.b64encode(img_file.read()).decode()

    st.markdown(
        f'<div style="display: flex; justify-content: center;"><img src="data:image/png;base64,{b64_string}"/></div>',
        unsafe_allow_html=True,
    )

    st.audio(audio, format="audio/wav", start_time=0, sample_rate=fs)


if __name__ == "__main__":
    main()
