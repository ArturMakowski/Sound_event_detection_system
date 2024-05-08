import warnings

import gradio as gr
import torch
import torchaudio
from app_utils import initialize_sed, load_config, load_model, make_prediction_on_audio

warnings.filterwarnings("ignore")


def gradio_inference(audio_file: str) -> str:
    """Perform sound event detection on an audio file and return the event plot.

    Args:
        audio_file (str): Path to the audio file.

    Returns:
        str: Path to the event plot.
    """
    config = load_config()
    sed, encoder = initialize_sed(config)
    sed = load_model(sed, model_path="../dvclive/artifacts/chalky-protagonist.ckpt")

    audio_tensor, sr = torchaudio.load(audio_file)

    if audio_tensor.shape[0] == 2:
        audio_tensor = torch.mean(audio_tensor, dim=0)

    audio_tensor = audio_tensor.squeeze(0)

    if sr != config["data"]["fs"]:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=config["data"]["fs"]
        )
        audio_tensor = resampler(audio_tensor)

    vis, audio_container = make_prediction_on_audio(
        sed,
        audio_tensor,
        config,
        encoder,
        filename=audio_file,
        print_df=False,
        median_filter=24,
    )

    return_filename = "event_plot.png"

    vis.save(filename=return_filename)

    return return_filename


iface = gr.Interface(
    fn=gradio_inference,
    inputs=gr.Audio(type="filepath", label="Upload Audio File"),
    outputs=[
        gr.Image(type="filepath", label="Event Plot"),
    ],
    title="Sound Event Detection Inference",
    description="Upload an audio file to perform sound event detection.",
)

if __name__ == "__main__":
    iface.launch()
