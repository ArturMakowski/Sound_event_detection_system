import gradio as gr
from app_utils import load_config, initialize_sed, load_model, make_prediction_on_audio
import torch
import torchaudio

# Assuming all necessary imports and functions from your provided code are already here

def gradio_inference(audio_file):
    config = load_config()
    sed, encoder = initialize_sed(config)
    sed = load_model(sed, model_path="../dvclive/artifacts/chalky-protagonist.ckpt")

    # Read the audio file and convert to tensor
    audio_tensor, sr = torchaudio.load(audio_file)

    # If stereo, take the mean to convert to mono
    if audio_tensor.shape[0] == 2:
        audio_tensor = torch.mean(audio_tensor, dim=0)
        
    audio_tensor = audio_tensor.squeeze(0)
    print(audio_tensor.shape)
    print(sr)

    # Resample if necessary
    if sr != config["data"]["fs"]:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=config["data"]["fs"])
        audio_tensor = resampler(audio_tensor)
        

    vis, audio_container = make_prediction_on_audio(
        sed,
        audio_tensor,
        config,
        encoder,
        filename=audio_file,
        print_df=False
    )

    # Generate and save the plot
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
    description="Upload an audio file to perform sound event detection."
)

if __name__ == "__main__":
    iface.launch()
