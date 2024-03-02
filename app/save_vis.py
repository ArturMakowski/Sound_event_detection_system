from app import (
    load_config,
    initialize_sed,
    load_model,
    create_dataset,
    make_prediction,
)
import warnings

warnings.filterwarnings("ignore")


def save_visualizations(
    save_dir: str = "vis_images",
) -> None:
    config = load_config()
    sed, encoder = initialize_sed(config)
    sed = load_model(sed, model_path="../dvclive/artifacts/epoch=96-step=10282.ckpt")
    dataset_strong = create_dataset(config, encoder)
    for idx, item in enumerate(dataset_strong):
        try:
            vis, _ = make_prediction(sed, dataset_strong, config, encoder, idx, False)
            vis.save(filename=f"{save_dir}/{idx}_vis.png")
        except Exception:
            pass


if __name__ == "__main__":
    save_visualizations()
