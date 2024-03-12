import warnings
from tqdm import tqdm
from app import (
    create_dataset,
    initialize_sed,
    load_config,
    load_model,
    make_prediction,
)

warnings.filterwarnings("ignore")


def save_visualizations(
    save_dir: str = "vis_images",
) -> None:
    config = load_config()
    sed, encoder = initialize_sed(config)
    sed = load_model(sed, model_path="../dvclive/artifacts/critical-bond.ckpt")
    dataset_strong = create_dataset(config, encoder)
    pbar = tqdm(enumerate(dataset_strong), total=len(dataset_strong))
    for idx, item in pbar:
        try:
            vis, _ = make_prediction(sed, dataset_strong, config, encoder, idx, False)
            vis.save(filename=f"{save_dir}/{idx}_vis.png")
        except Exception:
            pass
        else:
            pbar.set_description(
                f"-------- Saved .png visualization for sample {idx + 1} --------"
            )


if __name__ == "__main__":
    save_visualizations()
