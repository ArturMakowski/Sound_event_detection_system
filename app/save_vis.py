import warnings
from tqdm import tqdm
from app import (
    create_dataset,
    initialize_sed,
    load_config,
    load_model,
    make_prediction,
)
from multiprocessing import Pool, cpu_count

warnings.filterwarnings("ignore")


def process_item(idx_item_tuple):
    idx, item = idx_item_tuple
    try:
        config = load_config()
        sed, encoder = initialize_sed(config)
        sed = load_model(sed, model_path="../dvclive/artifacts/critical-bond.ckpt")
        dataset_strong = create_dataset(config, encoder)
        vis, _ = make_prediction(sed, dataset_strong, config, encoder, idx, False)
        vis.save(filename=f"vis_images/{idx}_vis.png")
        return f"-------- Saved .png visualization for sample {idx + 1} --------"
    except Exception as e:
        return f"Failed to save visualization for sample {idx + 1}: {e}"


def save_visualizations_concurrent(
    save_dir: str = "vis_images",
) -> None:
    config = load_config()
    _, encoder = initialize_sed(config)
    dataset_strong = create_dataset(config, encoder)

    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_item, enumerate(dataset_strong)), total=len(dataset_strong)))

    for result in results:
        print(result)


if __name__ == "__main__":
    save_visualizations_concurrent()
