import argparse

import pandas as pd
import torch
import yaml
import os

from desed_task.dataio import ConcatDatasetBatchSampler
from desed_task.dataio.datasets import StronglyAnnotatedSet, WeakSet, UnlabeledSet
from model import CRNN
from encoder import ManyHotEncoder
from desed_task.utils.schedulers import ExponentialWarmup
import randomname
from trainer import SED

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# from pytorch_lightning.loggers import TensorBoardLogger
from dvclive.lightning import DVCLiveLogger

from utils import classes_labels


def single_run(
    config,
    log_dir,
    gpus,
    real_data=False,
    checkpoint_resume=None,
    test_state_dict=None,
    fast_dev_run=False,
    evaluation=False,
    callbacks=None,
):
    """
    Running sound event detection training and testing.

    Args:
        config (dict): the dictionary of configuration params
        log_dir (str): path to log directory
        gpus (int): number of gpus to use
        checkpoint_resume (str, optional): path to checkpoint to resume from. Defaults to "".
        test_state_dict (dict, optional): if not None, no training is involved. This dictionary is the state_dict
            to be loaded to test the model.
        fast_dev_run (bool, optional): whether to use a run with only one batch at train and validation, useful
            for development purposes.
    """
    config.update({"log_dir": log_dir})

    # handle seed
    seed = config["training"]["seed"]
    if seed:
        pl.seed_everything(seed, workers=True)

    encoder = ManyHotEncoder(
        list(classes_labels.keys()),
        audio_len=config["data"]["audio_max_len"],
        frame_len=config["feats"]["n_filters"],
        frame_hop=config["feats"]["hop_length"],
        net_pooling=config["data"]["net_subsample"],
        fs=config["data"]["fs"],
    )

    #####* test data prep #####
    if not evaluation:
        devtest_df = pd.read_csv(config["data"]["test_tsv"], sep="\t")
        devtest_dataset = StronglyAnnotatedSet(
            config["data"]["test_folder"],
            devtest_df,
            encoder,
            return_filename=True,
            pad_to=config["data"]["audio_max_len"],
        )
    else:
        devtest_dataset = UnlabeledSet(
            config["data"]["eval_folder"], encoder, pad_to=None, return_filename=True
        )

    test_dataset = devtest_dataset

    #####* model definition #####
    sed = CRNN(**config["net"])

    # * if test_state_dict is not None, no training is involved and the model is tested

    if test_state_dict is None:
        #####* train, valid data prep #####
        synth_df = pd.read_csv(config["data"]["synth_tsv"], sep="\t")
        synth_set = StronglyAnnotatedSet(
            config["data"]["synth_folder"],
            synth_df,
            encoder,
            pad_to=config["data"]["audio_max_len"],
        )

        if real_data:
            real_df = pd.read_csv(config["data"]["strong_tsv"], sep="\t")
            real_set = StronglyAnnotatedSet(
                config["data"]["strong_folder"],
                real_df,
                encoder,
                pad_to=config["data"]["audio_max_len"],
            )

        weak_df = pd.read_csv(config["data"]["weak_tsv"], sep="\t")
        train_weak_df = weak_df.sample(
            frac=config["training"]["weak_split"],
            random_state=config["training"]["seed"],
        )
        valid_weak_df = weak_df.drop(train_weak_df.index).reset_index(drop=True)
        train_weak_df = train_weak_df.reset_index(drop=True)
        weak_set = WeakSet(
            config["data"]["weak_folder"],
            train_weak_df,
            encoder,
            pad_to=config["data"]["audio_max_len"],
        )
        
        strong_df_val = pd.read_csv(config["data"]["synth_val_tsv"], sep="\t")
        strong_val = StronglyAnnotatedSet(
            config["data"]["synth_val_folder"],
            strong_df_val,
            encoder,
            return_filename=True,
            pad_to=config["data"]["audio_max_len"],
        )

        weak_val = WeakSet(
            config["data"]["weak_folder"],
            valid_weak_df,
            encoder,
            pad_to=config["data"]["audio_max_len"],
            return_filename=True,
        )
        
        if real_data:
            synth_set = torch.utils.data.ConcatDataset([real_set, synth_set])

        tot_train_data = [synth_set, weak_set]
        train_dataset = torch.utils.data.ConcatDataset(tot_train_data)

        batch_sizes = config["training"]["batch_size"]
        samplers = [torch.utils.data.RandomSampler(x) for x in tot_train_data]
        batch_sampler = ConcatDatasetBatchSampler(samplers, batch_sizes)

        valid_dataset = torch.utils.data.ConcatDataset([strong_val, weak_val])
        
        #####* training params and optimizers #####
        epoch_len = min(
            [
                len(tot_train_data[indx])
                // (
                    config["training"]["batch_size"][indx]
                    * config["training"]["accumulate_batches"]
                )
                for indx in range(len(tot_train_data))
            ]
        )

        opt = torch.optim.Adam(
            sed.parameters(), config["opt"]["lr"], betas=(0.9, 0.999)
        )
        exp_steps = config["training"]["n_epochs_warmup"] * epoch_len
        exp_scheduler = {
            "scheduler": ExponentialWarmup(opt, config["opt"]["lr"], exp_steps),
            "interval": "step",
        }

        # logger = TensorBoardLogger(
        #     os.path.dirname(config["log_dir"]), config["log_dir"].split("/")[-1])

        logger = DVCLiveLogger(save_dvc_exp=True, log_model=True)

        logger.log_hyperparams(config)
        print(f"experiment dir: {logger.log_dir}")

        def generate_unique_model_name(checkpoint_dir):
            while True:
                model_name = randomname.get_name()
                # Check if there is any file that starts with the model_name
                if not any(f.startswith(model_name) for f in os.listdir(checkpoint_dir)):
                    return model_name
        model_name = generate_unique_model_name("dvclive/artifacts/")
        config.update({"model_name": model_name})
        
        if callbacks is None:
            callbacks = [
                EarlyStopping(
                    monitor="val/obj_metric",
                    patience=config["training"]["early_stop_patience"],
                    verbose=True,
                    mode="max",
                ),
                ModelCheckpoint(
                    logger.log_dir,
                    filename=model_name,
                    monitor="val/obj_metric",
                    save_top_k=1,
                    mode="max",
                    save_last=True,
                ),
            ]
    else:
        train_dataset = None
        valid_dataset = None
        opt = None
        exp_scheduler = None
        logger = True
        callbacks = None

    #####* training #####

    sed_model = SED(
        config,
        encoder=encoder,
        sed=sed,
        opt=opt,
        train_data=train_dataset,
        valid_data=valid_dataset,
        test_data=test_dataset,
        train_sampler=batch_sampler,
        scheduler=exp_scheduler,
        fast_dev_run=fast_dev_run,
        evaluation=evaluation,
    )

    if fast_dev_run:
        log_every_n_steps = 1
        limit_train_batches = 2
        limit_val_batches = 2
        limit_test_batches = 2
        n_epochs = 3
    else:
        log_every_n_steps = 40
        limit_train_batches = 1.0
        limit_val_batches = 1.0
        limit_test_batches = 1.0
        n_epochs = config["training"]["n_epochs"]

    if gpus == "0":
        accelerator = "cpu"
    elif gpus == "1":
        accelerator = "gpu"
    else:
        raise NotImplementedError()

    trainer = pl.Trainer(
        precision=config["training"]["precision"],
        max_epochs=n_epochs,
        callbacks=callbacks,
        accelerator=accelerator,
        devices=1,
        strategy=config["training"].get("backend"),
        accumulate_grad_batches=config["training"]["accumulate_batches"],
        logger=logger,
        gradient_clip_val=config["training"]["gradient_clip"],
        check_val_every_n_epoch=config["training"]["validation_interval"],
        num_sanity_val_steps=0,
        log_every_n_steps=log_every_n_steps,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        deterministic=config["training"]["deterministic"],
        enable_progress_bar=config["training"]["enable_progress_bar"],
    )
    if test_state_dict is None:
        trainer.fit(sed_model, ckpt_path=checkpoint_resume)
        best_path = trainer.checkpoint_callback.best_model_path
        print(f"best model: {best_path}")
        test_state_dict = torch.load(best_path)["state_dict"]

    sed_model.load_state_dict(test_state_dict)
    trainer.test(sed_model)


def prepare_run(argv=None):
    parser = argparse.ArgumentParser("Training a SED system")
    parser.add_argument(
        "--conf_file",
        default="params.yaml",
        help="The configuration file with all the experiment parameters.",
    )
    parser.add_argument(
        "--log_dir",
        default="./exp/",
        help="Directory where to save logs, saved models, etc.",
    )
    parser.add_argument(
        "--real_data",
        action="store_true",
        default=False,
        help="The strong annotations coming from Audioset will be included in the training phase.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Allow the training to be resumed, take as input a previously saved model (.ckpt).",
    )
    parser.add_argument(
        "--test_from_checkpoint", default=None, help="Test the model specified."
    )
    parser.add_argument(
        "--gpus",
        default="0",
        help="The number of GPUs to train on, or the gpu to use, default='1', "
        "so uses one GPU",
    )
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        default=False,
        help="Use this option to make a 'fake' run which is useful for development and debugging. "
        "It uses very few batches and epochs so it won't give any meaningful result.",
    )
    parser.add_argument(
        "--eval_from_checkpoint", default=None, help="Evaluate the model specified"
    )

    args = parser.parse_args(argv)

    with open(args.conf_file, "r") as f:
        configs = yaml.safe_load(f)

    evaluation = False
    test_from_checkpoint = args.test_from_checkpoint

    if args.eval_from_checkpoint is not None:
        test_from_checkpoint = args.eval_from_checkpoint
        evaluation = True

    test_model_state_dict = None
    if test_from_checkpoint is not None:
        if args.gpus == "0":
            checkpoint = torch.load(test_from_checkpoint, map_location="cpu")
        else:
            checkpoint = torch.load(test_from_checkpoint)
        configs_ckpt = checkpoint["hyper_parameters"]
        configs_ckpt["data"] = configs["data"]
        print(
            f"loaded model: {test_from_checkpoint} \n"
            f"at epoch: {checkpoint['epoch']}"
        )
        test_model_state_dict = checkpoint["state_dict"]

    if evaluation:
        configs["training"]["batch_size_val"] = 1

    return configs, args, test_model_state_dict, evaluation


if __name__ == "__main__":
    # * prepare run
    configs, args, test_model_state_dict, evaluation = prepare_run()

    # * launch run
    single_run(
        configs,
        args.log_dir,
        args.gpus,
        args.real_data,
        args.resume_from_checkpoint,
        test_model_state_dict,
        args.fast_dev_run,
        evaluation,
    )
