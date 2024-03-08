import os
import random
from pathlib import Path
import yaml

import pandas as pd
import pytorch_lightning as pl
import torch
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram

from desed_task.data_augm import mixup
from desed_task.utils.scaler import TorchScaler
from encoder import ManyHotEncoder
import numpy as np

from utils import (
    batched_decode_preds,
    log_sedeval_metrics,
)
from desed_task.evaluation.evaluation_measures import (
    compute_per_intersection_macro_f1,
    compute_psds_from_operating_points,
    compute_psds_from_scores,
)

import sed_scores_eval

from utils import classes_labels
from model import CRNN


class SED(pl.LightningModule):

    """Pytorch lightning module for the SED task.
    Args:
        hparams: dict, the dictionary to be used for the current experiment.
        encoder: ManyHotEncoder object, object to encode and decode labels.
        sed: torch.Module, the  model to be trained.
        opt: torch.optimizer.Optimizer object, the optimizer to be used.
        train_data: torch.utils.data.Dataset subclass object, the training data to be used.
        valid_data: torch.utils.data.Dataset subclass object, the validation data to be used.
        test_data: torch.utils.data.Dataset subclass object, the test data to be used.
        train_sampler: torch.utils.data.Sampler subclass object, the sampler to be used in the training dataloader.
        scheduler: BaseScheduler subclass object, the scheduler to be used.
                   This is used to apply ramp-up during training for example.
        fast_dev_run: bool, whether to launch a run with only one batch for each set, this is for development purpose,
                   to test the code runs.
    """

    def __init__(
        self,
        hparams,
        encoder,
        sed,
        opt=None,
        train_data=None,
        valid_data=None,
        test_data=None,
        train_sampler=None,
        scheduler=None,
        fast_dev_run=False,
        evaluation=False,
    ):
        super(SED, self).__init__()
        self.hparams.update(hparams)

        self.encoder = encoder
        self.sed = sed
        self.opt = opt
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.train_sampler = train_sampler
        self.scheduler = scheduler
        self.fast_dev_run = fast_dev_run
        self.evaluation = evaluation

        if self.fast_dev_run:
            self.num_workers = 1
        else:
            self.num_workers = self.hparams["training"]["num_workers"]

        feat_params = self.hparams["feats"]

        self.mel_spec = MelSpectrogram(
            sample_rate=feat_params["sample_rate"],
            n_fft=feat_params["n_window"],
            win_length=feat_params["n_window"],
            hop_length=feat_params["hop_length"],
            f_min=feat_params["f_min"],
            f_max=feat_params["f_max"],
            n_mels=feat_params["n_mels"],
            window_fn=torch.hamming_window,
            wkwargs={"periodic": False},
            power=1,
            center=False,
        )

        # * instantiating loss fn and scaler
        self.loss_fn = torch.nn.BCELoss()

        self.get_weak_f1_seg_macro = pl.metrics.classification.F1( # type: ignore
            len(self.encoder.labels),
            average="macro",
            multilabel=True,
            compute_on_step=False,
        )

        self.scaler = self._init_scaler()

        # * buffer for event based scores which we compute using sed-eval
        self.val_buffer_synth = {
            k: pd.DataFrame() for k in self.hparams["training"]["val_thresholds"]
        }

        self.val_buffer_test = {
            k: pd.DataFrame() for k in self.hparams["training"]["val_thresholds"]
        }

        self.val_scores_postprocessed_buffer_synth = {}

        test_n_thresholds = self.hparams["training"]["n_test_thresholds"]
        test_thresholds = np.arange(
            1 / (test_n_thresholds * 2), 1, 1 / test_n_thresholds
        )
        self.test_psds_buffer = {k: pd.DataFrame() for k in test_thresholds}
        self.decoded_05_buffer = pd.DataFrame()
        self.test_scores_raw_buffer = {}
        self.test_scores_postprocessed_buffer = {}

    _exp_dir = None

    @property
    def exp_dir(self):
        if self._exp_dir is None:
            try:
                self._exp_dir = self.logger.log_dir
                if self._exp_dir is None:
                    self._exp_dir = self.hparams["log_dir"]
            except Exception:
                self._exp_dir = self.hparams["log_dir"]
        return self._exp_dir

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step()

    def _init_scaler(self):
        """Scaler inizialization function. It can be either a dataset or instance scaler.

        Raises:
            NotImplementedError: in case of not Implemented scaler

        Returns:
            TorchScaler: returns the scaler
        """

        if self.hparams["scaler"]["statistic"] == "instance":
            scaler = TorchScaler(
                "instance",
                self.hparams["scaler"]["normtype"],
                self.hparams["scaler"]["dims"],
            )
            return scaler
        elif self.hparams["scaler"]["statistic"] == "dataset":
            scaler = TorchScaler(
                "dataset",
                self.hparams["scaler"]["normtype"],
                self.hparams["scaler"]["dims"],
            )
        else:
            raise NotImplementedError

        if self.hparams["scaler"]["savepath"] is not None:
            if os.path.exists(self.hparams["scaler"]["savepath"]):
                scaler = torch.load(self.hparams["scaler"]["savepath"])
                print(
                    "Loaded Scaler from previous checkpoint from {}".format(
                        self.hparams["scaler"]["savepath"]
                    )
                )
                return scaler

        self.train_loader = self.train_dataloader()
        scaler.fit(
            self.train_loader,
            transform_func=lambda x: self.take_log(self.mel_spec(x[0])),
        )

        if self.hparams["scaler"]["savepath"] is not None:
            torch.save(scaler, self.hparams["scaler"]["savepath"])
            print(
                "Saving Scaler from previous checkpoint at {}".format(
                    self.hparams["scaler"]["savepath"]
                )
            )
            return scaler

    def take_log(self, mels):
        """Apply the log transformation to mel spectrograms.
        Args:
            mels: torch.Tensor, mel spectrograms for which to apply log.

        Returns:
            Tensor: logarithmic mel spectrogram of the mel spectrogram given as input
        """

        amp_to_db = AmplitudeToDB(stype="amplitude")
        amp_to_db.amin = 1e-5  # amin= 1e-5 as in librosa
        return amp_to_db(mels).clamp(min=-50, max=80)

    def training_step(self, batch, batch_indx):
        """Apply the training for one batch (a step). Used during trainer.fit

        Args:
            batch: torch.Tensor, batch input tensor
            batch_indx: torch.Tensor, 1D tensor of indexes to know which data are present in each batch.

        Returns:
           torch.Tensor, the loss to take into account.
        """

        indx_synth, indx_weak = self.hparams["training"]["batch_size"]
        
        audio, labels, _ = batch
        features = self.mel_spec(audio)

        batch_num = features.shape[0]
        # deriving masks for each dataset
        strong_mask = torch.zeros(batch_num).to(features).bool()
        weak_mask = torch.zeros(batch_num).to(features).bool()
        strong_mask[:indx_synth] = 1
        weak_mask[indx_synth : indx_weak + indx_synth] = 1
        
        labels_weak = (torch.sum(labels[weak_mask], -1) > 0).float()
        
        mixup_type = self.hparams["training"].get("mixup")
        if mixup_type is not None and 0.5 > random.random():
            
            features[weak_mask], labels_weak = mixup(
                features[weak_mask], labels_weak, mixup_label_type=mixup_type
            )
            features[strong_mask], labels[strong_mask] = mixup(
                features[strong_mask], labels[strong_mask], mixup_label_type=mixup_type
            )

        
        strong_preds, weak_preds = self.sed(self.scaler(self.take_log(features)))

        loss_strong = self.loss_fn(
            strong_preds[strong_mask], labels[strong_mask]
        )
        # supervised loss on weakly labelled
        loss_weak = self.loss_fn(weak_preds[weak_mask], labels_weak)
        
        # total supervised loss
        total_loss = loss_strong + loss_weak

        self.log("train/loss_strong", loss_strong, prog_bar=True, sync_dist=True)
        self.log("train/loss_weak", loss_weak, prog_bar=True, sync_dist=True)
        self.log("train/total_loss", total_loss, prog_bar=True, sync_dist=True)
        self.log(
            "train/step",
            self.scheduler["scheduler"].step_num,
            prog_bar=True,
            sync_dist=True,
        )
        self.log("train/lr", self.opt.param_groups[-1]["lr"], sync_dist=True)

        return total_loss

    def validation_step(self, batch, batch_indx):
        """Apply validation to a batch (step). Used during trainer.fit

        Args:
            batch: torch.Tensor, input batch tensor
            batch_indx: torch.Tensor, 1D tensor of indexes to know which data are present in each batch.
        Returns:
        """

        audio, labels, _, filenames = batch

        features = self.mel_spec(audio)
        strong_preds, weak_preds = self.sed(self.scaler(self.take_log(features)))

        loss_strong = self.loss_fn(strong_preds, labels)

        weak_mask = (
            torch.tensor(
                [
                    str(Path(x).parent)
                    == str(Path(self.hyparams["data"]["weak_folder"]))
                    for x in filenames
                ]
            )
            .to(audio)
            .bool()
        )
        strong_mask = (
            torch.tensor(
                [
                    str(Path(x).parent)
                    == str(Path(self.hyparams["data"]["synth_val_folder"]))
                    for x in filenames
                ]
            )
            .to(audio)
            .bool()
        )
        
        if torch.any(weak_mask):
            labels_weak = (torch.sum(labels[weak_mask], -1) >= 1).float()
            loss_weak = self.loss_fn(
                weak_preds[weak_mask], labels_weak
            )
            self.log("val/synth/loss_weak", loss_weak, prog_bar=True)
            self.get_weak_f1_seg_macro(
                weak_preds[weak_mask], labels_weak
            )
            
        if torch.any(strong_mask):
            loss_strong = self.loss_fn(
                strong_preds[strong_mask], labels[strong_mask]
            )
            self.log("val/synth/loss_strong", loss_strong, prog_bar=True)

            filenames_synth = [
                x
                for x in filenames
                if Path(x).parent == Path(self.hparams["data"]["synth_val_folder"])
            ]

            (
                scores_raw_strong,
                scores_postprocessed_strong,
                decoded_strong,
            ) = batched_decode_preds(
                strong_preds[strong_mask],
                filenames_synth,
                self.encoder,
                median_filter=self.hparams["training"]["median_window"],
                thresholds=list(self.val_buffer_synth.keys()),
            )

            self.val_scores_postprocessed_buffer_synth.update(scores_postprocessed_strong)
            for th in self.val_buffer_synth.keys():
                self.val_buffer_synth[th] = pd.concat(
                    [self.val_buffer_synth[th], decoded_strong[th]], ignore_index=True
                )

        return

    def validation_epoch_end(self, outputs):
        """Function applied at the end of all the validation steps of the epoch.

        Args:
            outputs: torch.Tensor, the concatenation of everything returned by validation_step.

        Returns:
            torch.Tensor, the objective metric to be used to choose the best model from for example.
        """

        weak_f1_macro = self.get_weak_f1_seg_macro.compute()
        
        # * synth val dataset
        ground_truth = sed_scores_eval.io.read_ground_truth_events(
            self.hparams["data"]["synth_val_tsv"]
        )
        audio_durations = sed_scores_eval.io.read_audio_durations(
            self.hparams["data"]["synth_val_dur"]
        )
        if self.fast_dev_run:
            ground_truth = {
                audio_id: ground_truth[audio_id]
                for audio_id in self.val_scores_postprocessed_buffer_synth
            }
            audio_durations = {
                audio_id: audio_durations[audio_id]
                for audio_id in self.val_scores_postprocessed_buffer_synth
            }
        else:
            # * drop audios without events
            ground_truth = {
                audio_id: gt for audio_id, gt in ground_truth.items() if len(gt) > 0
            }
            audio_durations = {
                audio_id: audio_durations[audio_id] for audio_id in ground_truth.keys()
            }
        psds1_sed_scores_eval = compute_psds_from_scores(
            self.val_scores_postprocessed_buffer_synth,
            ground_truth,
            audio_durations,
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            cttc_threshold=None,
            alpha_ct=0,
            alpha_st=1,
            # save_dir=os.path.join(save_dir, "", "scenario1"),
        )
        intersection_f1_macro = compute_per_intersection_macro_f1(
            self.val_buffer_synth,
            self.hparams["data"]["synth_val_tsv"],
            self.hparams["data"]["synth_val_dur"],
        )
        synth_event_macro = log_sedeval_metrics(
            self.val_buffer_synth[0.5],
            self.hparams["data"]["synth_val_tsv"],
        )[0]

        obj_metric_synth_type = self.hparams["training"].get("obj_metric_synth_type")
        if obj_metric_synth_type is None:
            synth_metric = psds1_sed_scores_eval
        elif obj_metric_synth_type == "event":
            synth_metric = synth_event_macro
        elif obj_metric_synth_type == "intersection":
            synth_metric = intersection_f1_macro
        elif obj_metric_synth_type == "psds":
            synth_metric = psds1_sed_scores_eval
        else:
            raise NotImplementedError(
                f"obj_metric_synth_type: {obj_metric_synth_type} not implemented."
            )

        obj_metric = torch.tensor(weak_f1_macro.item() + synth_metric)

        self.log("val/obj_metric", obj_metric, prog_bar=True, sync_dist=True)
        self.log("val/weak/macro_F1", weak_f1_macro)
        self.log(
            "val/synth/psds1_sed_scores_eval",
            psds1_sed_scores_eval,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val/synth/intersection_f1_macro",
            intersection_f1_macro,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val/synth/event_f1_macro", synth_event_macro, prog_bar=True, sync_dist=True
        )

        # * free the buffers
        self.val_buffer_synth = {
            k: pd.DataFrame() for k in self.hparams["training"]["val_thresholds"]
        }
        self.val_scores_postprocessed_buffer_synth = {}
        
        self.get_weak_f1_seg_macro.reset()

        return obj_metric

    def on_save_checkpoint(self, checkpoint):
        checkpoint["sed"] = self.sed.state_dict()
        return checkpoint

    def test_step(self, batch, batch_indx):
        """Apply Test to a batch (step), used only when (trainer.test is called)

        Args:
            batch: torch.Tensor, input batch tensor
            batch_indx: torch.Tensor, 1D tensor of indexes to know which data are present in each batch.
        Returns:
        """

        audio, labels, _, filenames = batch

        features = self.mel_spec(audio)
        preds, _ = self.sed(self.scaler(self.take_log(features)))

        if not self.evaluation:
            loss = self.loss_fn(preds, labels)

            self.log("test/loss", loss)

        # * compute psds (Polyphonic Sound Detection Score)
        (
            scores_raw_strong,
            scores_postprocessed_strong,
            decoded_strong,
        ) = batched_decode_preds(
            preds,
            filenames,
            self.encoder,
            median_filter=self.hparams["training"]["median_window"],
            thresholds=list(self.test_psds_buffer.keys()) + [0.5],
        )

        self.test_scores_raw_buffer.update(scores_raw_strong)
        self.test_scores_postprocessed_buffer.update(scores_postprocessed_strong)
        for th in self.test_psds_buffer.keys():
            self.test_psds_buffer[th] = pd.concat(
                [self.test_psds_buffer[th], decoded_strong[th]], ignore_index=True
            )

        self.decoded_05_buffer = pd.concat(
            [self.decoded_05_buffer, decoded_strong[0.5]]
        )

    def on_test_epoch_end(self):
        save_dir = os.path.join(self.exp_dir, "metrics_test")

        # * if evaluation is True, we only save the scores
        if self.evaluation:
            save_dir_raw = os.path.join(save_dir, "_scores", "raw")
            sed_scores_eval.io.write_sed_scores(
                self.test_scores_raw_buffer, save_dir_raw
            )
            print(f"\nRaw scores for  saved in: {save_dir_raw}")

            save_dir_postprocessed = os.path.join(save_dir, "_scores", "postprocessed")
            sed_scores_eval.io.write_sed_scores(
                self.test_scores_postprocessed_buffer, save_dir_postprocessed
            )
            print(f"\nPostprocessed scores for  saved in: {save_dir_postprocessed}")
        else:
            # * calculate the metrics and save them
            ground_truth = sed_scores_eval.io.read_ground_truth_events(
                self.hparams["data"]["test_tsv"]
            )
            audio_durations = sed_scores_eval.io.read_audio_durations(
                self.hparams["data"]["test_dur"]
            )
            if self.fast_dev_run:
                ground_truth = {
                    audio_id: ground_truth[audio_id]
                    for audio_id in self.test_scores_postprocessed_buffer
                }
                audio_durations = {
                    audio_id: audio_durations[audio_id]
                    for audio_id in self.test_scores_postprocessed_buffer
                }
            else:
                # drop audios without events
                ground_truth = {
                    audio_id: gt for audio_id, gt in ground_truth.items() if len(gt) > 0
                }
                audio_durations = {
                    audio_id: audio_durations[audio_id]
                    for audio_id in ground_truth.keys()
                }
            psds1_psds_eval = compute_psds_from_operating_points(
                self.test_psds_buffer,
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
                dtc_threshold=0.7,
                gtc_threshold=0.7,
                alpha_ct=0,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "", "scenario1"),
            )
            psds1_sed_scores_eval = compute_psds_from_scores(
                self.test_scores_postprocessed_buffer,
                ground_truth,
                audio_durations,
                dtc_threshold=0.7,
                gtc_threshold=0.7,
                cttc_threshold=None,
                alpha_ct=0,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "", "scenario1"),
            )

            psds2_psds_eval = compute_psds_from_operating_points(
                self.test_psds_buffer,
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
                dtc_threshold=0.1,
                gtc_threshold=0.1,
                cttc_threshold=0.3,
                alpha_ct=0.5,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "", "scenario2"),
            )
            psds2_sed_scores_eval = compute_psds_from_scores(
                self.test_scores_postprocessed_buffer,
                ground_truth,
                audio_durations,
                dtc_threshold=0.1,
                gtc_threshold=0.1,
                cttc_threshold=0.3,
                alpha_ct=0.5,
                alpha_st=1,
                save_dir=os.path.join(save_dir, "", "scenario2"),
            )

            event_macro = log_sedeval_metrics(
                self.decoded_05_buffer,
                self.hparams["data"]["test_tsv"],
                os.path.join(save_dir, ""),
            )[0]

            # synth dataset
            intersection_f1_macro = compute_per_intersection_macro_f1(
                {"0.5": self.decoded_05_buffer},
                self.hparams["data"]["test_tsv"],
                self.hparams["data"]["test_dur"],
            )

            results = {
                "test/psds1_psds_eval": psds1_psds_eval,
                "test/psds1_sed_scores_eval": psds1_sed_scores_eval,
                "test/psds2_psds_eval": psds2_psds_eval,
                "test/psds2_sed_scores_eval": psds2_sed_scores_eval,
                "test/event_f1_macro": event_macro,
                "test/intersection_f1_macro": intersection_f1_macro,
            }

        if self.logger is not None:
            self.logger.log_metrics(results)
            self.logger.log_hyperparams(self.hparams, results)

        for key in results.keys():
            self.log(key, results[key], prog_bar=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        return [self.opt], [self.scheduler]

    def train_dataloader(self):
        self.train_loader = torch.utils.data.DataLoader(
            self.train_data,
            batch_sampler=self.train_sampler,
            num_workers=self.num_workers,
        )
        return self.train_loader

    def val_dataloader(self):
        self.val_loader = torch.utils.data.DataLoader(
            self.valid_data,
            batch_size=self.hparams["training"]["batch_size_val"],
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )
        return self.val_loader

    def test_dataloader(self):
        self.test_loader = torch.utils.data.DataLoader(
            self.test_data,
            batch_size=self.hparams["training"]["batch_size_val"],
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )
        return self.test_loader

    def forward(self, x):
        features = self.mel_spec(x)
        features = features.unsqueeze(0)
        preds, _ = self.sed(self.scaler(self.take_log(features)))
        return preds


if __name__ == "__main__":
    with open("params.yaml", "r") as f:
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
    print(sed.state_dict().keys())
