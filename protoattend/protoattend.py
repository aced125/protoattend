from protoattend.module import ProtoAttendModule
from protoattend import utils
from typing import Union, Tuple, Dict
import numpy as np
import torch
import torch.utils.data as tud
import pytorch_lightning as pl
from argparse import Namespace
import logging

logger = logging.getLogger(__name__)


class ProtoAttend:
    def __init__(self, encoder, num_classes, model_output_dim: int = None, **hparams):
        self.hparams = utils.load_default_config()
        self.hparams.update(hparams)
        if not isinstance(self.hparams, Namespace):
            self.hparams = utils.wrap_namespace(self.hparams)

        self.hparams.num_classes = num_classes

        self.hparams.d_encoder = model_output_dim
        if not model_output_dim:
            self.hparams.d_encoder = utils.infer_model_output_dim(encoder)

        self.module = ProtoAttendModule(encoder, self.hparams)

    @staticmethod
    def make_dataset(
        X_train: Union[np.ndarray, torch.Tensor, tud.Dataset, tud.DataLoader],
        X_val: Union[np.ndarray, torch.Tensor, tud.Dataset, tud.DataLoader],
        y_train: Union[np.ndarray, torch.Tensor] = None,
        y_val: Union[np.ndarray, torch.Tensor] = None,
    ) -> Tuple[tud.Dataset, tud.Dataset]:
        if isinstance(X_train, tud.Dataset):
            assert isinstance(
                X_val, tud.Dataset
            ), f"If X_train is a tud.Dataset, X_val must be a tud.Dataset as well!"
            return X_train, X_val
        return tud.TensorDataset(X_train, y_train), tud.TensorDataset(X_val, y_val)

    def process_dataloaders(
        self, train_dataloader: tud.DataLoader, val_dataloader: tud.DataLoader
    ) -> Tuple[tud.DataLoader, tud.DataLoader]:
        """
        Given a pair of DataLoaders, it adds `candidate_batch_size` onto the
        current batch size of each DataLoader. Also, if train and val batch
        sizes are different, sets them the same and gives a warning. Finally,
        sets module config to appropriately split batches during training.
        Parameters
        ----------
        train_dataloader: the tud.DataLoader for train set.
        val_dataloader: the tud.DataLoader for val set.

        Returns
        -------
        train_dataloader: modified tud.DataLoader for train set.
        val_dataloader: modified tud.DataLoader for val set.
        """
        train_dataloader_kwargs = {
            k: v for k, v in train_dataloader.__dict__.items() if not k.startswith("_")
        }
        train_dataloader_kwargs.pop("batch_sampler")
        val_dataloader_kwargs = {
            k: v for k, v in val_dataloader.__dict__.items() if not k.startswith("_")
        }
        val_dataloader_kwargs.pop("batch_sampler")

        cand_bs = self.hparams.candidate_batch_size
        self.module.hparams_to_init.input_ratio = utils.compute_input_candidate_ratio(
            cand_bs, train_dataloader.batch_size
        )

        if train_dataloader.batch_size != val_dataloader.batch_size:
            logger.warning(
                f"Setting val_dataloader `batch_size` equal to train_dataloader. This is required for ProtoAttend."
            )
        new_batch_size = train_dataloader.batch_size + cand_bs
        train_dataloader_kwargs.update({"batch_size": new_batch_size})
        val_dataloader_kwargs.update({"batch_size": new_batch_size})

        train_loader = tud.DataLoader(**train_dataloader_kwargs)
        val_loader = tud.DataLoader(**val_dataloader_kwargs)
        return train_loader, val_loader

    def make_dataloaders(
        self,
        X_train: Union[np.ndarray, torch.Tensor, tud.Dataset, tud.DataLoader],
        X_val: Union[np.ndarray, torch.Tensor, tud.Dataset, tud.DataLoader],
        y_train: Union[np.ndarray, torch.Tensor] = None,
        y_val: Union[np.ndarray, torch.Tensor] = None,
    ) -> Tuple[tud.DataLoader, tud.DataLoader]:
        if not isinstance(X_train, tud.DataLoader):
            assert not isinstance(
                X_val, tud.DataLoader
            ), f"If X_train is a dataset, X_val must be a dataset as well!"

            train_dataset, val_dataset = self.make_dataset(
                X_train, X_val, y_train, y_val
            )
            train_dataloader_params = self.hparams["train_dataloader_params"]
            val_dataloader_params = self.hparams["val_dataloader_params"]
            train_dataloader = tud.DataLoader(train_dataset, **train_dataloader_params)
            val_dataloader = tud.DataLoader(val_dataset, **val_dataloader_params)
        else:
            train_dataloader = X_train
            val_dataloader = X_val
        return self.process_dataloaders(train_dataloader, val_dataloader)

    def fit(
        self,
        X_train: Union[np.ndarray, torch.Tensor, tud.Dataset, tud.DataLoader],
        X_val: Union[np.ndarray, torch.Tensor, tud.Dataset, tud.DataLoader],
        y_train: Union[np.ndarray, torch.Tensor] = None,
        y_val: Union[np.ndarray, torch.Tensor] = None,
    ):

        train_dataloader, val_dataloader = self.make_dataloaders(
            X_train, X_val, y_train, y_val
        )
        self.module.train_loader = train_dataloader
        self.module.val_loader = val_dataloader

        trainer_params = self.hparams.trainer_params.__dict__
        trainer = pl.Trainer(**trainer_params)
        trainer.fit(self.module)

    def predict(self, X_test):
        # test_loader = self.make_dataloaders(X_test)
        test_loader = X_test
        out = self.module.infer(test_loader)
        return out
