import torch
import torch.nn.functional as F
import torch.utils.data as tud
import torchvision
import torchvision.transforms as T
import PIL
import pytorch_lightning as pl
from sparsemax import Sparsemax
import torch.nn as nn
from tqdm.notebook import tqdm
from argparse import Namespace
import copy
# from adabound import AdaBound
import math

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


class ProtoAttendModule(pl.LightningModule):
    def __init__(self, encoder, model_config, hparams):
        super().__init__()
        self._is_cached = False

        self.model_config = model_config
        self.hparams = hparams

        self.lr = self.hparams.lr

        self.d_att = model_config.d_att
        self.sqrt_d_att = math.sqrt(self.d_att)
        self.d_out = model_config.d_out
        self.d_intermediate = model_config.d_intermediate

        self.encoder = encoder
        self.encoder_and_ffn = nn.Sequential(
            *[
                encoder,
                nn.Linear(1000, self.d_intermediate),
                nn.ReLU(inplace=True),
                nn.LayerNorm(self.d_intermediate),
            ]
        )
        self.Q_inp = nn.Linear(self.d_intermediate, self.d_att)
        self.V_inp = nn.Linear(self.d_intermediate, self.d_out)

        self.K_cand = nn.Linear(self.d_intermediate, self.d_att)
        self.V_cand = nn.Linear(self.d_intermediate, self.d_out)

        self.sparsemax = Sparsemax(-1)

        self.classifier = nn.Linear(self.d_out, model_config.num_classes)

    def forward(self, x):
        return self.encoder(x)

    def compute_relational_attention(self, x, candidate_x):
        K_cand = self.K_cand(candidate_x)
        Q = self.Q_inp(x)
        attention_logits = Q.mm(K_cand.t()) / self.sqrt_d_att
        attention_probs = self.sparsemax(attention_logits)
        return attention_probs

    def compute_decision_logits(self, V, prototype_logits, alpha=0.5):
        logits = (1 - alpha) * V + alpha * prototype_logits
        return self.classifier(logits)

    def compute_prototype_logits(self, V_cand, attention_probs):
        return attention_probs.mm(V_cand)

    def compute_loss(self, V, prototype_logits, labels, alpha=0.5, return_logits=False):
        logits = self.compute_decision_logits(V, prototype_logits, alpha)
        loss = nn.CrossEntropyLoss()(logits, labels)
        if return_logits:
            return loss, logits
        return loss

    def compute_attention_sparsity_loss(self, attention_probs):
        entropy = (
            attention_probs * (attention_probs + self.hparams.sparsity_epsilon).log()
        )
        sparsity_loss = entropy.sum(-1).mean()
        return -sparsity_loss

    def compute_confidence_loss(self, attention_probs, img_labels, candidate_labels):
        indicator = img_labels.unsqueeze(-1).eq(
            candidate_labels.unsqueeze(0)
        )  # B_img x B_cand
        loss = -attention_probs * indicator
        return loss.sum(-1).mean()  # sum across prototypes, mean across batch

    def forward_training(self, batch):
        (
            (input_img, labels),
            (candidate_img, candidate_labels),
        ) = self.split_batch_into_input_and_candidate(batch)

        input_img = input_img.to(next(self.encoder.parameters()))
        candidate_img = candidate_img.to(next(self.encoder.parameters()))

        x_enc = self.encoder(input_img)
        cand_enc = self.encoder(candidate_img)

        # Compute attention probabilities
        attention_probs = self.compute_relational_attention(x_enc, cand_enc)

        # Project encodings to value encoding
        V = self.V_inp(x_enc)
        V_cand = self.V_cand(cand_enc)

        # Compute prototype logits
        prototype_logits = self.compute_prototype_logits(V_cand, attention_probs)

        # Compute supervised loss components
        self_loss = self.compute_loss(V, prototype_logits, labels, alpha=0)
        intermediate_loss, intermediate_logits = self.compute_loss(
            V, prototype_logits, labels, alpha=0.5, return_logits=True
        )
        prototype_loss = self.compute_loss(V, prototype_logits, labels, alpha=1)

        # Compute confidence loss
        confidence_loss = self.compute_confidence_loss(
            attention_probs, labels, candidate_labels
        )
        # Compute sparsity loss
        sparsity_loss = self.compute_attention_sparsity_loss(attention_probs)

        loss = (
            self_loss
            + intermediate_loss
            + prototype_loss
            + self.hparams.sparsity_weight * sparsity_loss
            + self.hparams.confidence_weight * confidence_loss
        )

        return {
            "loss": loss,
            "self_loss": self_loss,
            "intermediate_loss": intermediate_loss,
            "prototype_loss": prototype_loss,
            "confidence_loss": confidence_loss,
            "sparsity_loss": sparsity_loss,
            "intermediate_logits": intermediate_logits,
            "labels": labels,
        }

        # return (
        #     loss,
        #     self_loss,
        #     intermediate_loss,
        #     prototype_loss,
        #     confidence_loss,
        #     sparsity_loss,
        #     intermediate_logits,
        #     labels,
        # )

    def training_step(self, batch, batch_idx):
        output = self.forward_training(batch)
        output.pop("intermediate_logits")
        output.pop("labels")

        tensorboard_logs = {"loss": output["loss"]}
        output.update({"log": tensorboard_logs})
        return output

    def validation_step(self, batch, batch_idx):
        (
            loss,
            self_loss,
            intermediate_loss,
            prototype_loss,
            confidence_loss,
            sparsity_loss,
            intermediate_logits,
            labels,
        ) = self.forward_training(batch)

        predictions = intermediate_logits.argmax(-1)
        correct = (predictions == labels).float().mean()
        output = {
            "val_loss": loss,
            "val_self_loss": self_loss,
            "val_intermediate_loss": intermediate_loss,
            "val_prototype_loss": prototype_loss,
            "val_confidence_loss": confidence_loss,
            "val_sparsity_loss": sparsity_loss,
            "correct": correct,
        }
        return output

    def validation_epoch_end(self, output):
        output = self.collate(output)
        output.update(
            {
                "log": {
                    "val_loss": output["val_loss"],
                    "val_self_loss": output["val_self_loss"],
                    "val_intermediate_loss": output["val_intermediate_loss"],
                    "val_prototype_loss": output["val_prototype_loss"],
                    "val_confidence_loss": output["val_confidence_loss"],
                    "val_sparsity_loss": output["val_sparsity_loss"],
                    "correct": output["correct"],
                }
            }
        )

        return output

    def train_dataloader(self):
        return tud.DataLoader(
            train_dataset,
            batch_size=self.hparams.train_bs,
            shuffle=self.hparams.shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self):
        return tud.DataLoader(
            val_dataset,
            batch_size=self.hparams.val_bs,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def cache_train_img_encodings(self):
        train_loader = tud.DataLoader(
            train_dataset,
            batch_size=256,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

        encodings = []
        self.eval()
        with torch.no_grad():
            for batch in tqdm(train_loader):
                img, label = batch
                img = img.to(next(self.encoder.parameters()))
                encoding = self.encoder(img)
                encodings.append(encoding)

        encodings = torch.cat(encodings, 0)
        self.database_encodings = encodings
        self._is_cached = True
        self.train()

    def infer(self, loader):
        if not self._is_cached:
            self.cache_train_img_encodings()

        self.eval()
        prototype_idxs = []
        prototype_weights = []
        with torch.no_grad():
            for batch in tqdm(loader):
                img, *_ = batch
                img = img.to(next(self.encoder.parameters()))
                encoding = self.encoder(img)
                attention = self.compute_relational_attention(
                    encoding, self.database_encodings
                )
                # attn: B_input x B_database (e.g 16 x 30_000)
                values, indices = attention.topk(5, dim=-1)
                prototype_idxs.append(indices)
                prototype_weights.append(values)

        prototype_idxs = torch.cat(prototype_idxs, 0)
        prototype_weights = torch.cat(prototype_weights, 0)
        self.train()

        return prototype_idxs, prototype_weights

    def split_batch_into_input_and_candidate(self, batch):
        img, label = batch

        batch_size = img.size(0)
        cutoff = math.ceil(batch_size * self.hparams.input_ratio)
        rand = torch.randperm(batch_size, device=img.device)
        input_idxs = rand[:cutoff]
        cand_idxs = rand[cutoff:]

        input_img = img.index_select(0, input_idxs)
        input_label = label.index_select(0, input_idxs)
        candidate_img = img.index_select(0, cand_idxs)
        candidate_label = label.index_select(0, cand_idxs)

        return (input_img, input_label), (candidate_img, candidate_label)

    def collate(self, output):
        collated_output = {}
        example = output[0]
        for key in example:
            tensor_dim = example[key].dim()
            if tensor_dim == 0:
                collated_output[key] = torch.stack([x[key] for x in output]).mean()
            elif tensor_dim == 1:
                collated_output[key] = torch.cat([x[key] for x in output])
            else:
                raise ValueError(f"Unexpected tensor dimension: {tensor_dim}")
        return collated_output

    # def configure_optimizers(self):
    #     return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def configure_optimizers(self):
        optimizers = [AdaBound(self.parameters(), lr=self.lr)]
        schedulers = [
            # {
            #     'scheduler': torch.optim.lr_scheduler.OneCycleLR(
            #         optimizers[0],
            #         self.lr * 10,
            #         total_steps=10_000,
            #         epochs=None,
            #         steps_per_epoch=None,
            #         pct_start=0.3,
            #         anneal_strategy='cos',
            #         cycle_momentum=True,
            #         base_momentum=0.85,
            #         max_momentum=0.95,
            #         div_factor=25.0,
            #         final_div_factor=10000.0,
            #         last_epoch=-1
            #     ),
            #     # 'monitor': 'val_recall', # Default: val_loss
            #     'interval': 'step',
            #     # 'frequency': 1
            # },
            {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizers[0],
                    mode="min",
                    factor=0.7,
                    patience=3,
                    verbose=False,
                    threshold=0.0001,
                    threshold_mode="rel",
                    cooldown=0,
                    min_lr=0,
                    eps=1e-08,
                ),
                "monitor": "val_loss",  # Default: val_loss
                "interval": "epoch",
                "frequency": 1,
            },
        ]
        return optimizers, schedulers
