import torch
import torch.nn as nn
import pytorch_lightning as pl
from sparsemax import Sparsemax
from tqdm import tqdm
import math

from typing import Union, List, Dict, Iterable
import logging

logger = logging.getLogger(__name__)

# Types
LOG_TYPE = Dict[str, torch.Tensor]


class ProtoAttendModule(pl.LightningModule):
    def __init__(self, encoder, hparams):
        super().__init__()
        self._is_cached = False

        self.database_keys = None
        self.database_values = None
        self.database_labels = None

        self.train_loader = None
        self.val_loader = None

        self.model_config = hparams["model_config"]
        self.hparams = hparams

        self.lr = self.hparams.lr

        self.d_model = self.model_config.d_model
        self.d_att = self.model_config.d_att
        self.sqrt_d_att = math.sqrt(self.d_att)
        self.d_val = self.model_config.d_val
        self.d_intermediate = self.model_config.d_intermediate

        self.encoder = encoder
        self.encoder_and_ffn = nn.Sequential(
            encoder,
            nn.Linear(self.d_model, self.d_intermediate),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.d_intermediate),
        )
        self.Q_inp = nn.Sequential(
            nn.Linear(self.d_intermediate, self.d_att), nn.ReLU(inplace=True)
        )
        self.K_cand = nn.Sequential(
            nn.Linear(self.d_intermediate, self.d_att), nn.ReLU(inplace=True)
        )
        self.V = nn.Sequential(
            nn.Linear(self.d_intermediate, self.d_val),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.d_val),
        )

        self.sparsemax = Sparsemax(-1)
        self.softmax = nn.Softmax(-1)

        self.classifier = nn.Linear(self.d_val, self.model_config.num_classes)

    def attention(self, Q, K, sparse=True):
        """Computes attention logits, then pass through sparsemax or softmax"""
        attention_logits = Q.mm(K.t()) / self.sqrt_d_att
        if sparse:
            return self.sparsemax(attention_logits)
        return self.softmax(attention_logits)

    def attention_sparsity_loss(self, attention):
        eps = self.hparams.sparsity_epsilon
        entropy = attention * (attention + eps).log().sum(-1).mean()
        return -entropy

    @staticmethod
    def confidence_loss(attention, input_y, candidate_y, reduce=True):
        indicator = input_y.unsqueeze(-1).eq(candidate_y.unsqueeze(0))  # B_img x B_cand
        confidence = (attention * indicator).sum(-1)
        if reduce:
            return -confidence.mean()
        return -confidence

    def convex_superposition(self, p, q, alpha: Union[List[float], float] = 0.5):
        if isinstance(alpha, list):
            return [self.classifier(a * p + (1 - a) * q) for a in alpha]
        return self.classifier(alpha * p + (1 - alpha) * q)

    def to_fp16(self, tensors: Union[Iterable, torch.Tensor]):
        if not isinstance(tensors, torch.Tensor):
            return [self.to_fp16(x) for x in tensors]

    def forward(self, batch):
        (
            (input_x, input_y),
            (candidate_x, candidate_y),
        ) = self.split_batch_into_input_and_candidate(batch)

        input_x = input_x.to(next(self.encoder.parameters()))
        candidate_x = candidate_x.to(next(self.encoder.parameters()))

        input_enc = self.encoder_and_ffn(input_x)
        cand_enc = self.encoder_and_ffn(candidate_x)

        # Project encodings onto query, key and value spaces
        Q_inp = self.Q_inp(input_enc)
        K_cand = self.K_cand(cand_enc)
        V_inp = self.V(input_enc)
        V_cand = self.V(cand_enc)

        # Compute attention probabilities and weighted prototype embedding
        attention = self.attention(Q_inp, K_cand)
        weighted_prototype_embedding = attention.mm(V_cand)

        # Compute convex superpositions
        self_logits, intermediate_logits, prototype_logits = self.convex_superposition(
            V_inp, weighted_prototype_embedding, alpha=[0.0, 0.5, 1.0]
        )

        output = {"intermediate_logits": intermediate_logits, "labels": input_y}

        if input_y is not None:
            # Compute loss components
            ce_loss = nn.CrossEntropyLoss()

            # Self loss: Normal supervised loss
            self_loss = ce_loss(self_logits, input_y)

            # Intermediate loss: Equal mix of supervised and prototype loss
            intermediate_loss = ce_loss(intermediate_logits, input_y)

            # Prototype loss: Use logits only from weighted sum of prototype candidates
            prototype_loss = ce_loss(prototype_logits, input_y)

            # Compute confidence loss
            confidence_loss = self.confidence_loss(attention, input_y, candidate_y)

            # Compute sparsity loss
            sparsity_loss = self.attention_sparsity_loss(attention)

            # Compute overall loss
            loss = (
                self_loss
                + intermediate_loss
                + prototype_loss
                + self.hparams.sparsity_weight * sparsity_loss
                + self.hparams.confidence_weight * confidence_loss
            )

            output.update(
                {
                    "loss": loss,
                    "self_loss": self_loss,
                    "intermediate_loss": intermediate_loss,
                    "prototype_loss": prototype_loss,
                    "confidence_loss": confidence_loss,
                    "sparsity_loss": sparsity_loss,
                }
            )

        return output

    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        output.pop("intermediate_logits")
        output.pop("labels")

        tensorboard_logs = {"loss": output["loss"]}
        output.update({"log": tensorboard_logs})
        return output

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        predictions = output.pop("intermediate_logits").argmax(-1)
        correct = (predictions == output.pop("labels")).float().mean()

        # Replace all keys in output dictionary; eg "loss" -> "val_loss"
        for old_key, value in output.items():
            new_key = "val_" + old_key
            output[new_key] = output.pop(old_key)
        output.update({"correct": correct})
        return output

    def validation_epoch_end(self, output: List[LOG_TYPE]):
        output: Dict[str, Union[torch.Tensor, LOG_TYPE]] = self.collate(output)
        tensorboard_logs = dict()
        for key, value in output.items():
            tensorboard_logs[key] = value
        output.update({"log": tensorboard_logs})

        return output

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def cache_database_computations(self):
        """Computes keys, values and stores labels for the database."""

        # Initialize temporary state - disable dataloader shuffle, gradients, dropout, batchNorm
        original_shuffle_bool = getattr(self.train_loader, "shuffle")
        setattr(self.train_loader, "shuffle", False)
        torch.set_grad_enabled(False)
        self.eval()

        outputs = []
        for batch in tqdm(self.train_loader):
            x, y = batch
            x = x.to(next(self.encoder.parameters()))
            encoding = self.encoder_and_ffn(x)
            encoded_key = self.K_cand(encoding)
            encoded_value = self.V(encoding)
            output = {"keys": encoded_key, "values": encoded_value, "labels": y}
            outputs.append(output)

        outputs = self.collate(outputs)
        self.database_keys = outputs["keys"]
        self.database_values = outputs["values"]
        self.database_labels = outputs["labels"]

        self._is_cached = True

        # Restore original state
        self.train()
        torch.set_grad_enabled(True)
        setattr(self.train_loader, "shuffle", original_shuffle_bool)

    def infer(self, loader):
        """Runs inference on a tud.DataLoader."""

        if not self._is_cached:
            self.cache_database_computations()

        # Setup initial state
        torch.set_grad_enabled(False)
        self.eval()

        prototype_idxs = []
        prototype_weights = []
        outputs = []
        for batch in tqdm(loader):
            x, *_ = batch
            x = x.to(next(self.encoder.parameters()))

            encoding = self.encoder_and_ffn(x)
            queries = self.Q_inp(encoding)
            values = self.V(encoding)

            attention = self.attention(queries, self.database_keys)
            weighted_prototype_embedding = attention.mm(self.database_values)

            intermediate_logits = self.convex_superposition(
                values, weighted_prototype_embedding, alpha=0.5
            )
            predictions = intermediate_logits.argmax(-1)
            confidence = -self.confidence_loss(
                attention, predictions, self.database_values
            )
            # attn: B_input x B_database (e.g 16 x 30_000)
            attn_weights, attn_indices = attention.topk(5, dim=-1)
            output = {
                "prototype_idxs": attn_indices,
                "prototype_weights": attn_weights,
                "confidence": confidence,
                "prediction": predictions,
            }
            outputs.append(output)

        outputs = self.collate(outputs)

        # Restore back to initial state
        self.train()
        torch.set_grad_enabled(True)

        return outputs

    def split_batch_into_input_and_candidate(self, batch):
        """Splits a batch into input and candidate groups."""
        x, y = batch

        batch_size = x.size(0)
        cutoff = math.ceil(batch_size * self.hparams.input_ratio)
        rand = torch.randperm(batch_size, device=x.device)
        input_idxs = rand[:cutoff]
        cand_idxs = rand[cutoff:]

        input_x = x.index_select(0, input_idxs)
        input_y = y.index_select(0, input_idxs)
        candidate_x = x.index_select(0, cand_idxs)
        candidate_y = y.index_select(0, cand_idxs)

        return (input_x, input_y), (candidate_x, candidate_y)

    def configure_optimizers(self):
        """Configures optimizers and LR schedulers"""
        optimizer_class = getattr(torch.optim, self.hparams.optimizer)
        optimizer = optimizer_class(self.parameters(), **self.hparams.optimizer_params)
        scheduler_class = getattr(torch.optim, self.hparams.scheduler)
        scheduler = scheduler_class(optimizer, **self.hparams.scheduler_params)
        return optimizer, scheduler

    @staticmethod
    def collate(output: List[LOG_TYPE]) -> LOG_TYPE:
        """Collates output from many validation steps"""
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
