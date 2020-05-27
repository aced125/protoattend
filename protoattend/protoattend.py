from protoattend.module import ProtoAttendModule


class ProtoAttend:
    def __init__(self, model, hparams):
        self.hparams = hparams
        self.lightning_module = ProtoAttendModule(model, hparams)

    def make_dataset(
        self,
        X: Union[np.ndarray, torch.Tensor, tud.Dataset],
        y: Union[np.ndarray, torch.Tensor] = None,
    ) -> tud.Dataset:
        return tud.TensorDataset(X, y)

    def make_dataloaders(self, X, y):
        train_dataset, val_dataset = self.make_datasets(X, y)

        train_dataloader_params = self.hparams["train_dataloader_params"]
        train_dataloader_params["batch_size"] += self.hparams["candidate_batch_size"]
        val_dataloader_params = self.hparams["val_dataloader_params"]
        val_dataloader_params["batch_size"] += self.hparams["candidate_batch_size"]

        train_dataloader = tud.DataLoader(train_dataset, **train_dataloader_params)
        val_dataloader = tud.DataLoader(val_dataset, **val_dataloader_params)
        return train_dataloader, val_dataloader

    def fit(self, X, y):
        trainer_params = self.hparams["trainer_params"]
        trainer = pl.Trainer(**trainer_params)
        train_dataloader, val_dataloader = self.make_dataloaders(X, y)
        self.lightning_module.train_loader = train_dataloader
        self.lightning_module.val_loader = val_dataloader
        trainer.fit(self.lightning_module)
