import pytorch_lightning as pl
from torch.utils.data import DataLoader


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int,
                 num_workers: int):
        super().__init__()
        self.save_hyperparameters(logger=False)

        from framework.data.tools.collate import collate_motion_and_audio, collate_motion
        if hasattr(self.hparams, 'load_audio') and self.hparams.load_audio is True:
            self.dataloader_options = {"batch_size": batch_size, "num_workers": num_workers,
                                       "collate_fn": collate_motion_and_audio}
        else:
            self.dataloader_options = {"batch_size": batch_size, "num_workers": num_workers,
                                       "collate_fn": collate_motion}

    # create a dataset for class setting, contains a tiny subset of train data
    def get_sample_set(self, overrides={}):
        sample_params = self.hparams.copy()
        sample_params.update(overrides)
        return self.Dataset(**sample_params)

    # Lazy loading. The datasets are only created when first accessed, and cached for future use.
    def __getattr__(self, item):
        if item.endswith("_dataset") and not item.startswith("_"):
            subset = item[:-len("_dataset")]
            item_c = "_" + item
            if item_c not in self.__dict__:
                self.__dict__[item_c] = self.Dataset(split=subset, **self.hparams)
            return getattr(self, item_c)
        classname = self.__class__.__name__
        raise AttributeError(f"'{classname}' object has no attribute '{item}'")

    def setup(self, stage=None):
        # Use the getter the first time to load the data
        if stage in (None, "fit"):
            _ = self.train_dataset
            _ = self.val_dataset
        if stage in (None, "test"):
            _ = self.test_dataset

    # Turn off persistent_workers otherwise it will cause memory leak
    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, # persistent_workers=True,
                          **self.dataloader_options)

    def predict_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=False, # persistent_workers=True,
                          **self.dataloader_options)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, # persistent_workers=True,
                          **self.dataloader_options)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, # persistent_workers=True,
                          **self.dataloader_options)
