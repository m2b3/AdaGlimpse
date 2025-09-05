import argparse
from abc import ABC
from typing import Any, Optional

from lightning import LightningModule
from torch.optim import AdamW

from AdaGlimpse.architectures.utils import MaeScheduler
from AdaGlimpse.datasets.base import BaseDataModule


class AutoconfigLightningModule(LightningModule):
    internal_data = False
    checkpoint_metric = 'val/loss'
    checkpoint_metric_mode = 'min'


class BaseArchitecture(AutoconfigLightningModule, ABC):
    def __init__(self, datamodule: BaseDataModule = None, lr=1.5e-4, min_lr=1e-8, warmup_epochs=10, weight_decay=0,
                 epochs=100, compile_model=True, **_):
        super().__init__()
        self.lr = lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.compile_model = compile_model

        self.save_hyperparameters(ignore=['datamodule'])

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parser = parent_parser.add_argument_group(BaseArchitecture.__name__)
        parser.add_argument('--lr',
                            help='learning-rate',
                            type=float,
                            default=5e-4)
        parser.add_argument('--warmup-epochs',
                            help='epochs to warmup LR',
                            type=int,
                            default=10)
        parser.add_argument('--min-lr',
                            help='lower lr bound for cyclic schedulers that hit 0',
                            type=float,
                            default=1e-8)
        parser.add_argument('--weight-decay',
                            help='weight_decay',
                            type=float,
                            default=0.01)
        parser.add_argument('--epochs',
                            help='number of epochs',
                            type=int,
                            default=400)
        parser.add_argument('--compile-model',
                            help='use torch compile',
                            type=bool,
                            default=True,
                            action=argparse.BooleanOptionalAction)
        return parent_parser

    def _all_params(self):
        return self.parameters()

    def configure_optimizers(self):
        optimizer = AdamW(self._all_params(), lr=self.min_lr, weight_decay=self.weight_decay, betas=(0.9, 0.95))
        scheduler = MaeScheduler(optimizer=optimizer,
                                 lr=self.lr,
                                 warmup_epochs=self.warmup_epochs,
                                 min_lr=self.min_lr,
                                 epochs=self.epochs)
        scheduler.step(epoch=0)

        lr_schedulers = [
            {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        ]

        return [optimizer], lr_schedulers

    def lr_scheduler_step(self, scheduler, metric: Optional[Any]) -> None:
        scheduler.step(epoch=self.current_epoch, metrics=metric)

    def do_metrics(self, mode, out, batch):
        pass

    def training_step(self, batch, batch_idx):
        out = self.forward(batch)
        loss = out['loss']
        self.log('train/loss', loss, on_step=True, on_epoch=False, sync_dist=False, prog_bar=True)
        self.do_metrics('train', out, batch)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch)
        self.log('val/loss', out['loss'], on_step=False, on_epoch=True, sync_dist=True)
        self.do_metrics('val', out, batch)

    def test_step(self, batch, batch_idx):
        out = self.forward(batch)
        self.log('test/loss', out['loss'], on_step=False, on_epoch=True, sync_dist=True)
        self.do_metrics('test', out, batch)
