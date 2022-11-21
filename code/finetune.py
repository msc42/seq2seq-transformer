#!/usr/bin/env python3

from __future__ import annotations

import argparse
from enum import Enum
import json
import os

import git
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput

from evaluator import calculate_acc
from modeling_t5 import T5ForConditionalCopying
from utils import Seq2SeqDataset


class Mode(str, Enum):
    GENERATE = 'generate'
    COPY = 'copy'

    def __str__(self):
        return self.value


class DatasetType(str, Enum):
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'

    def __str__(self):
        return self.value


class T5Transformer(LightningModule):

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()

        self.args = args
        self.mode = args.mode

        if self.mode == Mode.COPY and args.beam_size != 1:
            raise NotImplementedError('copy mode is only supported with beam size of 1')

        self.hyps_file = None

        self.best_tfmr_path = os.path.join(args.save_path, 'best_tfmr')

        if (not args.overwrite and args.do_train and os.path.exists(self.best_tfmr_path) and
                len(os.listdir(self.best_tfmr_path)) > 0):
            raise FileExistsError(f'{self.best_tfmr_path} contains data, aborting to prevent data loss')

        self.write_hparams_to_file()

        self.tokenizer = (T5Tokenizer.from_pretrained(args.tokenizer)
                          if args.tokenizer
                          else T5Tokenizer.from_pretrained(args.model))

        model_config = {
            'prefix': args.prefix,
            'max_length': args.max_length,
            'num_beams': args.beam_size,
        }

        self.model = (T5ForConditionalGeneration.from_pretrained(args.model, **model_config)
                      if self.mode == 'generate'
                      else T5ForConditionalCopying.from_pretrained(args.model, mode=self.mode, **model_config))

        self.model.is_test = False

        modules_to_freeze = [self.model.shared] if args.freeze_embedding else []
        modules_to_freeze.extend([self.model.encoder.block[i] for i in range(args.freeze_encoders)])
        modules_to_freeze.extend([self.model.decoder.block[i] for i in range(args.freeze_decoders)])

        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False

    def write_hparams_to_file(self):
        repo = git.Repo(__file__, search_parent_directories=True)
        repo_infos = {
            "repo_sha": repo.head.object.hexsha,
            "repo_branch": str(repo.active_branch),
        }

        with open(os.path.join(self.args.save_path, 'hparams.json'), 'w') as json_file:
            json.dump({**args.__dict__, **repo_infos}, json_file, indent=4)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> Seq2SeqLMOutput:
        output = self.model(input_ids, **kwargs)
        return output

    def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model.generate(input_ids, **kwargs)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        if self.mode == 'generate':
            outputs = self.forward(input_ids=batch['input_ids'], labels=batch['decoder_input_ids'],
                                   attention_mask=batch['attention_mask'], return_dict=True)
        else:
            outputs = self.forward(input_ids=batch['input_ids'], labels=batch['decoder_input_ids'],
                                   copy_target=batch['copy_tgt_pos'],
                                   attention_mask=batch['attention_mask'], return_dict=True)

        return outputs.loss

    def validation_step(self, batch: dict[str, torch.Tensor],
                        batch_idx: int, dataset_type: str = DatasetType.VALID) -> dict[str, torch.Tensor]:
        if self.mode == 'generate':
            outputs = self.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        elif self.mode == Mode.COPY:
            outputs = self.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        else:
            raise NotImplementedError

        validation_step_output = {'outputs': outputs, 'decoder_input_ids': batch['decoder_input_ids']}

        if self.mode == Mode.COPY:
            validation_step_output['copy_tgt_pos'] = batch['copy_tgt_pos']

        return validation_step_output

    def validation_epoch_end(self, outputs: list[dict[str, torch.Tensor]], dataset_type: str = DatasetType.VALID) -> None:
        correction_acc = calculate_acc(outputs, self.tokenizer, reference_file_path=os.path.join(
            self.args.data_path, f'{dataset_type}.target') if self.mode == Mode.COPY else '',
            debug_file_path=os.path.join(args.save_path, 'debug.txt'))

        if self.hyps_file:
            for output in outputs:
                for pred in output['outputs']:
                    pred_tokens = self.tokenizer.decode(pred, skip_special_tokens=True)
                    self.hyps_file.write(pred_tokens + '\n')

        self.log(f'{dataset_type}_correction_acc', correction_acc)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, torch.Tensor]:
        return self.validation_step(batch, batch_idx, dataset_type=DatasetType.TEST)

    def test_epoch_end(self, outputs: list[dict[str, torch.Tensor]]) -> None:
        self.validation_epoch_end(outputs, dataset_type=DatasetType.TEST)

        if self.hyps_file:
            self.hyps_file.close()
            self.hyps_file = None

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), weight_decay=self.args.weight_decay,
                                         lr=self.args.learning_rate)
        elif self.args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), weight_decay=self.args.weight_decay,
                                          lr=self.args.learning_rate)
        elif self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), weight_decay=self.args.weight_decay,
                                        lr=self.args.learning_rate)
        else:
            raise NotImplementedError

        return optimizer

    def on_save_checkpoint(self, _) -> None:
        self.model.save_pretrained(self.best_tfmr_path)
        self.tokenizer.save_pretrained(self.best_tfmr_path)

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--beam_size', type=int, default=4)
        parser.add_argument('--freeze_embedding', action='store_true')
        parser.add_argument('--freeze_encoders', type=int, default=0, metavar='N',
                            help='freeze the first N encoders')
        parser.add_argument('--freeze_decoders', type=int, default=0, metavar='N',
                            help='freeze the first N decoders')
        parser.add_argument('--learning_rate', type=float, default=0.00025)
        parser.add_argument('--max_length', type=int, default=128)
        parser.add_argument('--mode', type=Mode, choices=tuple(Mode), default=Mode.GENERATE)
        parser.add_argument('--model', type=str, default='t5-small')
        parser.add_argument('--optimizer', type=str, default='adam')
        parser.add_argument('--prefix', type=str, default='custom: ')
        parser.add_argument('--shuffle_data', type=bool, default=True)
        parser.add_argument('--tokenizer', type=str, default='t5-small')
        parser.add_argument('--weight_decay', type=float, default=0.0)

        return parser

    @staticmethod
    def add_argparse_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--prefix', type=str, default='custom: ')
        return parser


class PredictionCallback(Callback):

    def __init__(self, hyps_file_path: str) -> None:
        super().__init__()
        self.hyps_file_path = hyps_file_path

    def on_validation_start(self, trainer: Trainer, model: LightningModule) -> None:
        model.model.is_test = True

    def on_validation_end(self, trainer: Trainer, model: LightningModule) -> None:
        model.model.is_test = False

    def on_test_start(self, trainer: Trainer, model: LightningModule) -> None:
        if self.hyps_file_path:
            model.hyps_file = open(self.hyps_file_path, 'w')

        model.model.is_test = True

    def on_test_end(self, trainer: Trainer, model: LightningModule) -> None:
        model.model.is_test = False


class Seq2SeqData(LightningDataModule):

    def __init__(self, data_dir: str, mode: str, tokenizer: PreTrainedTokenizer, max_source_length: int,
                 max_target_length: int, batch_size: int, prefix: str) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.batch_size = batch_size
        self.prefix = prefix

    def get_dataloader(self, type_path: str, shuffle: bool=False) -> DataLoader:
        dataset = Seq2SeqDataset(
            self.tokenizer,
            data_dir=self.data_dir,
            max_source_length=self.max_source_length,
            max_target_length=self.max_target_length,
            type_path=type_path,
            n_obs=None,
            src_lang=None,
            tgt_lang=None,
            prefix=self.prefix,
            copy_target=True if self.mode == Mode.COPY else False
        )

        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
            num_workers=1,
            sampler=None,
        )
        return train_loader

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(DatasetType.TRAIN, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader(DatasetType.VALID)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader(DatasetType.TEST)


def main(args: argparse.Namespace) -> None:
    model = T5Transformer(args)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_path,
        filename='{epoch}-{val_correction_acc}',
        monitor='valid_correction_acc',
        mode='max',
        save_top_k=2,
    )

    callbacks = [checkpoint_callback, PredictionCallback(args.predictions_file)]

    data_model = Seq2SeqData(model.args.data_path, model.mode, model.tokenizer, args.max_length, args.max_length,
                             model.args.batch_size, model.args.prefix)

    trainer = Trainer.from_argparse_args(args, callbacks=callbacks)

    if args.max_epochs > 0:
        trainer.fit(model, data_model, ckpt_path=args.checkpoint_path)

    if args.predictions_file:
        trainer.test(model=model if args.max_epochs == 0 else None,
                     datamodule=data_model, ckpt_path=args.checkpoint_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('save_path', type=str,
                        help='path where the output is saved')
    parser.add_argument('data_path', type=str,
                        help='path where training and evaluation data is saved')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='path where checkpoint is saved')
    parser.add_argument('--do_train', action='store_true',
                        help='do training')
    parser.add_argument('--predictions_file', type=str, default='',
                        help='file where the predictions are written to, if no file is given, no predictions are done')
    parser.add_argument('--overwrite', action='store_true',
                        help='overwrite trained models')

    parser = T5Transformer.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    main(args)
