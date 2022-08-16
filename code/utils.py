# based on: https://raw.githubusercontent.com/huggingface/transformers/v3.1.0/examples/seq2seq/utils.py

from __future__ import annotations

import linecache
from pathlib import Path

import torch
from torch.utils.data import Dataset


def encode_line(tokenizer, line, max_length, pad_to_max_length=True, return_tensors="pt"):
    return tokenizer(
        [line],
        max_length=max_length,
        padding="max_length" if pad_to_max_length else None,
        truncation=True,
        return_tensors=return_tensors,
    )


def trim_batch(
    input_ids, pad_token_id, attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


class Seq2SeqDataset(Dataset):

    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        type_path="train",
        n_obs=None,
        src_lang=None,
        tgt_lang=None,
        prefix="",
        copy_target=False
    ):
        super().__init__()

        self.src_file = Path(data_dir).joinpath(type_path + ".source")
        self.tgt_file = Path(data_dir).joinpath(type_path + ".target")

        self.copy_tgt_file = Path(data_dir).joinpath(type_path + ".copy_target") if copy_target else None

        self.src_lens = self.get_char_lens(self.src_file)
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        self.tokenizer = tokenizer
        self.prefix = prefix
        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.pad_token_id = self.tokenizer.pad_token_id
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __len__(self):
        return len(self.src_lens)

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")

        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        source_inputs = encode_line(self.tokenizer, source_line, self.max_source_length)
        target_inputs = encode_line(self.tokenizer, tgt_line, self.max_target_length)

        source_ids = source_inputs["input_ids"].squeeze()
        target_ids = target_inputs["input_ids"].squeeze()
        src_mask = source_inputs["attention_mask"].squeeze()
        item = {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "decoder_input_ids": target_ids,
        }

        if self.copy_tgt_file:
            copy_tgt_line = linecache.getline(str(self.copy_tgt_file), index).rstrip("\n")
            assert copy_tgt_line, f"empty target 2 line for index {index}"
            copy_tgt_pos_list = [int(token) - 1 for token in copy_tgt_line.split()]
            copy_tgt_pos = torch.Tensor(copy_tgt_pos_list + [-100] * (self.max_source_length - len(copy_tgt_pos_list)))
            copy_tgt_pos = copy_tgt_pos.type_as(source_ids)
            item["copy_tgt_pos"] = copy_tgt_pos

        return item

    @staticmethod
    def get_char_lens(data_file_path):
        with Path(data_file_path).open() as data_file:
            return [len(x) for x in data_file.readlines()]

    def collate_fn(self, batch) -> dict[str, torch.Tensor]:
        input_ids = torch.stack([x["input_ids"] for x in batch])
        masks = torch.stack([x["attention_mask"] for x in batch])
        target_ids = torch.stack([x["decoder_input_ids"] for x in batch])
        pad_token_id = self.pad_token_id
        y = trim_batch(target_ids, pad_token_id)
        source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)

        if self.copy_tgt_file:
            copy_tgt_pos = trim_batch(torch.stack([x["copy_tgt_pos"] for x in batch]), -100)

        batch = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "decoder_input_ids": y,
        }

        if self.copy_tgt_file:
            batch["copy_tgt_pos"] = copy_tgt_pos

        return batch
