#!/usr/bin/env python3

from __future__ import annotations

from contextlib import ExitStack

import torch
from transformers.tokenization_utils import PreTrainedTokenizer


def calculate_acc(outputs: list[dict[str, torch.Tensor]], tokenizer: PreTrainedTokenizer,
                  reference_file_path: str = '', debug_file_path: str='') -> float:
    if reference_file_path:
        with open(reference_file_path, 'r') as gold_file:
            reference_file_content = gold_file.read().splitlines()

    errors = 0
    total = 0

    with ExitStack() as stack:
        if debug_file_path:
            debug_file = stack.enter_context(open(debug_file_path, 'w'))

        for output_batch in outputs:
            reference_batch = reference_file_content[total:
                                                     ] if reference_file_path else output_batch['decoder_input_ids']
            for pred, ref in zip(output_batch['outputs'], reference_batch):
                total += 1
                pred_tokens = tokenizer.decode(pred, skip_special_tokens=True)
                ref_tokens = ref.rstrip() if reference_file_path else tokenizer.decode(ref, skip_special_tokens=True)

                if pred_tokens != ref_tokens:
                    errors += 1
                    if debug_file_path:
                        debug_file.write(f'{pred_tokens} but {ref_tokens}\n')

    return 1 - (errors / total)
