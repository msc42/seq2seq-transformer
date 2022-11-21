#!/usr/bin/env python3

from __future__ import annotations

import argparse

from tqdm import tqdm
from transformers import T5Tokenizer


def main(args: argparse.Namespace) -> None:
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer)

    prefix_len = len(tokenizer.encode(args.prefix)) - 1 if args.prefix else 0

    with open(args.src_path, 'r') as src_file, \
            open(args.tgt_path, 'r') as tgt_file, \
            open(args.copy_tgt_path, 'r') as copy_tgt_file:
        for src, tgt, copy_tgt in tqdm(zip(src_file, tgt_file, copy_tgt_file)):
            src = src.strip()
            tgt = tgt.strip()
            copy_tgt_old = [int(x) for x in copy_tgt.split()]
            copy_tgt = [x + prefix_len for x in copy_tgt_old]

            source_splitted = src.split()
            encoded_src = tokenizer.encode(src)

            words_pos = 0
            current_word_subset_tokens = ''
            current_word_subset_tokens_number = 0
            current_token_pos = 1 + prefix_len

            for token in encoded_src:
                current_word_subset_tokens += tokenizer.decode([token])
                current_word_subset_tokens_number += 1

                if words_pos >= len(source_splitted):
                    src_eos_pos = encoded_src.index(1) + 1 + prefix_len
                    copy_tgt.append(src_eos_pos)
                    break

                if current_word_subset_tokens == source_splitted[words_pos]:
                    if current_word_subset_tokens_number > 1:
                        copy_tgt = adapt_target(current_token_pos, current_word_subset_tokens_number, copy_tgt)
                        current_token_pos += current_word_subset_tokens_number
                    else:
                        current_token_pos += 1

                    words_pos += 1
                    current_word_subset_tokens = ''
                    current_word_subset_tokens_number = 0

            test_it(src, tgt, copy_tgt_old, copy_tgt, tokenizer, args.prefix)

            print(' '.join(str(x) for x in copy_tgt))


def adapt_target(current_token_pos: int, word_token_number: int, copy_target: list[int]) -> list[int]:
    adapted_numbers: list[int] = []
    for number in copy_target:
        if number > current_token_pos:
            adapted_numbers.append(number + word_token_number - 1)
        elif number == current_token_pos:
            adapted_numbers += list(range(number, number + word_token_number))
        else:
            adapted_numbers.append(number)

    return adapted_numbers


def test_it(src: str, tgt: str, copy_tgt_old: list[int], copy_tgt: list[int], tokenizer, prefix=''):
    src_splitted = src.split()
    src_tokens = tokenizer.encode(prefix + src)
    tgt_tokens = tokenizer.encode(tgt)

    try:
        restored_tgt = ' '.join(src_splitted[i - 1] for i in copy_tgt_old)
    except Exception:
        breakpoint()
    if tgt != restored_tgt:
        breakpoint()

    restored_tgt_ids = [src_tokens[i - 1] for i in copy_tgt]
    restored_tgt_from_tokens = tokenizer.decode(restored_tgt_ids)

    if tokenizer.decode(tokenizer.encode(tgt)) != restored_tgt_from_tokens:
        breakpoint()

    if len(copy_tgt) != len(tgt_tokens):
        breakpoint()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('tokenizer', type=str)
    parser.add_argument('src_path', type=str)
    parser.add_argument('tgt_path', type=str)
    parser.add_argument('copy_tgt_path', type=str)
    parser.add_argument('--prefix', type=str, default='')

    args = parser.parse_args()

    main(args)
