#!/usr/bin/env python3

from __future__ import annotations

import argparse
from contextlib import ExitStack
from enum import Enum
from itertools import repeat
import re

from transformers import T5Tokenizer


class Mode(str, Enum):
    STAT = 'stat',
    CORRECT = 'correct'
    ERROR = 'error'
    CLASSIFICATION = 'classification'

    def __str__(self):
        return self.value


def calculate_results(ref_file_path, hyp_file_path, mode, add_number_classification_false_negatives,
                      add_number_classification_true_negatives, source_file_path='',
                      case_insensitiv=False, remove_from_source_line='', add_to_source_line='', ignore_spaces=False):
    total = add_number_classification_false_negatives + add_number_classification_true_negatives
    errors = add_number_classification_false_negatives
    errors_correction = add_number_classification_false_negatives
    errors_extraction = add_number_classification_false_negatives

    tokenizer = T5Tokenizer.from_pretrained('t5-large')

    with open(ref_file_path, 'r') as ref_file, \
            open(hyp_file_path, 'r') as hyp_file, \
            ExitStack() as stack:
        source_file = stack.enter_context(open(source_file_path, 'r')) if source_file_path else repeat(None)

        for ref_line, hyp_line, src_line in zip(ref_file, hyp_file, source_file):
            error = False

            total += 1

            ref_line = ref_line.strip()
            hyp_line = hyp_line.strip()

            if src_line is not None:
                src_line = src_line.strip()

            if case_insensitiv:
                ref_line = ref_line.lower()
                hyp_line = hyp_line.lower()
                if src_line is not None:
                    src_line = src_line.lower()

            ref_line = tokenizer.decode(tokenizer.encode(ref_line)).removesuffix('</s>')

            if mode == Mode.CLASSIFICATION:
                adapted_src_line = tokenizer.decode(tokenizer.encode(src_line), skip_special_tokens=True)
                adapted_src_line = adapted_src_line.removesuffix(remove_from_source_line) + add_to_source_line
                print(1 if adapted_src_line == hyp_line else 0)

            ref_splitted = ref_line.split('|')
            hyp_splitted = hyp_line.split('|')

            ref_correction = ref_splitted[0].strip()
            ref_extraction = ref_splitted[1].strip() if len(ref_splitted) > 1 else ''

            hyp_correction = hyp_splitted[0].strip()
            hyp_extraction = hyp_splitted[1].strip() if len(hyp_splitted) > 1 else ''

            if ignore_spaces:
                ref_correction = ref_correction.replace(' ', '')
                hyp_correction = hyp_correction.replace(' ', '')

            if ref_correction != hyp_correction:
                errors_correction += 1
                error = True

            ref_extraction_splitted = ref_extraction.split(' - ')
            ref_extraction_1 = ref_extraction_splitted[0].strip().replace(' ', '')
            ref_extraction_2 = ref_extraction_splitted[1].strip().replace(
                ' ', '') if len(ref_extraction_splitted) > 1 else ''

            hyp_extraction_splitted = hyp_extraction.split(' - ')
            hyp_extraction_1 = hyp_extraction_splitted[0].strip().replace(' ', '')
            hyp_extraction_2 = hyp_extraction_splitted[1].strip().replace(
                ' ', '') if len(hyp_extraction_splitted) > 1 else ''

            if ignore_spaces:
                ref_extraction_1 = ref_extraction_1.replace(' ', '')
                ref_extraction_2 = ref_extraction_2.replace(' ', '')
                hyp_extraction_1 = hyp_extraction_1.replace(' ', '')
                hyp_extraction_2 = hyp_extraction_2.replace(' ', '')

            if re.fullmatch('(.*) -> \\1', hyp_extraction_1):
                hyp_extraction_1 = ''

            if re.fullmatch('(.*) -> \\1', hyp_extraction_2):
                hyp_extraction_2 = ''

            error_in_extraction_order_1 = ref_extraction_1 != hyp_extraction_1 or ref_extraction_2 != hyp_extraction_2
            error_in_extraction_order_2 = ref_extraction_1 != hyp_extraction_2 or ref_extraction_2 != hyp_extraction_1

            if error_in_extraction_order_1 and error_in_extraction_order_2:
                errors_extraction += 1
                error = True

            if error:
                errors += 1

            if mode == Mode.ERROR and error:
                print(src_line)
                print(ref_line)
                print(hyp_line)
                print()
            elif mode == Mode.CORRECT and not error:
                print(f'{hyp_line}')

    return 1 - (errors / total), 1 - (errors_correction / total), 1 - (errors_extraction / total)


def get_latex_str(results):
    return f'{results[1] * 100:.2f}\,\% & {results[2] * 100:.2f}\,\% & {results[0] * 100:.2f}\,\%'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('ref_file', type=str)
    parser.add_argument('hyp_file', type=str)

    parser.add_argument('--mode', type=Mode, default=Mode.STAT, choices=tuple(Mode))
    parser.add_argument('--source_file', type=str, default='')
    parser.add_argument('--add_number_classification_false_negatives',
                        type=int, default=0, help='for pipeline approach')
    parser.add_argument('--add_number_classification_true_negatives',
                        type=int, default=0, help='for pipeline approach')
    parser.add_argument('--case_insensitive', action='store_true')
    parser.add_argument('--ignore_spaces', action='store_true')
    parser.add_argument('--remove_from_source_line', type=str, default='',
                        help='only for classification mode, remove from end of source line, execute before add to source line')
    parser.add_argument('--add_to_source_line', type=str, default='',
                        help='only for classification mode, add to end of source line, executed after remove from source line')

    args = parser.parse_args()

    results = calculate_results(args.ref_file, args.hyp_file, args.mode,
                                args.add_number_classification_false_negatives,
                                args.add_number_classification_true_negatives,
                                source_file_path=args.source_file,
                                case_insensitiv=args.case_insensitive,
                                remove_from_source_line=args.remove_from_source_line,
                                add_to_source_line=args.add_to_source_line, ignore_spaces=args.ignore_spaces)

    if args.mode == Mode.STAT:
        print(results)
        print(get_latex_str(results))
