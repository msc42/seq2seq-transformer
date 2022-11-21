#!/usr/bin/env python3

from __future__ import annotations

import argparse
from contextlib import ExitStack
from enum import Enum

from sklearn.metrics import precision_score, recall_score

# 0 is the label for no correction detected and I is the label for correction detected
label_to_int = {'I':  0, 'O': 1}


class Mode(str, Enum):
    STAT = 'stat',
    CORRECT = 'correct'
    ERROR = 'error'
    EXPORT_CORRECT_CORRECTIONS = 'export_correct_corrections'

    def __str__(self):
        return self.value


def calculate_results(ref_file_path, hyp_file_path, mode, output_file_path):
    total = 0
    errors = 0

    with open(ref_file_path, 'r') as ref_file, \
            open(hyp_file_path, 'r') as hyp_file, \
            ExitStack() as stack:
        if output_file_path:
            output_file = stack.enter_context(open(output_file_path, 'w'))

        ref_list = []
        hyp_list = []

        for i, (ref_line, hyp_line) in enumerate(zip(ref_file, hyp_file)):
            ref_line = ref_line.strip()
            hyp_line = hyp_line.strip()

            source, ref_label = ref_line.split('\t')
            ref = label_to_int[ref_label]
            ref_list.append(ref)

            hyp = int(hyp_line)
            hyp_list.append(hyp)

            if ref != hyp:
                errors += 1

                if mode == Mode.ERROR:
                    print(source)
            else:
                if mode == Mode.CORRECT:
                    print(source)
                elif mode == Mode.EXPORT_CORRECT_CORRECTIONS and ref == label_to_int['I']:
                    output_file.write(f'{i}\n')

            total += 1

    acc = 1 - (errors / total)
    precision_correction = precision_score(ref_list, hyp_list, pos_label=0)
    precision_no_correction = precision_score(ref_list, hyp_list, pos_label=1)
    recall_correction = recall_score(ref_list, hyp_list, pos_label=0)
    recall_no_correction = recall_score(ref_list, hyp_list, pos_label=1)

    f1_correction = (2 * precision_correction * recall_correction / (precision_correction + recall_correction)
                     if precision_correction + recall_correction != 0
                     else 0)
    f1_no_correction = (2 * precision_no_correction * recall_no_correction /
                        (precision_no_correction + recall_no_correction)
                        if precision_no_correction + recall_no_correction != 0
                        else 0)

    return acc, precision_correction, recall_correction, f1_correction, precision_no_correction, recall_no_correction, f1_no_correction


def get_latex_str(results):
    return ' & '.join(f'{result * 100:.2f}\,\%' for result in results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('ref_file', type=str)
    parser.add_argument('hyp_file', type=str)

    parser.add_argument('--mode', type=Mode, default=Mode.STAT, choices=tuple(Mode))
    parser.add_argument('--output_file', type=str, default='',
                        help='used only in combination with export_correct_corrections mode')

    args = parser.parse_args()

    results = calculate_results(args.ref_file, args.hyp_file, args.mode, args.output_file)

    if args.mode in (Mode.STAT, Mode.EXPORT_CORRECT_CORRECTIONS):
        print(results)
        print(get_latex_str(results))
