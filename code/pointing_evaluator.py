#!/usr/bin/env python3

from __future__ import annotations

import argparse
from contextlib import ExitStack
from itertools import repeat
import re


regexes = ('I cannot see the (.*)\. Please point exactly to it\.',
           'I cannot see anything\. Please point exactly to the object you refer to\.',
           'I cannot see the (.*), but I see (?:a|an) (.*)\. Please point exactly to the (.*)\.',
           'I only see (.*)\. Please point exactly to the (.*)\.',
           'I see multiple (.*)\. Which (.*) do you mean\?',
           'Which (.*) do you mean\?',
           'Please point exactly to the object you refer to or describe it\.')


def calculate_results(ref_file_path: str, hyp_file_path: str, mode: str = 'stat',
                      source: str = '') -> list[dict[str, dict[str, float]]]:
    total = 0
    total_with_1_entities = 0
    total_with_2_entities = 0

    total_wo_this = 0
    total_with_1_entities_wo_this = 0
    total_with_2_entities_wo_this = 0

    total_w_this = 0
    total_with_1_entities_w_this = 0
    total_with_2_entities_w_this = 0

    total_wo_regex = 0
    total_with_1_entities_wo_regex = 0
    total_with_2_entities_wo_regex = 0

    total_w_regex = 0
    total_with_1_entities_w_regex = 0
    total_with_2_entities_w_regex = 0

    action_correct = 0
    first_entity_correct = 0
    second_entity_correct = 0
    action_and_entities_correct = 0

    action_correct_wo_this = 0
    first_entity_correct_wo_this = 0
    second_entity_correct_wo_this = 0
    action_and_entities_correct_wo_this = 0

    action_correct_w_this = 0
    first_entity_correct_w_this = 0
    second_entity_correct_w_this = 0
    action_and_entities_correct_w_this = 0

    action_correct_wo_regex = 0
    first_entity_correct_wo_regex = 0
    second_entity_correct_wo_regex = 0
    action_and_entities_correct_wo_regex = 0

    action_correct_w_regex = 0
    first_entity_correct_w_regex = 0
    second_entity_correct_w_regex = 0
    action_and_entities_correct_w_regex = 0

    regex_occurrences = len(regexes) * [0]

    with open(ref_file_path, 'r') as ref_file, \
            open(hyp_file_path, 'r') as hyp_file, \
            ExitStack() as stack:

        if source:
            source_file = stack.enter_context(open(source, 'r'))

        for ref, hyp, src in zip(ref_file, hyp_file, source_file if source else repeat(None)):
            ref = ref.strip()
            hyp = hyp.strip()
            if source:
                src = src.strip()

            current_action_correct = False
            current_first_entity_correct = False
            current_second_entity_correct = False

            total += 1

            ref_response, ref_entities_1, ref_entities_2 = check_error_responses(ref)

            if ref_response != -1:
                total_w_regex += 1
                regex_occurrences[ref_response] += 1
                hyp_response, hyp_entities_1, hyp_entities_2 = check_error_responses(hyp)
                ref_action = f'regex_{ref_response}'
                hyp_action = f'regex_{hyp_response}'
            else:
                total_wo_regex += 1
                ref_splitted = ref.split(' ')
                hyp_splitted = hyp.split(' ')

                ref_action = ref_splitted[0]
                hyp_action = hyp_splitted[0]

            w_this = '#' in ref or ref_response != -1
            if w_this:
                total_w_this += 1
            else:
                total_wo_this += 1

            if ref == hyp:
                action_and_entities_correct += 1

                if w_this:
                    action_and_entities_correct_w_this += 1
                else:
                    action_and_entities_correct_wo_this += 1

                if ref_action.startswith('regex'):
                    action_and_entities_correct_w_regex += 1
                else:
                    action_and_entities_correct_wo_regex += 1

            if ref_action == hyp_action:
                action_correct += 1
                current_action_correct = True

                if w_this:
                    action_correct_w_this += 1
                else:
                    action_correct_wo_this += 1

                if ref_action.startswith('regex'):
                    action_correct_w_regex += 1
                else:
                    action_correct_wo_regex += 1

            if not ref_action.startswith('regex'):
                ref_wo_action = ' '.join(ref_splitted[1:])
                hyp_wo_action = ' '.join(hyp_splitted[1:])

                ref_entities = ref_wo_action.split(', ')
                hyp_entities = hyp_wo_action.split(', ')

                ref_entities_1 = [ref_entities[0]] if len(ref_entities) > 0 else None
                hyp_entities_1 = [hyp_entities[0]] if len(hyp_entities) > 0 else None
                ref_entities_2 = [ref_entities[1]] if len(ref_entities) > 1 else None
                hyp_entities_2 = [hyp_entities[1]] if len(hyp_entities) > 1 else None

            if ref_entities_1 is not None or hyp_entities_1 is not None:
                total_with_1_entities += 1

                if w_this:
                    total_with_1_entities_w_this += 1
                else:
                    total_with_1_entities_wo_this += 1

                if ref_action.startswith('regex'):
                    total_with_1_entities_w_regex += 1
                else:
                    total_with_1_entities_wo_regex += 1

                if ref_entities_1 and hyp_entities_1 and set(ref_entities_1) == set(hyp_entities_1):
                    first_entity_correct += 1
                    current_first_entity_correct = True

                    if w_this:
                        first_entity_correct_w_this += 1
                    else:
                        first_entity_correct_wo_this += 1

                    if ref_action.startswith('regex'):
                        first_entity_correct_w_regex += 1
                    else:
                        first_entity_correct_wo_regex += 1
            else:
                current_first_entity_correct = True

            if ref_entities_2 is not None or hyp_entities_2 is not None:
                total_with_2_entities += 1

                if w_this:
                    total_with_2_entities_w_this += 1
                else:
                    total_with_2_entities_wo_this += 1

                if ref_action.startswith('regex'):
                    total_with_2_entities_w_regex += 1
                else:
                    total_with_2_entities_wo_regex += 1

                if ref_entities_2 and hyp_entities_2 and set(ref_entities_2) == set(hyp_entities_2):
                    second_entity_correct += 1
                    current_second_entity_correct = True

                    if w_this:
                        second_entity_correct_w_this += 1
                    else:
                        second_entity_correct_wo_this += 1

                    if ref_action.startswith('regex'):
                        second_entity_correct_w_regex += 1
                    else:
                        second_entity_correct_wo_regex += 1
            else:
                current_second_entity_correct = True

            if not current_action_correct or not current_first_entity_correct or not current_second_entity_correct:
                if mode == 'error':
                    print(f'{src + " " if source else ""}{ref} vs {hyp}')
            else:
                if mode == 'correct':
                    print(f'{src + " " if source else ""}{ref} vs {hyp}')

        if mode != 'stat':
            return []

        results = {'action': {}, 'first_entity': {}, 'second_entity': {}, 'all': {}}

        print('action correct', action_correct / total)
        print('first entity correct', first_entity_correct / total_with_1_entities)
        print('second entity correct', second_entity_correct / total_with_2_entities)
        print('all', action_and_entities_correct / total)
        print()
        results['action']['all'] = action_correct / total
        results['first_entity']['all'] = first_entity_correct / total_with_1_entities
        results['second_entity']['all'] = second_entity_correct / total_with_2_entities
        results['all']['all'] = action_and_entities_correct / total

        if total_w_this > 0:
            print('action correct w/ this', action_correct_w_this / total_w_this)
            print('first entity correct w/ this', first_entity_correct_w_this / total_with_1_entities_w_this)
            print('second entity correct w/ this', second_entity_correct_w_this / total_with_2_entities_w_this)
            print('all w/ this', action_and_entities_correct_w_this / total_w_this)
            print()
            results['action']['w_this'] = action_correct_w_this / total_w_this
            results['first_entity']['w_this'] = first_entity_correct_w_this / total_with_1_entities_w_this
            results['second_entity']['w_this'] = second_entity_correct_w_this / total_with_2_entities_w_this
            results['all']['w_this'] = action_and_entities_correct_w_this / total_w_this

        if total_wo_this > 0:
            print('action correct w/o this', action_correct_wo_this / total_wo_this)
            print('first entity correct w/o this', first_entity_correct_wo_this / total_with_1_entities_wo_this)
            print('second entity correct w/o this', second_entity_correct_wo_this /
                  total_with_2_entities_wo_this if total_with_2_entities_wo_this > 0 else 1)
            print('all w/o this', action_and_entities_correct_wo_this / total_wo_this)
            print()
            results['action']['wo_this'] = action_correct_wo_this / total_wo_this
            results['first_entity']['wo_this'] = first_entity_correct_wo_this / total_with_1_entities_wo_this
            results['second_entity']['wo_this'] = second_entity_correct_wo_this / \
                total_with_2_entities_wo_this if total_with_2_entities_wo_this > 0 else 1
            results['all']['wo_this'] = action_and_entities_correct_wo_this / total_wo_this

        if total_wo_regex > 0:
            print('action correct w/o regex', action_correct_wo_regex / total_wo_regex)
            print('first entity correct w/o regex', first_entity_correct_wo_regex /
                  total_with_1_entities_wo_regex)
            print('second entity correct w/o regex', second_entity_correct_wo_regex /
                  total_with_2_entities_wo_regex)
            print('all w/o regex', action_and_entities_correct_wo_regex / total_wo_regex)
            print()
            results['action']['wo_regex'] = action_correct_wo_regex / total_wo_regex
            results['first_entity']['wo_regex'] = first_entity_correct_wo_regex / total_with_1_entities_wo_regex
            results['second_entity']['wo_regex'] = second_entity_correct_wo_regex / total_with_2_entities_wo_regex
            results['all']['wo_regex'] = action_and_entities_correct_wo_regex / total_wo_regex

        if total_w_regex > 0:
            print('action correct w/ regex', action_correct_w_regex / total_w_regex)
            print('first entity correct w/ regex', first_entity_correct_w_regex / total_with_1_entities_w_regex)
            print('second entity correct w/ regex', second_entity_correct_w_regex / total_with_2_entities_w_regex)
            print('all w/ regex', action_and_entities_correct_w_regex / total_w_regex)
            print()
            results['action']['w_regex'] = action_correct_w_regex / total_w_regex
            results['first_entity']['w_regex'] = first_entity_correct_w_regex / total_with_1_entities_w_regex
            results['second_entity']['w_regex'] = second_entity_correct_w_regex / total_with_2_entities_w_regex
            results['all']['w_regex'] = action_and_entities_correct_w_regex / total_w_regex

        print(regex_occurrences)
        print()

        return results


def check_error_responses(ref: str) -> tuple[int, str, str]:
    regex_case = -1
    entities_1 = None
    entities_2 = None
    for i, regex in enumerate(regexes):
        regex_match = re.match(regex, ref)

        if regex_match:
            regex_case = i

            if regex_case in (0, 5):
                entities_1 = [regex_match.group(1)]
            elif regex_case == 2:
                entities_1 = [regex_match.group(1)] if regex_match.group(1) == regex_match.group(3) else False
            elif regex_case == 3:
                entities_1 = and_split(regex_match.group(1))
            elif regex_case == 4:
                entities_1 = [regex_match.group(1)] if regex_match.group(1) == regex_match.group(1) else False

            if regex_case == 2:
                entities_2 = [regex_match.group(2)]
            elif regex_case == 3:
                entities_2 = [regex_match.group(2)]

            break

    return regex_case, entities_1, entities_2


def and_split(string_to_split: str) -> list[str]:
    splitted = string_to_split.split(', ')

    prefix = 'and '
    if len(string_to_split) > 1 and splitted[-1].startswith(prefix):
        splitted[-1] = splitted[-1][len(prefix):]

    return splitted


def print_results(results: list[dict[str, dict[str, float]]], transpose: bool = True) -> None:
    mapping = {'all': 'complete',
               'action': 'action',
               'first_entity': 'first object',
               'second_entity': 'second object'
               }

    datasets = {'all': 'all',
                'w_this': 'with pointing',
                'wo_this': 'without pointing',
                'wo_regex': 'without clarification responses',
                'w_regex': 'only clarification responses'}

    rows = datasets if transpose else mapping
    columns = mapping if transpose else datasets

    print(' & '.join(columns.values()))

    for row in rows:
        print(datasets[row] if transpose else mapping[row], end=' & ')

        print(' & '.join(' / '.join(((f'{results_n[column if transpose else row][row if transpose else column] * 100:.1f}'
                                      if not transpose and column in results_n[row] or transpose and row in results_n[column]
                                      else '-')
                                     for results_n in results))
                         for column in columns),
              end=r' \\' + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('ref_1_file', type=str, nargs='?')
    parser.add_argument('hyp_1_file', type=str, nargs='?')
    parser.add_argument('ref_2_file', type=str, nargs='?')
    parser.add_argument('hyp_2_file', type=str, nargs='?')
    parser.add_argument('ref_3_file', type=str)
    parser.add_argument('hyp_3_file', type=str)

    parser.add_argument('--mode', type=str, default='stat', choices=('stat', 'correct', 'error'))
    parser.add_argument('--source', type=str, default='')

    args = parser.parse_args()

    results_3 = calculate_results(args.ref_3_file, args.hyp_3_file, args.mode, args.source)

    if args.mode == 'stat':
        results = []

        if args.ref_1_file is not None and args.hyp_1_file is not None:
            results_1 = calculate_results(args.ref_1_file, args.hyp_1_file)
            results.append(results_1)

        if args.ref_2_file is not None and args.hyp_2_file is not None:
            results_2 = calculate_results(args.ref_2_file, args.hyp_2_file)
            results.append(results_2)

        results.append(results_3)

        print_results(results)
