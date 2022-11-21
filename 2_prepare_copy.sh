#!/bin/sh

for data_path in data/seq2seq_copy data/seq2seq_copy_e2e; do
	for set in train valid test; do
		mv $data_path/$set.copy_target $data_path/$set.copy_target_not_adapted
		python3 code/adapt_to_tokenizer.py --prefix "correct: " t5-large $data_path/$set.source $data_path/$set.target $data_path/$set.copy_target_not_adapted > $data_path/$set.copy_target
	done
done
