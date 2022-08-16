#!/bin/sh

MODEL=t5-large
EPOCHS=2
RUNS=3

# parameters: save_path, data_path, mode
train() {
	mkdir -p $1

	python3 code/finetune.py \
		--model $MODEL \
		--tokenizer $MODEL \
		--min_epochs $EPOCHS \
		--max_epochs $EPOCHS \
		--gpus 1 \
		--prefix "correct: " \
		--predictions_file $1/predictions.txt \
		--num_sanity_val_steps 1 \
		--default_root_dir $1 \
		--mode $3 \
		--accumulate_grad_batches 1 \
		--batch_size 32 \
		--optimizer adam \
		--freeze_embedding \
		--freeze_encoders 2 \
		--freeze_decoders 0 \
		$1 \
		$2
}

for i in $(seq $RUNS); do
	train experiments/1nlu_with_pointing_$i data/nlu_with_pointing generate
done
