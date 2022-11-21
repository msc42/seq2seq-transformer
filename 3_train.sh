#!/bin/sh

MODEL=t5-large
EPOCHS=1
RUNS=1

case $1 in
	'nlu_pointing')
		MODEL_DIR=experiments/nlu_with_pointing
		DATA_DIR=data/nlu_with_pointing
		MODE=generate
	;;

	'generate')
		MODEL_DIR=/project/OML/error_correction/models/ICNLSP/exp/e2e_seq2seq_generate
		DATA_DIR=data/seq2seq
		MODE=generate
	;;

	'e2e_generate')
		MODEL_DIR=e2e_seq2seq_generate
		DATA_DIR=data/seq2seq_e2e
		MODE=generate
	;;

	'copy')
		MODEL_DIR=/project/OML/error_correction/models/ICNLSP/exp/e2e_seq2seq_copy
		DATA_DIR=data/seq2seq_copy
		MODE=copy
	;;

	'e2e_copy')
		MODEL_DIR=e2e_seq2seq_generate
		DATA_DIR=data/seq2seq_copy_e2e
		MODE=copy
	;;

	*)
		MODEL_DIR=experiments/nlu_with_pointing
		DATA_DIR=data/nlu_with_pointing
		MODE=generate
	;;
esac

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
		--batch_size 24 \
		--beam_size 1 \
		--optimizer adam \
		--freeze_embedding \
		--freeze_encoders 2 \
		--freeze_decoders 0 \
		$1 \
		$2
}

for i in $(seq $RUNS); do
	train $MODEL_DIR $DATA_DIR $MODE
done
