#!/bin/sh

# model to evaluate in more detail
MODEL_PATH=experiments/nlu_with_pointing_3/best_tfmr

# models to evaluate to find out what model to use, the folowing is appended to the path (i in 1 2 3): _$i/best_tfmr
MODEL_PATH_PREFIX=experiments/nlu_with_pointing

SAVE_PATH=experiments/tmp
mkdir -p $SAVE_PATH

DATA_PATH=data/nlu_with_pointing

# parameters: prediction_file, model
predict() {
	python3 code/finetune.py \
		--predictions_file $1 \
		--model $2 \
		--max_epochs 0 \
		--gpus 1 \
		--prefix "correct: " \
		--num_sanity_val_steps 0 \
		--default_root_dir $SAVE_PATH \
		--mode generate \
		--batch_size 32 \
		$SAVE_PATH \
		$DATA_PATH
}

evaluate_models_on_validation_data() {
	mv $DATA_PATH/test.source $DATA_PATH/real_test.source
	mv $DATA_PATH/test.target $DATA_PATH/real_test.target

	for i in $(seq 1 3); do
		cp $DATA_PATH/valid.source $DATA_PATH/test.source
		cp $DATA_PATH/valid.target $DATA_PATH/test.target
		predict $DATA_PATH/predictions.valid.$i.target ${MODEL_PATH_PREFIX}_${i}/best_tfmr
		python3 code/pointing_evaluator.py $DATA_PATH/valid.target $DATA_PATH/predictions.valid.$i.target
	done

	mv $DATA_PATH/real_test.source $DATA_PATH/test.source
	mv $DATA_PATH/real_test.target $DATA_PATH/test.target
}

evaluate() {
	mv $DATA_PATH/test.source $DATA_PATH/real_test.source
	mv $DATA_PATH/test.target $DATA_PATH/real_test.target

	cp $DATA_PATH/$1.source $DATA_PATH/test.source
	cp $DATA_PATH/$1.target $DATA_PATH/test.target
	predict $DATA_PATH/predictions.$1.3.target ${MODEL_PATH_PREFIX}_${i}/best_tfmr
	python3 code/pointing_evaluator.py $DATA_PATH/$1.target $DATA_PATH/predictions.$1.3.target

	mv $DATA_PATH/real_test.source $DATA_PATH/test.source
	mv $DATA_PATH/real_test.target $DATA_PATH/test.target
}

evaluate_models_on_validation_data

evaluate valid
evaluate test.gold
evaluate test.hyp

evaluate test.e2e_without_clips
evaluate test.e2e_with_clips
