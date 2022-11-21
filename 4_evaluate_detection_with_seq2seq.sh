#!/bin/sh

MODEL=t5-large

SAVE_PATH=experiments/tmp
mkdir -p $SAVE_PATH

CLASSIFICATION_DATA_DIR=data/seq_classification

case $1 in
	'detection_with_seq2seq_generate')
		MODEL_FOR_PREDICTION=/project/OML/error_correction/models/ICNLSP/e2e_seq2seq_generate.ckpt
		DATA_PATH=data/seq2seq_e2e
		MODE=generate
	;;

	'detection_with_seq2seq_copy')
		MODEL_FOR_PREDICTION=/project/OML/error_correction/models/ICNLSP/e2e_seq2seq_copy.ckpt
		DATA_PATH=data/seq2seq_copy_e2e
		MODE=copy
	;;

	*)
		echo 'Please specify mode'
		exit
	;;
esac

# parameters: prediction_file, model, mode
predict() {
	python3 code/finetune.py \
		--predictions_file $1 \
		--model $2 \
		--max_epochs 0 \
		--gpus 1 \
		--prefix "correct: " \
		--num_sanity_val_steps 0 \
		--default_root_dir $SAVE_PATH \
		--mode $3 \
		--batch_size 32 \
		--beam_size 1 \
		--checkpoint_path $MODEL_FOR_PREDICTION \
		$SAVE_PATH \
		$DATA_PATH
}

# parameters: set
predict_and_evaluate() {
	predict $DATA_PATH/$1.predictions $MODEL $MODE
	python3 code/error_correction_evaluator.py --mode classification --remove_from_source_line " - ->" --add_to_source_line " |" --source_file $DATA_PATH/$1.source $DATA_PATH/$1.target $DATA_PATH/$1.predictions > $CLASSIFICATION_DATA_DIR/$1.seq2seq_$MODE
	python3 code/error_detection_evaluator.py --mode export_correct_corrections --output_file $CLASSIFICATION_DATA_DIR/$1.correct_seq2seq_$MODE $CLASSIFICATION_DATA_DIR/$1.tsv $CLASSIFICATION_DATA_DIR/$1.seq2seq_$MODE
}

for ext in source target; do
	mv $DATA_PATH/test.$ext $DATA_PATH/test_backup.$ext
	cp $DATA_PATH/valid.$ext $DATA_PATH/test.$ext
done

if [ $MODE = 'copy' ]
then
	mv $DATA_PATH/test.copy_target $DATA_PATH/test_backup.copy_target
	cp $DATA_PATH/valid.copy_target $DATA_PATH/test.copy_target
fi

predict_and_evaluate valid

for ext in source target; do
	mv $DATA_PATH/test_backup.$ext $DATA_PATH/test.$ext
done

if [ $MODE = 'copy' ]
then
	mv $DATA_PATH/test_backup.copy_target $DATA_PATH/test.copy_target
fi

predict_and_evaluate test
