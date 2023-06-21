#!/bin/sh

MODEL=t5-large

SAVE_PATH=experiments/tmp
mkdir -p $SAVE_PATH

case $1 in
	'detection_correction_with_classification_and_e2e_seq2seq_generate')
		FALSE_NEGATIVES_VALID=0
		TRUE_NEGATIVES_VALID=264
		FALSE_NEGATIVES_TEST=220
		TRUE_NEGATIVES_TEST=959
		MODEL_FOR_PREDICTION=/project/OML/error_correction/models/ICNLSP/e2e_seq2seq_generate.ckpt
		DATA_PATH=data/seq2seq_after_classification
		MODE=generate
	;;

	'detection_correction_with_classification_and_seq2seq_generate')
		FALSE_NEGATIVES_VALID=0
		TRUE_NEGATIVES_VALID=264
		FALSE_NEGATIVES_TEST=220
		TRUE_NEGATIVES_TEST=959
		MODEL_FOR_PREDICTION=/project/OML/error_correction/models/ICNLSP/seq2seq_generate.ckpt
		DATA_PATH=data/seq2seq_after_classification
		MODE=generate
	;;

	'detection_correction_with_classification_and_e2e_seq2seq_copy')
		FALSE_NEGATIVES_VALID=0
		TRUE_NEGATIVES_VALID=264
		FALSE_NEGATIVES_TEST=220
		TRUE_NEGATIVES_TEST=959
		MODEL_FOR_PREDICTION=/project/OML/error_correction/models/ICNLSP/e2e_seq2seq_copy.ckpt
		DATA_PATH=data/seq2seq_copy_after_classification
		MODE=copy
	;;

	'detection_correction_with_classification_and_seq2seq_copy')
		FALSE_NEGATIVES_VALID=0
		TRUE_NEGATIVES_VALID=264
		FALSE_NEGATIVES_TEST=220
		TRUE_NEGATIVES_TEST=959
		MODEL_FOR_PREDICTION=/project/OML/error_correction/models/ICNLSP/seq2seq_copy.ckpt
		DATA_PATH=data/seq2seq_copy_after_classification
		MODE=copy
	;;

	'detection_correction_with_e2e_seq2seq_generate')
		FALSE_NEGATIVES_VALID=0
		TRUE_NEGATIVES_VALID=0
		FALSE_NEGATIVES_TEST=0
		TRUE_NEGATIVES_TEST=0
		MODEL_FOR_PREDICTION=/project/OML/error_correction/models/ICNLSP/e2e_seq2seq_generate.ckpt
		DATA_PATH=data/seq2seq_e2e
		MODE=generate
	;;

	'detection_correction_with_e2e_seq2seq_copy')
		FALSE_NEGATIVES_VALID=0
		TRUE_NEGATIVES_VALID=0
		FALSE_NEGATIVES_TEST=0
		TRUE_NEGATIVES_TEST=0
		MODEL_FOR_PREDICTION=/project/OML/error_correction/models/ICNLSP/e2e_seq2seq_copy.ckpt
		DATA_PATH=data/seq2seq_copy_e2e
		MODE=copy
	;;

	'correction_with_e2e_seq2seq_generate')
		FALSE_NEGATIVES_VALID=0
		TRUE_NEGATIVES_VALID=0
		FALSE_NEGATIVES_TEST=0
		TRUE_NEGATIVES_TEST=0
		MODEL_FOR_PREDICTION=/project/OML/error_correction/models/ICNLSP/e2e_seq2seq_generate.ckpt
		DATA_PATH=data/seq2seq
		MODE=generate
	;;

	'correction_with_e2e_seq2seq_copy')
		FALSE_NEGATIVES_VALID=0
		TRUE_NEGATIVES_VALID=0
		FALSE_NEGATIVES_TEST=0
		TRUE_NEGATIVES_TEST=0
		MODEL_FOR_PREDICTION=/project/OML/error_correction/models/ICNLSP/e2e_seq2seq_copy.ckpt
		DATA_PATH=data/seq2seq_copy
		MODE=copy
	;;

	'correction_with_seq2seq_generate')
		FALSE_NEGATIVES_VALID=0
		TRUE_NEGATIVES_VALID=0
		FALSE_NEGATIVES_TEST=0
		TRUE_NEGATIVES_TEST=0
		MODEL_FOR_PREDICTION=/project/OML/error_correction/models/ICNLSP/seq2seq_generate.ckpt
		DATA_PATH=data/seq2seq
		MODE=generate
	;;

	'correction_with_seq2seq_copy')
		FALSE_NEGATIVES_VALID=0
		TRUE_NEGATIVES_VALID=0
		FALSE_NEGATIVES_TEST=0
		TRUE_NEGATIVES_TEST=0
		MODEL_FOR_PREDICTION=/project/OML/error_correction/models/ICNLSP/seq2seq_copy.ckpt
		DATA_PATH=data/seq2seq_copy
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

for ext in source target; do
	mv $DATA_PATH/test.$ext $DATA_PATH/test_backup.$ext
	cp $DATA_PATH/valid.$ext $DATA_PATH/test.$ext
done

if [ $MODE = 'copy' ]
then
	mv $DATA_PATH/test.copy_target $DATA_PATH/test_backup.copy_target
	cp $DATA_PATH/valid.copy_target $DATA_PATH/test.copy_target
fi

predict $DATA_PATH/valid.predictions $MODEL $MODE
python3 code/error_correction_evaluator.py --add_number_classification_false_negatives $FALSE_NEGATIVES_VALID --add_number_classification_true_negatives $TRUE_NEGATIVES_VALID $DATA_PATH/valid.target $DATA_PATH/valid.predictions
echo "Is FALSE_NEGATIVES_VALID = $FALSE_NEGATIVES_VALID and TRUE_NEGATIVES_VALID = $TRUE_NEGATIVES_VALID correct?"


for ext in source target; do
	mv $DATA_PATH/test_backup.$ext $DATA_PATH/test.$ext
done

if [ $MODE = 'copy' ]
then
	mv $DATA_PATH/test_backup.copy_target $DATA_PATH/test.copy_target
fi

predict $DATA_PATH/test.predictions $MODEL $MODE
python3 code/error_correction_evaluator.py --add_number_classification_false_negatives $FALSE_NEGATIVES_TEST --add_number_classification_true_negatives $TRUE_NEGATIVES_TEST $DATA_PATH/test.target $DATA_PATH/test.predictions
echo "Is FALSE_NEGATIVES_TEST = $FALSE_NEGATIVES_TEST and TRUE_NEGATIVES_TEST = $TRUE_NEGATIVES_TEST correct?"

