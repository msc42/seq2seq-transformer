# seq2seq-transformer

## copyright
This project was created in my work time as employee of the Karlsruhe Institute of Technology (KIT).
This work has been supported by the German Federal Ministry of Education and Research (BMBF) under the project OML (01IS18040A).

The validation dataset of the NLU component of the dialog using pointing gestures was collected by Datoid, LLC.

## setup
```
./1_setup.sh
conda activate seq2seq-transformer
```

to use copy mode, please patch the Hugging Face transformers code:
```
patch $(dirname $(which python))/../lib/python*/site-packages/transformers/generation_utils.py code/generation_utils.patch
```

## data adaption
to use data for the copy mode
```./2_prepare_copy.sh```

## training
```./3_train.sh```

you can skip the training and download the models from <https://www.dropbox.com/sh/vzgfznqi93x29bg/AADJjER-CqX2ZeWQ0bT7gjOea?dl=0> for the NLU component of the dialog using pointing gestures system and from <https://www.dropbox.com/sh/hovvnj3cky55psi/AAAZHk_OfXjuLAlHeFPoWOAua?dl=0> for the error correction and extraction system

## evaluation
you can change `MODEL_PATH` to the model that you want to evaluate

evaluate NLU component for dialog using pointing gestures:
```./4_evaluate_nlu_with_pointing.sh```


you can change `MODEL_FOR_PREDICTION` to the model that you want to evaluate

evaluate error correction detection:
```./4_evaluate_detection_with_seq2seq```

evaluate error correction and extraction:
```./4_evaluate_seq2seq.sh```
