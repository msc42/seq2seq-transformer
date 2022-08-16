# seq2seq-transformer

## copyright
This project was created in my work time as employee of the Karlsruhe Institute of Technology (KIT).
This work has been supported by the German Federal Ministry of Education and Research (BMBF) under the project OML (01IS18040A).

The validation dataset was collected by Datoid, LLC.

## setup
```
./1_setup.sh
conda activate seq2seq-transformer
```

to use copy mode, please patch the Hugging Face transformers code:
```
patch $(dirname $(which python))/../lib/python*/site-packages/transformers/generation_utils.py code/generation_utils.patch
```

## training
```./2_train.sh```

you can skip the training and download the models from

## evaluation
you can change `MODEL_PATH` to the model that you want to evaluate

```./3_evaluate_nlu_with_pointing.sh```
