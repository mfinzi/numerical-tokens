# numerical-tokens

## Installation
```shell
pip install -r requirements.txt
```

To finetune a together model on the data:

First we need to generate the datasets in the right text format.
To do this run 

```shell
python datasets.py --dataset qm9
```
(will take around 1 hour per dataset, options are qm9 or superpixelmnist)

Then upload the training files to together:
```shell
together files upload qm9.jsonl
```
and take note of the file id produced from the upload.

Then run finetuning using the together API
```shell
together finetune create --training-file $FILE_ID --model $MODEL_NAME --wandb-api-key $WANDB_API_KEY
```