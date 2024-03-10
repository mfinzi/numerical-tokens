import together
import os
from datasets import write_dataset
from evaluate import eval_qm9
import fire

def finetune(model_name="togethercomputer/llama-2-7b", dataset='qm9',datadir=None,debug=False, aug=True, overwrite=False):
    """ Finetune a model on a dataset
        Args:
        model_name: the model to finetune
        dataset: the dataset to finetune on
        datadir: the directory where the dataset is stored
        debug: if true, only use a small subset of the dataset
        """
    together.api_key = os.environ['TOGETHER_API_KEY']
    # check if the dataset files are present
    #output_file = f'dataset_files/{dataset}{"_debug" if debug else ""}.jsonl'.lower()
    output_file = write_dataset(model_name, dataset, datadir, debug, aug, overwrite)

    response = together.Files.upload(output_file)
    file_id = response['id']
    # finetune the model
    if not debug:
        response = together.Finetune.create(
            training_file=file_id,
            model=model_name,
            n_epochs=1,
            suffix=f"finetuned_{dataset}{'_debug' if debug else ''}",
            wandb_api_key=os.environ['WANDB_API_KEY'],
            confirm_inputs=False,
        )
        fine_tune_id = response['id']
        # evaluate the model
        #eval_qm9(model_name, model_name, datadir, debug=debug)

if __name__ == "__main__":
    fire.Fire(finetune)
    
    # check if the dataset files are present
    #output_file = f'dataset_files/{args.dataset}{"_debug" if args.debug else ""}.jsonl'.lower()