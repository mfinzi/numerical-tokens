import together
from datasets import QM9, MNISTSuperpixels
from tokenizer import TokenizerSettings, tokenize
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import re
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tokenizer import TokenizerSettings
import fire
import requests

def sample_completions(model_name, prefix, patience=5):
    try:
        output = together.Complete.create(
                prompt = prefix, 
                model = model_name, max_tokens=300, temperature=1.0)
        text_out = output['output']['choices'][0]['text']
        return text_out
    except requests.exceptions.HTTPError as e:
        print("Request failed, retrying...")
        return sample_completions(model_name, prefix, patience-1) if patience > 0 else f"Request failed too many times with {e}"

def eval_qm9(model_name,tokenizer_model="meta-llama/Llama-2-7b-hf", datadir=None, samples=3, debug=False):
    base_tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, token=True)
    settings = TokenizerSettings(base_tokenizer)
    dataset = QM9(root=datadir, split='val') if datadir is not None else QM9(split='val')
    pattern_dict = r'(\b(?:ε_HOMO|ε_LUMO|U_0|U|H|G|cv|μ|α|Δε|⟨R^2⟩|ZPVE)):\s*(-?\d+\.?\d*)'
    all_preds = []
    for n, elem in enumerate(tqdm(dataset)):
        tokenized = tokenize(elem, settings)
        text = settings.base_tokenizer.decode(tokenized)
        prefix, target = text.split('targets:')
        prefix += 'targets:'
        matches = []
        for i in range(samples):
            text_out = sample_completions(model_name, prefix)
            matches_dict = {match[0]: float(match[1]) for match in re.findall(pattern_dict,text_out)}
            matches_dict['sample'] = i
            matches.append(matches_dict)

        target_dict = {match[0]: float(match[1]) for match in re.findall(pattern_dict, target)}
        target_dict['sample'] = -1
        matches.append(target_dict)
        df = pd.DataFrame(matches)
        df['n'] = n
        all_preds.append(df)
    final_df = pd.concat(all_preds)
    final_df = final_df.set_index(['n', final_df.index])
    final_df.reset_index(inplace=True)
    final_df.to_csv(f'dataset_files/{model_name}{"_debug" if debug else ""}.csv'.lower())
    preds = final_df[final_df['sample'] != -1]
    targets = final_df[final_df['sample'] == -1]

    median_preds = preds.groupby('n').median() #Median for MAE

    median_preds.reset_index(inplace=True) # Reset index to make 'n' a column again for easier comparison
    targets.reset_index(drop=True, inplace=True)  # Drop the old index to align with median_preds

    # Compute MAE for each property (assuming your properties are columns in the DataFrame)
    mae_values = {}
    for column in median_preds.columns:
        if column not in ['n', 'sample']:  # Skip non-property columns
            # Compute MAE between median prediction and target for the current property
            mae = np.mean(np.abs(median_preds[column] - targets[column]))
            mae_values[column] = mae

    for prop, mae in mae_values.items():
        print(f"MAE for {prop}: {mae:.3f}")


if __name__ == '__main__':
    fire.Fire(eval_qm9)
