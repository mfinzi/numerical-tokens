import os
from torch.utils.data import Dataset
import numpy as np
import torch_geometric
from torch_geometric.datasets import QM9 as tgQM9
import h5py
from utils import fixed_seed
import json
from collections import namedtuple, OrderedDict
import fire
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tokenizer import TokenizerSettings
atomic_symbols = {
    1: 'H',   # Hydrogen
    2: 'He',  # Helium
    3: 'Li',  # Lithium
    4: 'Be',  # Beryllium
    5: 'B',   # Boron
    6: 'C',   # Carbon
    7: 'N',   # Nitrogen
    8: 'O',   # Oxygen
    9: 'F',   # Fluorine
    10: 'Ne'  # Neon
}
from tokenizer import Unitful


class QM9(Dataset):
    def __init__(self, root="~/datasets/QM9", position=True, edges=False, seed=37, split='train'):
        root = os.path.expanduser(root)
        super().__init__()
        self.pos = position
        self.edges = edges
        self.ds = tgQM9(root=root)
        with fixed_seed(seed):
            ids = np.random.permutation(len(self.ds)) # 130k
        all_ids = {'train':ids[:100000], 'val':ids[100000:100000+100], 'test':ids[110000:]}
        self.split_ids = all_ids[split]
        with open('qm9_props.json', 'r') as f:
            self.propdict = json.load(f)

    def __len__(self):
        return len(self.split_ids)
    def __getitem__(self, index):
        idx = self.split_ids[index]
        row = self.ds[idx]
        elem_symbols = [atomic_symbols[z.item()] for z in row.z]
        elem_pos = [(elem, pos) for elem, pos in zip(elem_symbols, row.pos)]
        x1 = elem_pos if self.pos else elem_symbols
        inp = (set(x1), row.edge_index.T) if self.edges else set(x1)
        targets = OrderedDict({vals['property']: Unitful(row.y[0,vals['index']].item(),vals['unit']) for k,vals in self.propdict.items()})
        return OrderedDict({'molecule': inp, 'targets':targets})

Superpixel = namedtuple('Superpixel', ['x', 'y', 'c'])

from tokenizer import tokenize
@tokenize.dispatch
def tokenize(obj: Superpixel, settings):
    # want (x,y: c)
    s=settings
    out = tokenize('(',s)+tokenize(obj.x,s)+tokenize(',',s)
    out += tokenize(obj.y,s)+tokenize(':',s)+tokenize(obj.c,s)+tokenize(')',s)
    return out


class MNISTSuperpixels(torch_geometric.datasets.MNISTSuperpixels):
    def __init__(self,root=os.path.expanduser('~/datasets/mnist-superpixels'),**kwargs):
        super().__init__(root,**kwargs)
    # coord scale is 0-25, std of unif [0-25] is 
    def __getitem__(self,index):
        datapoint = super().__getitem__(int(index))
        coords = (datapoint.pos-13.5)/5 # 2 x M array of coordinates
        bchannel = datapoint.x.T[0]#(datapoint.x.T-.1307)/0.3081 # 1 x M array of blackwhite info
        label = int(datapoint.y.item())
        img = [Superpixel(*coords[i], bchannel[i]) for i in range(len(bchannel)) if bchannel[i]>0.]
        return OrderedDict({'image': img, 'class': label})
        #return ((coords,bchannel),label)


import os
import requests
import zipfile
from typing import Dict, Tuple
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.io import read_off
from torch_geometric.transforms import FarthestPointSampling

class ModelNet10(Dataset):
    def __init__(self, root="~/datasets/ModelNet", categories=None, split='train', transform=None, num_points=100):
        super().__init__()
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        
        # Define the categories
        if categories is None:
            categories = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']
        self.categories = categories
        
        # Check if the data file exists
        data_file = os.path.join(self.root, f"modelnet10_{split}.pt")
        if os.path.exists(data_file):
            # Load the data from the file
            self.data = torch.load(data_file)
        else:
            # Download the dataset if not already downloaded
            if not os.path.exists(self.root):
                self.download()
            
            # Load the data
            self.data = []
            for category in self.categories:
                folder = os.path.join(self.root, "ModelNet10", category, split)
                files = [f for f in os.listdir(folder) if f.endswith('.off')]
                for file in files:
                    data = read_off(os.path.join(folder, file))
                    data.y = torch.tensor([self.categories.index(category)], dtype=torch.long)
                    self.data.append(data)
            
            self.data = FarthestPointSampling(num_points)(self.data)
            # Save the data to a file
            torch.save(self.data, data_file)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index) -> OrderedDict:
        data = self.data[index]
        if self.transform is not None:
            data = self.transform(data)
        
        # Convert point cloud to a set of tuples
        point_cloud = set(tuple(point.tolist()) for point in data.pos)
        
        # Get the class name
        class_name = self.categories[data.y.item()]
        
        return OrderedDict({'point_cloud': point_cloud, 'class': class_name})
    
    def download(self):
        url = "http://vision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
        filename = "ModelNet10.zip"
        
        print(f"Downloading {url}...")
        response = requests.get(url)
        
        with open(filename, 'wb') as f:
            f.write(response.content)
        
        print(f"Extracting {filename}...")
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(self.root)



import json
from tqdm.auto import tqdm
import math
import numpy as np
def write_tokenized_dataset_to_file(dataset, tokenizer_settings, output_file, debug=False, epochs=1):
    """ write out dataset to a json that looks like
        {"text": "..."}
        {"text": "..."}
        {"text": "..."}
        for together finetuning
        """
    with open(output_file, "w") as f:
        n = math.ceil(epochs)
        for e in range(n):
            # iterate through the dataset 
            perm = np.random.permutation(len(dataset))
            outlen = len(dataset) if e!=n-1 else int(len(dataset)*(epochs-e))
            for i in tqdm(range(outlen)):
                tokenized = tokenize(dataset[perm[i]], tokenizer_settings)
                text = tokenizer_settings.base_tokenizer.decode(tokenized)
                json_line = json.dumps({"text": text})
                f.write(json_line + "\n")
                if i==500 and debug: break

def write_dataset(model_name="meta-llama/Llama-2-7b-hf", dataset='qm9',datadir=None,
                    debug=False, aug=True, overwrite=False, epochs=1, suffix=""):
    output_file = f'dataset_files/{dataset}{"_debug" if debug else ""}{"_aug" if aug else ""}_{epochs:.1f}_{suffix}.jsonl'.lower()
    if os.path.exists(output_file) and not overwrite:
        return output_file
    
    tokenizer_model_name = "meta-llama/Llama-2-7b-hf" #TODO: fix this up? how do we get the tokenizer for the together models
    base_tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name, use_auth_token=True)
    settings = TokenizerSettings(base_tokenizer, random_transform=aug)
    ds = {
        'superpixelmnist': MNISTSuperpixels,
        'qm9': QM9,
        'modelnet10': ModelNet10,
    }[dataset]
    dataset = ds(root=datadir) if datadir is not None else ds()
    write_tokenized_dataset_to_file(dataset, settings, output_file, debug=debug, epochs=epochs)
    return output_file

if __name__ == '__main__':
    fire.Fire(write_dataset)
    
    