import os
from torch.utils.data import Dataset
import torch
import numpy as np
from ogb.nodeproppred import NodePropPredDataset
import json


class ArxivDataset(Dataset):

    def __init__(self, root, split="train"):
        self.dataset = NodePropPredDataset(name = "ogbn-arxiv")
        self.split_idx = self.dataset.get_idx_split()
        self.split_idx = self.split_idx[split]
        
        self.graph, self.labels = self.dataset[0]
        self.labels = self.labels.squeeze(1)

    def get_neighbors(self, index):

        neighbors = self.graph["edge_index"][1][self.graph["edge_index"][0] == index]
        return neighbors

    def build_string(self, index):

        '''
        Construct a string representation of a node and its neighbors:
        [NODE] feature1 feature2 ... featureN [NEIGHBORS] neighbor1 neighbor2 ... neighborN
        '''

        neighbors = self.get_neighbors(index)
        features = self.graph["node_feat"][index]

        string = "[NODE] "
        for feature in features:
            string += str(feature.item()) + " "

        string += "[NEIGHBORS] "
        
        for neighbor in neighbors:
            string += str(neighbor.item()) + " "
        
        return string


    def __len__(self):
        return len(self.split_idx)
    
    def __getitem__(self, index):
        string = self.build_string(index)
        return string, self.labels[index]

class NLGraph(Dataset):

    def __init__(self, root="./NLGraph", task="connectivity", split="train"):
        with open(root + "/" + task + "/" + split + ".json") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        index_string = str(index)
        return self.data[index_string]["question"], self.data[index_string]["answer"]

if __name__ == '__main__':
    # dataset = NLGraph(root="./NLGraph", task="connectivity", split="train")
    # dataset = NLGraph(root="./NLGraph", task="cycle", split="train")
    # dataset = NLGraph(root="./NLGraph", task="topology", split="train")
    dataset = ArxivDataset(root="./ogbn-arxiv")
    print(dataset[0])
