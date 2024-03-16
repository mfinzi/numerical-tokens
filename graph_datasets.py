import os
from torch.utils.data import Dataset
import torch
import numpy as np
from ogb.nodeproppred import NodePropPredDataset
import json


class ArxivDataset(Dataset):

    '''
    Dataset from LLaGA -> larger graphs, probably out of scope
    '''

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

    def __init__(self, root="./NLGraph", task="connectivity", split="train", string_only=False):
        with open(root + "/" + task + "/" + split + ".json") as f:
            self.data = json.load(f)

        self.labels = []
        self.string_data = [] # full string (original data from NLGraph)
        self.graph_data = [] # parsed set of edges
        self.context_data = [] # context string - i.e., description of question

        keys = list(self.data.keys()) # maintaining same ordering
        for key in keys:
            data_string = self.data[key]["question"]
            self.string_data.append(data_string)
            self.labels.append(self.data[key]["answer"])

        # only returning string info
        if string_only:
            self.graph_data = self.string_data
            self.context_data = self.string_data
        else:
            for key in keys:
                data_string = self.data[key]["question"]
                graph, context = self.convert_string(data_string)
                self.graph_data.append(graph)
                self.context_data.append(context)

    def convert_string(self, string, ):
        '''
        Convert string into graph data structure representation
        '''

        # get index of Graph:
        # graph_index = string.find("Graph: ")
        graph_index = string.find(": ")
        graph_end_index = string.find("\nQ:")
        graph_string = string[graph_index + 3:graph_end_index]
        context_string = (string[:graph_index], string[graph_end_index:])

        graph = set()
        edges_string = graph_string.split(" ") 
        for edge in edges_string:
            # remove parens
            edge = edge.replace("(", "").replace(")", "")
            edge = edge.split(",")
            graph.add((int(edge[0]), int(edge[1])))
        return graph, context_string

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.graph_data[index], self.context_data[index], self.labels[index]


if __name__ == '__main__':
    dataset = NLGraph(root="./NLGraph", task="connectivity", split="train")
    # dataset = NLGraph(root="./NLGraph", task="cycle", split="train")
    
    # dataset = ArxivDataset(root="./ogbn-arxiv")
    print(dataset[0])
