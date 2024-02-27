import os
from torch.utils.data import Dataset
import numpy as np
import torch_geometric
from torch_geometric.datasets import QM9
import h5py


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

def text_qm9_dataset(tokenizer, pos=True,seed=37, split='train', dir="~/datasets/QM9"):
    # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.QM9.html#torch_geometric.datasets.QM9
    root = os.path.expanduser(dir)
    ds = QM9(root=root)
    ids = np.random.permutation(len(ds)) # 130k
    all_ids = {'train':ids[:100000], 'val':ids[100000:100000+100], 'test':ids[110000:]}
    split_ids = all_ids[split]
    def gen():
        for idx in split_ids:
            row = ds[idx]
            elem_symbols = [atomic_symbols[z.item()] for z in row.z]
            elem_pos = [(elem, pos) for elem, pos in zip(elem_symbols, row.pos)]
            inp = (elem_pos if pos else elem_symbols, row.edge_index.T)
            target = row.y[0,2] #HOMO attribute TODO: make more general
            prompt = "\n\n HOMO: "
            yield (inp, (prompt, target))
    return gen


class ModelNet40(Dataset):
    ignored_index = -100
    class_weights = None
    stratify=True
    num_targets=40
    classes=['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car',
        'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot',
        'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor',
        'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
        'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase',
        'wardrobe', 'xbox']
    default_root_dir = '~/datasets/ModelNet40/'
    def __init__(self,root_dir=default_root_dir,train=True,transform=None,size=1024):
        super().__init__()
        #self.transform = torchvision.transforms.ToTensor() if transform is None else transform
        train_x,train_y,test_x,test_y = load_data(os.path.expanduser(root_dir),classification=True)
        self.coords = train_x if train else test_x
        # SWAP y and z so that z (gravity direction) is in component 3
        self.coords[...,2] += self.coords[...,1]
        self.coords[...,1] = self.coords[...,2]-self.coords[...,1]
        self.coords[...,2] -= self.coords[...,1]
        # N x m x 3
        self.labels = train_y if train else test_y
        self.coords_std = np.std(train_x,axis=(0,1))
        self.coords /= self.coords_std
        self.coords = self.coords.transpose((0,2,1)) # B x n x c -> B x c x n
        self.size=size
        #pt_coords = torch.from_numpy(self.coords)
        #self.coords = FarthestSubsample(ds_frac=size/2048)((pt_coords,pt_coords))[0].numpy()

    def __getitem__(self,index):
        return torch.from_numpy(self.coords[index]).float(), int(self.labels[index])
    def __len__(self):
        return len(self.labels)

class MNISTSuperpixels(torch_geometric.datasets.MNISTSuperpixels):
    ignored_index = -100
    class_weights = None
    stratify=True
    num_targets = 10
    # def __init__(self,*args,**kwargs):
    #     super().__init__(*args,**kwargs)
    # coord scale is 0-25, std of unif [0-25] is 
    def __getitem__(self,index):
        datapoint = super().__getitem__(int(index))
        coords = (datapoint.pos.T-13.5)/5 # 2 x M array of coordinates
        bchannel = (datapoint.x.T-.1307)/0.3081 # 1 x M array of blackwhite info
        label = int(datapoint.y.item())
        return ((coords,bchannel),label)



def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = []
    return (data, label, seg)

def _load_data_file(name):
    f = h5py.File(name)
    data = f["data"][:]
    label = f["label"][:]
    return data, label

def load_data(dir,classification = False):
    data_train0, label_train0,Seglabel_train0  = load_h5(dir + 'ply_data_train0.h5')
    data_train1, label_train1,Seglabel_train1 = load_h5(dir + 'ply_data_train1.h5')
    data_train2, label_train2,Seglabel_train2 = load_h5(dir + 'ply_data_train2.h5')
    data_train3, label_train3,Seglabel_train3 = load_h5(dir + 'ply_data_train3.h5')
    data_train4, label_train4,Seglabel_train4 = load_h5(dir + 'ply_data_train4.h5')
    data_test0, label_test0,Seglabel_test0 = load_h5(dir + 'ply_data_test0.h5')
    data_test1, label_test1,Seglabel_test1 = load_h5(dir + 'ply_data_test1.h5')
    train_data = np.concatenate([data_train0,data_train1,data_train2,data_train3,data_train4])
    train_label = np.concatenate([label_train0,label_train1,label_train2,label_train3,label_train4])
    train_Seglabel = np.concatenate([Seglabel_train0,Seglabel_train1,Seglabel_train2,Seglabel_train3,Seglabel_train4])
    test_data = np.concatenate([data_test0,data_test1])
    test_label = np.concatenate([label_test0,label_test1])
    test_Seglabel = np.concatenate([Seglabel_test0,Seglabel_test1])

    if classification:
        return train_data, train_label, test_data, test_label
    else:
        return train_data, train_Seglabel, test_data, test_Seglabel