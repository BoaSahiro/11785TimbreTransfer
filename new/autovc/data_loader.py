from torch.utils import data
import torch
import numpy as np
import pickle 
import os    
       
from multiprocessing import Process, Manager   


class Utterances(data.Dataset):
    """Dataset class for the Utterances dataset."""

    def __init__(self, root_dir, emb_dir, len_crop):
        """Initialize and preprocess the Utterances dataset."""
        self.root_dir = root_dir
        self.emb_dir = emb_dir
        self.len_crop = len_crop
        self.step = 10

        # load embeddings
        dataset = dict()
        for filename in os.listdir(self.emb_dir):
            if os.path.splitext(filename)[1] == '.npy':
                filepath = os.path.join(self.emb_dir, filename)
                ins_name = filename.split('_')[0]
                ins_emb = np.load(filepath)
                # ins_emb = np.reshape(ins_emb, (-1))
                ins_emb = ins_emb[10]
                dataset[ins_name] = [ins_emb]
        # load spectrograms
        for filename in os.listdir(self.root_dir):
            if os.path.splitext(filename)[1] == '.npy':
                filepath = os.path.join(self.root_dir, filename)
                ins_name = filename.split('_')[0]
                dataset[ins_name].append(np.load(filepath))

        self.train_dataset = dataset

        self.num_tokens = len(list(self.train_dataset.keys()))
                   
        
    def __getitem__(self, index):
        # pick a random instrument
        dataset = self.train_dataset

        list_ins = list(dataset.keys())
        list_uttrs = dataset[list_ins[index]]
        emb_org = list_uttrs[0]
        
        # pick random uttr with random crop
        a = np.random.randint(1, len(list_uttrs))
        tmp = list_uttrs[a].T
        if tmp.shape[0] < self.len_crop:
            len_pad = self.len_crop - tmp.shape[0]
            uttr = np.pad(tmp, ((0,len_pad),(0,0)), 'constant')
        elif tmp.shape[0] > self.len_crop:
            left = np.random.randint(tmp.shape[0]-self.len_crop)
            uttr = tmp[left:left+self.len_crop, :]
        else:
            uttr = tmp
        
        return uttr, emb_org
    

    def __len__(self):
        """Return the number of spkrs."""
        return self.num_tokens
    
    
    

def get_loader(root_dir, emb_dir, batch_size=16, len_crop=128, num_workers=0):
    """Build and return a data loader."""
    
    dataset = Utterances(root_dir, emb_dir, len_crop)
    
    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2**32))
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  worker_init_fn=worker_init_fn)
    return data_loader






