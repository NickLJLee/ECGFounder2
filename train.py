import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from scipy.io import loadmat
import wfdb
from scipy import signal
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from scipy.interpolate import interp1d
from mne.filter import filter_data, notch_filter

import torch
from torch.utils import data
from torch import nn
import torch.optim as optim

import sys
sys.path.append('../..')

import clip
from model import CLIP
from simple_tokenizer import SimpleTokenizer

import torch
import torch
from torch.utils import data
import matplotlib.pyplot as plt
import os


class ECGDataset(Dataset):
    def __init__(self, txt_path, ecg_path = ''):
        self.window_size = 5000 #2500
        self.fs = 500.0 #250.0
        self.data = pd.read_csv(txt_path)
        # Drop rows where HashFileName or deid_t_diagnosis_original is NaN
        self.data = self.data.dropna(subset=['HashFileName', 'deid_t_diagnosis_original'])
        self.ecg_path = ecg_path
        self.leads = ['I', 'II', 'III', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'aVF', 'aVL', 'aVR']

    def __len__(self):
        return len(self.data)
    
    def normalization(self,signal):
        return (signal - np.mean(signal)) / (np.std(signal) +1e-8) 
    
    def resample_unequal(self, ts, fs_in, fs_out):
        if fs_in == 0:
            return ts
        t = len(ts) / fs_in
        fs_in, fs_out = int(fs_in), int(fs_out)
        if fs_out == fs_in:
            return ts
        if 2*fs_out == fs_in:
            return ts[::2]
        else:
            x_old = np.linspace(0, 1, num=len(ts), endpoint=True)
            x_new = np.linspace(0, 1, num=int(t * fs_out), endpoint=True)
            y_old = ts
            f = interp1d(x_old, y_old, kind='linear')
            y_new = f(x_new)
            return y_new
    
    def preprocess(self, arr, sample_rate):
        """
        arr has shape (n_channel, n_length)

        """
        out = []
        for tmp in arr:

            # resample
            if sample_rate != self.fs:
                tmp = self.resample_unequal(tmp, sample_rate, self.fs)

            # filter
            tmp = notch_filter(tmp, self.fs, 60, verbose='ERROR')
            tmp = filter_data(tmp, self.fs, 0.5, 50, verbose='ERROR')

            out.append(tmp)

        out = np.array(out)
        n_length = out.shape[1]

        if n_length > self.window_size: # crop center window_size for longer
            i_start = (n_length-self.window_size)//2
            i_end = i_start+self.window_size
            out = out[:, i_start:i_end]
        elif n_length < self.window_size: # pad zeros for shorter
            pad_len = np.zeros((len(self.leads), self.window_size-n_length))
            out = np.concatenate([out, pad_len], axis=1)

        return out
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        hash_file_name = row['HashFileName']
        txt = row['deid_t_diagnosis_original']
        s_dirs = [f"S{i:04d}" for i in range(1, 5)] # Assuming there are only 4 'S' directories, modify as needed
        year_dirs = [str(i) for i in range(1987, 2024)] # Assuming years range from 1980 to 2020
        month_dirs = [f"{i:02}" for i in range(1, 13)]
        try:
        # Iterate over all possible combinations to find the file
          file_found = False
          for s_dir in s_dirs:
              for year_dir in year_dirs:
                  for month_dir in month_dirs:
                      file_path = f"{self.ecg_path}/{s_dir}/{year_dir}/{month_dir}/{hash_file_name}"
                      mat_path = f"{file_path}.mat"
                      if os.path.exists(mat_path):
                          mat_data = loadmat(mat_path)
                          ecg_data = mat_data['val']
                          hea_path = f"{file_path}.hea"
                          with open(hea_path, 'r') as hea_file:
                              lines = hea_file.readlines()
                              first_line = lines[0].strip() 
                              elements = first_line.split()  
                              sample_rate = 500
                              sample_rate = elements[2] if len(elements) > 2 else "Unknown"
                              sample_rate = int(sample_rate)
                          file_found = True
                          break
                  if file_found:
                      break
              if file_found:
                  break
          #hd5_file = h5py.File(f"{self.ecg_path}/{hash_file_name}", "r")
          #for k in list(hd5_file['ecg'].keys()):
          #  ecg_data_list = [torch.tensor(hd5_file[i,:]) for lead in self.leads]
          #  ecg_data = torch.stack(ecg_data_list, dim=0)
          if file_found ==False:
            normal_file_path = '/data1/1shared/lijun/data/HEEDB/ECG/WFDB/S0001/2018/12/de_112344838_20190330104655_20190512102946.mat'
            row_1 = self.data.iloc[0] #number of normal sample
            txt_1 = row_1['deid_t_diagnosis_original']
            mat_data_1 = loadmat(normal_file_path)
            ecg_data_1 = mat_data_1['val']
            hea_path_1 = '/data1/1shared/lijun/data/HEEDB/ECG/WFDB/S0001/2018/12/de_112344838_20190330104655_20190512102946.hea'
            with open(hea_path_1, 'r') as hea_file_1:
                lines_1 = hea_file_1.readlines()
                first_line_1 = lines_1[0].strip() 
                elements_1 = first_line_1.split()  
                sample_rate_1 = elements_1[2] if len(elements_1) > 2 else "Unknown"  
                sample_rate_1 = int(sample_rate_1)
            ecg_data_1 = np.array(ecg_data_1, dtype=float)
            ecg_data_1 = self.normalization(ecg_data_1)
            ecg_data_1 = self.preprocess(ecg_data_1, sample_rate_1)
            ecg_data_1 = torch.tensor(ecg_data_1, dtype=torch.float)
            # txt_1 = torch.tensor(txt_1, dtype=torch.float)
            ecg_data = ecg_data_1
            txt = txt_1

          ecg_data = np.array(ecg_data, dtype=float)
          ecg_data = self.normalization(ecg_data)
          ecg_data = self.preprocess(ecg_data, sample_rate)
          ecg_data = torch.tensor(ecg_data, dtype=torch.float)
          # txt = torch.tensor(txt, dtype=torch.float)
          sample = {'ecg': ecg_data, 'txt': txt}
          return sample
    
        except Exception as e:
          #print(f"Error reading file {file_path}: {e}")
          normal_file_path = '/data1/1shared/lijun/data/HEEDB/ECG/WFDB/S0001/2018/12/de_112344838_20190330104655_20190512102946.mat'
          row_1 = self.data.iloc[0] #number of normal sample
          txt_1 = row_1['deid_t_diagnosis_original']
          mat_data_1 = loadmat(normal_file_path)
          ecg_data_1 = mat_data_1['val']
          hea_path_1 = '/data1/1shared/lijun/data/HEEDB/ECG/WFDB/S0001/2018/12/de_112344838_20190330104655_20190512102946.hea'
          with open(hea_path_1, 'r') as hea_file_1:
              lines_1 = hea_file_1.readlines()
              first_line_1 = lines_1[0].strip() 
              elements_1 = first_line_1.split()  
              sample_rate_1 = elements_1[2] if len(elements_1) > 2 else "Unknown"  
              sample_rate_1 = int(sample_rate_1)
          ecg_data_1 = np.array(ecg_data_1, dtype=float)
          ecg_data_1 = self.normalization(ecg_data_1)
          ecg_data_1 = self.preprocess(ecg_data_1, sample_rate_1)
          ecg_data_1 = torch.tensor(ecg_data_1, dtype=torch.float)
          # txt_1 = torch.tensor(txt_1, dtype=torch.float)
          sample = {'ecg': ecg_data_1, 'txt': txt_1}
          return sample

def load_data(txt_path='', batch_size=128): 
    if torch.cuda.is_available():  
        dev = "cuda:6" 
        cuda_available = True
        print('Using CUDA.')
    else:  
        dev = "cpu"  
        cuda_available = False
        print('Using CPU.')
    
    device = torch.device(dev)
    
    if cuda_available: 
        torch.cuda.set_device(device)
    
    torch_dset = ECGDataset(txt_path=txt_path)
    data_loader = data.DataLoader(torch_dset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())
    
    return data_loader, device
    
# Testing the dataset with the provided files
#cxr_filepath = '/home/ubuntu/data'
#txt_filepath = '/home/ubuntu/data/sample.csv'
#dataset = ECGDataset(txt_filepath, cxr_filepath)
#ECGdataloader,dev = load_data(txt_path = txt_filepath, ecg_path = cxr_filepath)

    
def load_clip(model_path=None, pretrained=False, context_length=77):
    '''
    FUNCTION: load_clip
    -------------------------------
    This function loads in a model with the CLIP model 
    architecture. 
    
    args: 
        * model_path (optional) - path to model weights that the model
        will be initialized with 
        * pretrained (optional) - if True, will load the pretrained 
        CLIP model
        * context_length (optional) - length of the maximum number of 
        tokens that can be inputted into the CLIP model
    '''

    params = {
        'context_length': context_length,
        'vocab_size': 49408,
        'transformer_width': 512,
        'transformer_heads': 8,
        'transformer_layers': 12
    }
    
    # set device 
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    
    if pretrained: 
        # load clip pre-trained model
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        print("Loaded in pretrained model.")
    else: 
        model = CLIP(**params)
        print("Loaded in clip model.")
    
    # if a model_path is provided, load in weights to backbone
    if model_path != None: 
        model.load_state_dict(torch.load(model_path, map_location=device))
    return model
    
    
def preprocess_text(texts, model):
#     if model.context_length is None: 
#         model = model.module
        
    _tokenizer = SimpleTokenizer()
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(str(text)) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), model.context_length, dtype=torch.long)
    
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > model.context_length:
            tokens = tokens[:model.context_length]
            tokens[model.context_length - 1] = eot_token
        result[i, :len(tokens)] = torch.tensor(tokens)
    return result
