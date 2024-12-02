import torch
import pandas as pd
from torch.utils.data import Dataset, ConcatDataset
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms
from PIL import Image
import wfdb
from tqdm import tqdm
import os
from scipy.io import loadmat
from scipy.interpolate import interp1d
from mne.filter import filter_data, notch_filter

# these two datasets will read the raw ecg

class Ori_MIMIC_E_T_Dataset(Dataset):
    def __init__(self, ecg_meta_path, transform=None, **args):
        self.ecg_meta_path = ecg_meta_path
        self.mode = args['train_test']
        self.text_csv = args['text_csv']
        self.record_csv = args['record_csv']
        self.transform = transform

    def __len__(self):
        return (self.text_csv.shape[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get ecg
        study_id = self.text_csv['study_id'].iloc[idx]
        if study_id == self.record_csv['study_id'].iloc[idx]:
            path = self.record_csv['path'].iloc[idx]
        else:
            print('Error: study_id not match!')
        path = os.path.join(self.ecg_meta_path, path)
        ecg = wfdb.rdsamp(path)[0]
        ecg = ecg.T

        # check nan and inf
        if np.isinf(ecg).sum() == 0:
            for i in range(ecg.shape[0]):
                nan_idx = np.where(np.isnan(ecg[:, i]))[0]
                for idx in nan_idx:
                    ecg[idx, i] = np.mean(ecg[max(0, idx-6):min(idx+6, ecg.shape[0]), i])
        if np.isnan(ecg).sum() == 0:
            for i in range(ecg.shape[0]):
                inf_idx = np.where(np.isinf(ecg[:, i]))[0]
                for idx in inf_idx:
                    ecg[idx, i] = np.mean(ecg[max(0, idx-6):min(idx+6, ecg.shape[0]), i])

        # noramlize
        ecg = (ecg - np.min(ecg))/(np.max(ecg) - np.min(ecg) + 1e-8)

        # get raw text
        report = self.text_csv.iloc[idx][['report_0', 'report_1',
       'report_2', 'report_3', 'report_4', 'report_5', 'report_6', 'report_7',
       'report_8', 'report_9', 'report_10', 'report_11', 'report_12',
       'report_13', 'report_14', 'report_15', 'report_16', 'report_17']]
        # only keep not NaN
        report = report[~report.isna()]
        # concat the report
        report = '. '.join(report)
        # preprocessing on raw text
        report = report.replace('EKG', 'ECG')
        report = report.replace('ekg', 'ecg')
        report = report.strip('*** ')
        report = report.strip(' ***')
        report = report.strip('***')
        report = report.strip('=-')
        report = report.strip('=')
        # convert to all lower case
        report = report.lower()

        sample = {'ecg': ecg, 'raw_text': report}

        if self.transform:
            if self.mode == 'train':
                sample['ecg'] = self.transform(sample['ecg'])
                sample['ecg'] = torch.squeeze(sample['ecg'], dim=0)
            else:
                sample['ecg'] = self.transform(sample['ecg'])
                sample['ecg'] = torch.squeeze(sample['ecg'], dim=0)
        return sample


class Ori_ECG_TEXT_Dsataset:

    def __init__(self, ecg_path, csv_path, dataset_name='mimic'):
        # if you use this dataset, please replace ecg_path from config.yaml to the 'your path/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0'
        self.ecg_path = ecg_path
        self.csv_path = csv_path
        self.dataset_name = dataset_name
        self.csv = pd.read_csv(self.csv_path, low_memory=False)
        self.record_csv = pd.read_csv(os.path.join(self.ecg_path, 'record_list.csv'), low_memory=False)
        
        # sort and reset index by study_id
        self.csv = self.csv.sort_values(by=['study_id'])
        self.csv.reset_index(inplace=True, drop=True)
        self.record_csv = self.record_csv.sort_values(by=['study_id'])
        self.record_csv.reset_index(inplace=True, drop=True)

        # split train and val
        self.train_csv, self.val_csv, self.train_record_csv, self.val_record_csv = \
            train_test_split(self.csv, self.record_csv, test_size=0.02, random_state=42)
        # sort and reset index by study_id
        self.train_csv = self.train_csv.sort_values(by=['study_id'])
        self.val_csv = self.val_csv.sort_values(by=['study_id'])
        self.train_csv.reset_index(inplace=True, drop=True)
        self.val_csv.reset_index(inplace=True, drop=True)

        self.train_record_csv = self.train_record_csv.sort_values(by=['study_id'])
        self.val_record_csv = self.val_record_csv.sort_values(by=['study_id'])
        self.train_record_csv.reset_index(inplace=True, drop=True)
        self.val_record_csv.reset_index(inplace=True, drop=True)
        
        print(f'train size: {self.train_csv.shape[0]}')
        print(f'val size: {self.val_csv.shape[0]}')

    def get_dataset(self, train_test, T=None):

        if train_test == 'train':
            print('Apply Train-stage Transform!')

            Transforms = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            print('Apply Val-stage Transform!')

            Transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

        
        if self.dataset_name == 'mimic':
            
            if train_test == 'train':
                misc_args = {'train_test': train_test,
                   'text_csv': self.train_csv,
                   'record_csv': self.train_record_csv}
            else:
                misc_args = {'train_test': train_test,
                   'text_csv': self.val_csv,
                   'record_csv': self.val_record_csv}
            
        
            dataset = Ori_MIMIC_E_T_Dataset(ecg_data=self.ecg_path,
                                       transform=Transforms,
                                       **misc_args)
            print(f'{train_test} dataset length: ', len(dataset))
        
        return dataset


# these two datasets will read the ecg from preprocessed npy file
# we suggest to use these two datasets for accelerating the IO speed


class MIMIC_E_T_Dataset(Dataset):
    def __init__(self, ecg_meta_path, transform=None, **args):
        self.ecg_meta_path = ecg_meta_path
        self.mode = args['train_test']
        if self.mode == 'train':
            self.ecg_data = os.path.join(ecg_meta_path, 'mimic_ecg_train.npy')
            self.ecg_data = np.load(self.ecg_data, 'r')
            
        else:
            self.ecg_data = os.path.join(ecg_meta_path, 'mimic_ecg_val.npy')
            self.ecg_data = np.load(self.ecg_data, 'r')


        self.text_csv = args['text_csv']

        self.transform = transform

    def __len__(self):
        return (self.text_csv.shape[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # we have to divide 1000 to get the real value
        ecg = self.ecg_data[idx]/1000
        # ecg = (ecg - np.min(ecg))/(np.max(ecg) - np.min(ecg) + 1e-8)
        

        # get raw text
        report = self.text_csv.iloc[idx]['total_report']

        sample = {'ecg': ecg, 'raw_text': report}

        if self.transform:
            if self.mode == 'train':
                sample['ecg'] = self.transform(sample['ecg'])
                sample['ecg'] = torch.squeeze(sample['ecg'], dim=0)
            else:
                sample['ecg'] = self.transform(sample['ecg'])
                sample['ecg'] = torch.squeeze(sample['ecg'], dim=0)
        return sample


class ECG_TEXT_Dsataset:

    def __init__(self, data_path, dataset_name='mimic'):
        self.data_path = data_path
        self.dataset_name = dataset_name

        print(f'Load {dataset_name} dataset!')
        self.train_csv = pd.read_csv(os.path.join(self.data_path, 'train.csv'), low_memory=False)
        self.val_csv = pd.read_csv(os.path.join(self.data_path, 'val.csv'), low_memory=False)

        print(f'train size: {self.train_csv.shape[0]}')
        print(f'val size: {self.val_csv.shape[0]}')
        print(f'total size: {self.train_csv.shape[0] + self.val_csv.shape[0]}')
        
    def get_dataset(self, train_test, T=None):

        if train_test == 'train':
            print('Apply Train-stage Transform!')

            Transforms = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            print('Apply Val-stage Transform!')

            Transforms = transforms.Compose([
                transforms.ToTensor(),
            ])
        
            
        if self.dataset_name == 'mimic':
            
            if train_test == 'train':
                misc_args = {'train_test': train_test,
                   'text_csv': self.train_csv,
                   }
            else:
                misc_args = {'train_test': train_test,
                   'text_csv': self.val_csv,
                   }
            
        
            dataset = MIMIC_E_T_Dataset(ecg_meta_path=self.data_path,
                                       transform=Transforms,
                                       **misc_args)
            print(f'{train_test} dataset length: ', len(dataset))
        
        return dataset


class train_HEEDB_Dataset(Dataset):
    def __init__(self, txt_path='/data/0shared/lijun/code/ECGFounder_CLIP/data/csv/train.csv', ecg_path = '/data/0shared/lijun/data/HEEDB/WFDB/'):
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
            normal_file_path = '/data/0shared/lijun/data/HEEDB/WFDB/S0001/2018/12/de_118498412_20190407080833_20190409122412.mat'
            row_1 = self.data.iloc[-1] #number of normal sample
            txt_1 = row_1['deid_t_diagnosis_original']
            mat_data_1 = loadmat(normal_file_path)
            ecg_data_1 = mat_data_1['val']
            hea_path_1 = '/data/0shared/lijun/data/HEEDB/WFDB/S0001/2018/12/de_118498412_20190407080833_20190409122412.hea'
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
          normal_file_path = '/data/0shared/lijun/data/HEEDB/WFDB/S0001/2018/12/de_118498412_20190407080833_20190409122412.mat'
          row_1 = self.data.iloc[-1] #number of normal sample
          txt_1 = row_1['deid_t_diagnosis_original']
          mat_data_1 = loadmat(normal_file_path)
          ecg_data_1 = mat_data_1['val']
          hea_path_1 = '/data/0shared/lijun/data/HEEDB/WFDB/S0001/2018/12/de_118498412_20190407080833_20190409122412.hea'
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


class val_HEEDB_Dataset(Dataset):
    def __init__(self, txt_path='/data/0shared/lijun/code/ECGFounder_CLIP/data/csv/val.csv', ecg_path = '/data/0shared/lijun/data/HEEDB/WFDB/'):
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
            normal_file_path = '/data/0shared/lijun/data/HEEDB/WFDB/S0001/2018/12/de_117585665_20200430112630_20200502164636.mat'
            row_1 = self.data.iloc[-1] #number of normal sample
            txt_1 = row_1['deid_t_diagnosis_original']
            mat_data_1 = loadmat(normal_file_path)
            ecg_data_1 = mat_data_1['val']
            hea_path_1 = '/data/0shared/lijun/data/HEEDB/WFDB/S0001/2018/12/de_117585665_20200430112630_20200502164636.hea'
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
          normal_file_path = '/data/0shared/lijun/data/HEEDB/WFDB/S0001/2018/12/de_117585665_20200430112630_20200502164636.mat'
          row_1 = self.data.iloc[-1] #number of normal sample
          txt_1 = row_1['deid_t_diagnosis_original']
          mat_data_1 = loadmat(normal_file_path)
          ecg_data_1 = mat_data_1['val']
          hea_path_1 = '/data/0shared/lijun/data/HEEDB/WFDB/S0001/2018/12/de_117585665_20200430112630_20200502164636.hea'
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