from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import os
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np

class PatientHRVDataset(Dataset):
    def __init__(self, individuals_data, additional_feature_names = [], features_duration = '5m', hrv_duration = "1h"):
        """
        Args:
            individuals_data (list): A list of IndividualData objects.
        """
        individuals_data = [individual for individual in individuals_data if int( individual['id']) != 18]
        
        # inputs : ecg_times_series, hrv_times_series, features 
        # labels
        self.device = torch.device('cpu')
        self.ecg_time_series = []
        self.hrv_time_series = []
        self.features = []
        self.labels = []
        self.ids = []
        for individual in individuals_data:
            # print(individual["additional_features"][features_duration].keys())
            for i in range(len(individual["hrv_time_series"][hrv_duration])):
                # self.ecg_time_series.append(individual["ecg_time_series"][ecg_duration][i,:])
                self.hrv_time_series.append(individual["hrv_time_series"][hrv_duration][i])
                features = [individual["features"][features_duration][i,:-1]]
                for add_feature_name in additional_feature_names:
                    features.append(individual["additional_features"][features_duration][add_feature_name][i,:])
                
                self.features.append(np.concatenate(features,axis=-1))

                self.ids.append(individual['id'])
                self.labels.append(individual['label'])

        self.raw_hrv = self.hrv_time_series
        self.ecg_time_series = np.array(self.ecg_time_series)
        self.features = np.array(self.features)
        self.ids = np.array(self.ids)
        self.labels = np.array(self.labels,dtype=int)
        self.assert_all_information_is_here()

        self.pad_hrv()
        self.to_tensor()
        self.compute_normalizers()
        
        print(self.hrv_time_series.shape)
        print(self.features.shape)
        print(self.labels.shape)
        print(self.ids.shape)
        # self.print_feature_size()
        self.data_length = self.hrv_time_series.shape[0]

    def __len__(self):
        # Return the total number of individuals
        return self.data_length

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the individual to retrieve.

        Returns:
            A dictionary containing the individual's ID, features, ECG time series, and HRV time series.
        """
        

        # Convert the individual's data to a dictionary

        features = ( self.features[idx] - self.features_mean) / self.features_std
        # ecg_time_series = (self.ecg_time_series[idx,:]- self.ecg_mean) / self.ecg_std
        hrv_time_series = (self.hrv_time_series[idx,:] - self.hrv_mean) / self.hrv_std
        return {
            'id': self.ids[idx],
            'features': features, #.to(self.device),
            # 'sample_weight':torch.FloatTensor([individual.sample_weight]),
            'ecg_time_series': torch.empty((1,1)), #.to(self.device),
            'hrv_time_series': hrv_time_series.unsqueeze(0), #.to(self.device),
            # 'label':  self.labels[idx].to(self.device),
            #'hrv_features' : individual.hrv_features,
        }, self.labels[idx]# .to(self.device)

    def get_max_hrv_length(self):
        #return max([max([len(hrv_ts) for hrv_ts in individual.hrv_time_series[duration]]) for individual in self.individuals_data])
        return max([len(hrv_time_series) for hrv_time_series in self.hrv_time_series])

    def pad_hrv(self,length = None, mode='median', pad_value=0):
        """
        Pads the HRV time series to the length of the longest time series.
        """
        if length is None:
            length = self.get_max_hrv_length()

        if mode == 'constant':
            self.hrv_time_series = np.array([np.pad(signal[:length], (0, length - len(signal[:length])), mode= 'constant', constant_values=pad_value)
                            for signal in self.hrv_time_series])
        
        else:
            self.hrv_time_series = np.array([np.pad(signal[:length], (0, length - len(signal[:length])), mode= mode) #constant', constant_values=pad_value)
                            for signal in self.hrv_time_series])
        

    def compute_normalizers(self):
        """
        Computes the mean and standard deviation of the ECG and HRV time series.
        """
        
            
        # self.ecg_mean = torch.mean(self.ecg_time_series)
        # self.ecg_std = torch.std(self.ecg_time_series)
        self.hrv_mean = torch.mean(self.hrv_time_series)
        self.hrv_std = torch.std(self.hrv_time_series)
        self.features_mean = torch.mean(self.features, dim=0)
        self.features_std = torch.std(self.features, dim=0)

    def assert_all_information_is_here(self):
        
        for attr_name, attr_value in self.__dict__.items():
            if False and attr_value is None:
                print(f"Invidual {individual.id} is missing {attr_name}.")
            elif isinstance(attr_value,np.ndarray) and np.isnan(attr_value).any():
                # Get the indices of NaN values
                nan_indices = np.argwhere(np.isnan(attr_value))
                print(f'Missing  value in {attr_name}')
                for nan_indice in nan_indices:
                    print(f'Individual {self.ids[nan_indice[0]]} is missing a value')
                # print(nan_indices)
                # print(f"Invidual {individual.id} has None values in {attr_name}.")
            elif torch.is_tensor(attr_value) and torch.isnan(attr_value).any():
                # print(f"Invidual {individual.id} has None values in {attr_name}.")
                # Get the indices of NaN values
                nan_indices = np.argwhere(np.isnan(attr_value))
                print(f'Missing  value in {attr_name}')
                for nan_indice in nan_indices:
                    print(f'Individual {self.ids[nan_indice[0]]} is missing a value')

    def to(self, device):
        """ 
        Redefine self.device
        """
        self.device = device
    
    def to_tensor(self):
        
        # self.ecg_time_series = torch.FloatTensor(self.ecg_time_series)
        self.hrv_time_series = torch.FloatTensor(self.hrv_time_series)
        self.features = torch.FloatTensor(self.features)
        self.labels = torch.LongTensor(self.labels)
        self.ids = torch.tensor(self.ids)
    
    def get_feature_size(self):
        feature_size = self.features.shape[-1]
        # print(feature_size)
        return feature_size
    
    def get_num_classes(self):
        return len(np.unique(self.labels))

# if __name__ == '__main__':
#     import os
#     import sys
#     sys.path.append('src') 
#     import config
#     data_list = torch.load(os.path.join(config.DATA_DIR,'data_list0.pth'))


    
#     ecg_duration = '2m'
#     hrv_duration = '5m'
#     features_duration = hrv_duration

#     dataset = PatientDataset(data_list,
#                             additional_feature_names=[],#['shannon_encoding','multifracs','autoreg'],
#                             features_duration=features_duration,
#                             ecg_duration=ecg_duration,
#                             hrv_duration=hrv_duration,

#                             )

#     torch.save(dataset,os.path.join(config.DATA_DIR,f'dataset_ecg_{ecg_duration}_hrv_{hrv_duration}_feats_{features_duration}.pth'))

class HRVDataLoader(BaseDataLoader):
    """
    data loading using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                  additional_feature_names = [], features_duration = '5m',hrv_duration = "5m",
                  seed = None):
        # trsfm = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.1307,), (0.3081,))
        # ])
        self.seed =seed
        self.data_dir = data_dir
        if os.path.isdir(self.data_dir):
            individuals_data = []
            for data_file in os.listdir(self.data_dir):
                file_path = os.path.join(self.data_dir, data_file)
                individuals_data.append(torch.load(file_path))
        else:
            individuals_data = torch.load(self.data_dir) 
        # self.validation_split = validation_split
        self.dataset = PatientHRVDataset(individuals_data=individuals_data,
                                      additional_feature_names=additional_feature_names,
                                      features_duration=features_duration,
                                      hrv_duration=hrv_duration)
        
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(0,len(self.dataset),10)

        # np.random.seed(0)
        # np.random.shuffle(idx_full)
        # y = [patient[1] for patient in self.dataset]# if int(patient.id) !=18]
        ids = [patient[0]['id'].item() for patient in self.dataset]
        unique_ids = np.unique(ids)
        y = [self.dataset[ids.index(unique_id)][1] for unique_id in unique_ids]

        train_ids, valid_ids = train_test_split(unique_ids,
                                                    test_size=self.validation_split,
                                                    random_state=self.seed,
                                                    shuffle=True,
                                                    stratify=y)
        
        train_idx = [ind for ind,patient in enumerate(self.dataset) if patient[0]['id'].item() in train_ids]
        valid_idx = [ind for ind,patient in enumerate(self.dataset) if patient[0]['id'].item() in valid_ids]

        # print(max(train_idx))
        # print(max(valid_idx))
        # if isinstance(split, int):
        #     assert split > 0
        #     assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
        #     len_valid = split
        # else:
        #     len_valid = int(self.n_samples * split)

        # valid_idx = idx_full[0:len_valid]
        # train_idx = np.delete(idx_full, np.arange(0, len_valid))
        # print(train_idx)
        # print(valid_idx)
        idx = train_idx + valid_idx
        # print(len(idx))
        # print(max(idx))
        # print(len(self.dataset))
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler