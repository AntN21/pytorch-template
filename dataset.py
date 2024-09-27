import torch
from torch.utils.data import Dataset

import numpy as np
# from dataset import PatientData, DictObj
class PatientData:
    def __init__(self, individual_id,
                label = None,
                features = None,
                ecg_time_series = None,
                hrv_time_series = None,
                hrv_features = None,
                sample_weight = None):

        self.id = individual_id
        self.label = label
        self.features = features
        self.ecg_time_series = ecg_time_series

        self.hrv_time_series = hrv_time_series
        self.hrv_features = hrv_features

        self.additional_features = None

        if sample_weight is None:
            self.sample_weight = 1.
        else:
            self.sample_weight = sample_weight
            
    def add_features(self, new_feats):
        self.features = torch.cat([self.features,new_feats],axis=-1)

class DictObj:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __repr__(self):
        return f"{self.__dict__}"

# import torch
# from torch.utils.data import Dataset

class PatientDataset(Dataset):
    def __init__(self, individuals_data, features_duration = '5m', ecg_duration = "30s", hrv_duration = "1h", ecg_index = 4, hrv_index = 2):
        """
        Args:
            individuals_data (list): A list of IndividualData objects.
        """
        self.individuals_data = [individual for individual in individuals_data if int( individual.id) != 18]
        # print(self.individuals_data)
        # for individual in self.individuals_data:
        #     print(individual.id)

        self.ecg_duration = ecg_duration
        self.hrv_duration = hrv_duration
        self.ecg_index = ecg_index
        self.hrv_index = hrv_index
        self.assert_all_information_is_here()

        for ind,individual in enumerate(self.individuals_data):
            # print(type(individual.features))
            # print(individual.features.__dict__.keys())
            self.individuals_data[ind].features = individual.features[features_duration][2,:]
        # self.features = self.features[features_duration]


        self.pad_hrv()
        self.compute_normalizers()

    def __len__(self):
        # Return the total number of individuals
        return len(self.individuals_data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the individual to retrieve.

        Returns:
            A dictionary containing the individual's ID, features, ECG time series, and HRV time series.
        """
        individual = self.individuals_data[idx]

        # Convert the individual's data to a dictionary

        features = ( individual.features - self.features_mean) / self.features_std
        ecg_time_series = (individual.ecg_time_series[self.ecg_duration][np.random.randint(10),:]- self.ecg_mean) / self.ecg_std
        hrv_time_series = (individual.hrv_time_series[self.hrv_duration][np.random.randint(10),:] - self.hrv_mean) / self.hrv_std
        return {
            'id': individual.id,
            'features': torch.FloatTensor(features),
            'sample_weight':torch.FloatTensor([individual.sample_weight]),
            'ecg_time_series': torch.FloatTensor(ecg_time_series).unsqueeze(0) ,
            'hrv_time_series': torch.FloatTensor(hrv_time_series).unsqueeze(0),
            'label':  individual.label,
            #'hrv_features' : individual.hrv_features,
        }#, individual.label

    def get_max_hrv_length(self, duration):
        return max([max([len(hrv_ts) for hrv_ts in individual.hrv_time_series[duration]]) for individual in self.individuals_data])
        #return max([len(individual.hrv_time_series[]) for individual in self.individuals_data])

    def pad_hrv(self):
        """
        Pads the HRV time series to the length of the longest time series.
        """
        durations =  self.individuals_data[0].hrv_time_series.__dict__.keys()
        for duration in durations:
          max_length = self.get_max_hrv_length(duration)
          for ind,individual in enumerate(self.individuals_data):
              self.individuals_data[ind].hrv_time_series[duration] = np.array([np.pad(signal, (0, max_length - len(signal)), mode= 'median') #constant', constant_values=pad_value)
                               for signal in individual.hrv_time_series[duration]])
              # self.individuals_data[ind].hrv_time_series[duration] =  torch.nn.functional.pad(individual.hrv_time_series, (0, max_length - len(individual.hrv_time_series)), mode='constant')
    def compute_normalizers(self):
        """
        Computes the mean and standard deviation of the ECG and HRV time series.
        """
        ecgs = []
        hrvs = []
  
        features = []

        for individual in self.individuals_data:
            ecgs.append(individual.ecg_time_series[self.ecg_duration])
            
            hrvs.append(individual.hrv_time_series[self.hrv_duration])
            
            features.append(individual.features)
            
        self.ecg_mean = np.mean(ecgs)
        self.ecg_std = np.std(ecgs)
        self.hrv_mean = np.mean(hrvs)
        self.hrv_std = np.std(hrvs)
        self.features_mean = np.mean(features,axis=0)
        self.features_std = np.std(features,axis=0)

    def assert_all_information_is_here(self):
        for individual in self.individuals_data:
            for attr_name, attr_value in individual.__dict__.items():
                if attr_value is None:
                    print(f"Invidual {individual.id} is missing {attr_name}.")
                elif isinstance(attr_value,np.ndarray) and attr_value.isnan().any():
                    print(f"Invidual {individual.id} has None values in {attr_name}.")


class PatientDataset(Dataset):
    def __init__(self, individuals_data, additional_feature_names = [], features_duration = '5m', ecg_duration = "30s", hrv_duration = "1h"):
        """
        Args:
            individuals_data (list): A list of IndividualData objects.
        """
        # self.individuals_data = [individual for individual in individuals_data if int( individual.id) != 18]
        self.data_length = len(individuals_data) * 10
        # inputs : ecg_times_series, hrv_times_series, features 
        # labels
        self.device = torch.device('cpu')
        self.ecg_time_series = []
        self.hrv_time_series = []
        self.features = []
        self.labels = []
        self.ids = []
        for individual in individuals_data:
            for i in range(10):
                self.ecg_time_series.append(individual.ecg_time_series[ecg_duration][i,:])
                self.hrv_time_series.append(individual.hrv_time_series[hrv_duration][i])
                features = [individual.features[features_duration][i,:]]
                for add_feature_name in additional_feature_names:
                    features.append(individual.additional_features[features_duration][add_feature_name][i,:])
                
                self.features.append(np.concatenate(features,axis=-1))

                self.ids.append(individual.id)
                self.labels.append(individual.label)


        self.ecg_time_series = np.array(self.ecg_time_series)
        self.features = np.array(self.features)
        self.ids = np.array(self.ids)
        self.labels = np.array(self.labels,dtype=int)

        self.assert_all_information_is_here()

        self.pad_hrv()
        self.to_tensor()
        self.compute_normalizers()
        

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
        ecg_time_series = (self.ecg_time_series[idx,:]- self.ecg_mean) / self.ecg_std
        hrv_time_series = (self.hrv_time_series[idx,:] - self.hrv_mean) / self.hrv_std
        return {
            'id': self.ids[idx],
            'features': features.to(self.device),
            # 'sample_weight':torch.FloatTensor([individual.sample_weight]),
            'ecg_time_series': ecg_time_series.to(self.device),
            'hrv_time_series': hrv_time_series.to(self.device),
            'label':  self.labels[idx].to(self.device),
            #'hrv_features' : individual.hrv_features,
        }#, individual.label

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
        
            
        self.ecg_mean = torch.mean(self.ecg_time_series)
        self.ecg_std = torch.std(self.ecg_time_series)
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
        
        self.ecg_time_series = torch.FloatTensor(self.ecg_time_series)
        self.hrv_time_series = torch.FloatTensor(self.hrv_time_series)
        self.features = torch.FloatTensor(self.features)
        self.labels = torch.LongTensor(self.labels)
        self.ids = torch.tensor(self.ids)
    

if __name__ == '__main__':
    import os
    import sys
    sys.path.append('src') 
    import config
    data_list = torch.load(os.path.join(config.DATA_DIR,'data_list0.pth'))


    
    ecg_duration = '2m'
    hrv_duration = '5m'
    features_duration = hrv_duration

    dataset = PatientDataset(data_list,
                             additional_feature_names=[],#['shannon_encoding','multifracs','autoreg'],
                             features_duration=features_duration,
                             ecg_duration=ecg_duration,
                             hrv_duration=hrv_duration,

                             )

    torch.save(dataset,os.path.join(config.DATA_DIR,f'dataset_ecg_{ecg_duration}_hrv_{hrv_duration}_feats_{features_duration}.pth'))