import argparse
import collections
import torch
import numpy as np
import os
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device

from experiments import get_experiment_config
from constants import RESULT_FOLDER
# Function to compute and plot the confusion matrix
from sklearn.metrics import confusion_matrix
# from data_loader.data_loaders import PatientData, DictObj
# from dataset import PatientData, DictObj
import csv

# fix random seeds for reproducibility
SEED = np.random.randint(999)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

# File to store results
RESULT_FILE = '_crossvalidation.csv'
def get_list(best_logs,key):
    return [best_log[key] for best_log in best_logs]

def log_experiment(best_logs, config):
    if config['name'] == 'MyTraining':
        log_myexperiment(best_logs,config)
    else:
        log_basic_experiment(best_logs,config)

def log_myexperiment(best_logs, config):
    file_path = os.path.join(RESULT_FOLDER,str(config['name']) + RESULT_FILE)
    headers = ['ID','Accuracy','std','LOSS','std','ECG','HRV','FEATURES','MULTIFRACS','SHANNON','AUTOREG']

    accuracies = get_list(best_logs,'val_accuracy')
    losses = get_list(best_logs,'val_loss')

    with open(file_path, mode='a+', newline='') as file:
        file.seek(0)  # Move to the start of the file to check if it is empty
        is_empty = file.read(1) == ''  # Check if the first character is empty
        
        writer = csv.writer(file)
        
        # If the file is empty, write headers
        if is_empty:
            writer.writerow(headers)
        
        add_feature_list = config['data_loader']['args']["additional_feature_names"]

        ecg_config = False if not config['arch']['args']['use_ecg_time_series'] else config['data_loader']['args']["ecg_duration"]
        hrv_config = False if not config['arch']['args']['use_hrv_time_series'] else config['data_loader']['args']["hrv_duration"]
        features_config = False if not config['arch']['args']['use_features'] else config['data_loader']['args']["features_duration"]
        writer.writerow([config['id'],
                          np.mean(accuracies), np.std(accuracies),
                          np.mean(losses), np.std(losses),
                          ecg_config, hrv_config, features_config,
                          'newmultifracs' in add_feature_list,
                          'shannon_encoding' in add_feature_list,
                          'autoreg' in add_feature_list,
                          ])
        
def log_basic_experiment(best_logs, config):
    file_path = os.path.join(RESULT_FOLDER,str(config['name']) + RESULT_FILE)
    headers = ['ID','Accuracy','std','LOSS','std']
    accuracies = get_list(best_logs,'val_multilabel_accuracy')
    losses = get_list(best_logs,'val_loss')
    with open(file_path, mode='a+', newline='') as file:
        file.seek(0)  # Move to the start of the file to check if it is empty
        is_empty = file.read(1) == ''  # Check if the first character is empty
        
        writer = csv.writer(file)
        
        # If the file is empty, write headers
        if is_empty:
            writer.writerow(headers)
        
        writer.writerow([config['id'],
                          np.mean(accuracies), np.std(accuracies),
                          np.mean(losses), np.std(losses),
                          ])

def show_confusion_matrix(model, test_loader, device='cuda'):
    device = 'cpu'
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []
    model = model.to(device)
    # Disable gradient calculation for efficiency
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs, labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Flatten the lists
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Compute the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)


def main(config):
    config = get_experiment_config(config, config['id'])
    print(config["name"])
    best_logs = []
    for _ in range(5):
        logger = config.get_logger('train')

        # setup data_loader instances
        data_loader = config.init_obj('data_loader', module_data)
        valid_data_loader = data_loader.split_validation()

        # for data, label in data_loader:
        #     print(data)
        # print(config.__dict__)
        
        if config["name"] == "MyTraining" or 'HRV':
            config['arch']['args']['feature_size'] = data_loader.dataset.get_feature_size()
            config['arch']['args']['num_classes'] = data_loader.dataset.get_num_classes()
        else:
            config['arch']['args']['num_classes'] = 5

        
        # build model architecture, then print to console
        model = config.init_obj('arch', module_arch)
        logger.info(model)
        
        # prepare for (multi-device) GPU training
        device, device_ids = prepare_device(config['n_gpu'])
        model = model.to(device)
        # data_loader.dataset.to(device)
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)

        # get function handles of loss and metrics
        criterion = getattr(module_loss, config['loss'])
        metrics = [getattr(module_metric, met) for met in config['metrics']]

        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

        trainer = Trainer(model, criterion, metrics, optimizer,
                        config=config,
                        device=device,
                        data_loader=data_loader,
                        valid_data_loader=valid_data_loader,
                        lr_scheduler=lr_scheduler)

        best_log = trainer.train()
        best_logs.append(best_log)

    log_experiment(best_logs,config)
    # show_confusion_matrix(model, valid_data_loader, device=device)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument("-e", "--exp", type=int, required=True, help="Experiment id") # args.add_argument("-e", "--exp", nargs="+", type=int, required=True, help="Experiment id")

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
