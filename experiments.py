# from shared import (
#     ID, NAME, NB_EPOCHS, DATALOADER, BATCH_SIZE,
#     TRAIN, VALIDATION, TEST,
#     ARCHITECTURE, MODEL,
#     N_PARAMS,
#     OPTIMIZER, LR, PARAMS
# )
# from model import ConvModel
# import torch
# # from data_loader import get_dataloaders
# from typing import Tuple
import itertools

def get_experiment_config(config, exp: int) -> dict:
    
    # config["id"] = exp

    # Define possible values for each option
    option1 = [True,False]        # 2 possibilities
    option2 = [True,False]         # 2 possibilities
    option3 = [True,False]         # 2 possibilities
    option4 = ['30s','1m','2m','5m','1h']      # 5 possibilities
    option5 = ['30s','1m','2m','5m']      # 4 possibilities

    # Your list
    elements = ['newmultifracs','shannon_encoding','autoreg']
    # Use itertools.product to generate all combinations
    subsets = [list(subset) for subset in itertools.product(*[[None, e] for e in elements])]
    # Remove None values to get proper subsets
    subsets = [[e for e in subset if e is not None] for subset in subsets]

    option6 =['newmultifracs', 'shannon_encoding','autoreg'] , # 1 possibility #subsets      # 8 possibilities

    all_combinations = list(itertools.product(option1, option2, option3, option4, option5, option6))
    filtered_combinations = []
    for comb in all_combinations:
        # print(comb)
        if comb[3] in ['30s','1m','2m'] and comb[2] and 'newmultifracs' in comb[5]:
            continue
        else:
            filtered_combinations.append(comb)
        
    experiment_combination = filtered_combinations[exp]

    config['arch']['args']['use_ecg_time_series'] = experiment_combination[0]
    config['arch']['args']['use_hrv_time_series'] =  experiment_combination[1]
    config['arch']['args']['use_features'] = experiment_combination[2]
    config['data_loader']['args']["features_duration"] = experiment_combination[3]
    config['data_loader']['args']["hrv_duration"] = experiment_combination[3]
    config['data_loader']['args']["ecg_duration"] = experiment_combination[4]
    config['data_loader']['args']["additional_feature_names"] = experiment_combination[5]

    print(f"Experiment {exp} / {len(filtered_combinations)}:\n\
                USE_ECG: {experiment_combination[0]}\n\
                USE_HRV: {experiment_combination[1]}\n\
                USE_FEATURES: {experiment_combination[2]}\n\
                HRV_DURATION: {experiment_combination[3]}\n\
                ECG_DURATION: {experiment_combination[4]}\n\
                ADDITIONAL_FEATURES: {experiment_combination[5]}\
                ")
    return config


# def get_training_content(config: dict, device="cuda") -> Tuple[ConvModel, torch.optim.Optimizer, dict]:
#     model = ConvModel(1, **config[MODEL][ARCHITECTURE])
#     assert config[MODEL][NAME] == ConvModel.__name__
#     config[MODEL][N_PARAMS] = model.count_parameters()
#     optimizer = torch.optim.Adam(model.parameters(), **config[OPTIMIZER][PARAMS])
#     dl_dict = get_dataloaders(config, factor=1, device=device)
#     return model, optimizer, dl_dict


# if __name__ == "__main__":
#     config = get_experiment_config(0)
#     print(config)
#     model, optimizer, dl_dict = get_training_content(config)
#     print(len(dl_dict[TRAIN].dataset))
