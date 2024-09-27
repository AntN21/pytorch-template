import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy as np
from experiments import get_experiment_config

def main(config):
    config = get_experiment_config(config, config['id'])
    logger = config.get_logger('show_dist')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )
    config['arch']['args']['feature_size'] = data_loader.dataset.get_feature_size()
    config['arch']['args']['num_classes'] = data_loader.dataset.get_num_classes()

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    outputs = []
    labels = []
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)

            out1 = model.ecg_conv(data["ecg_time_series"])
            out2 = model.hrv_conv(data["hrv_time_series"])

            features = model.features(data["features"])
        
            concat = torch.cat([data.flatten(out1), data.flatten(out2), features],dim=-1)
            outputs.append(concat)
            labels.append(target)
            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            # loss = loss_fn(output, target)
            # batch_size = data.shape[0]
            # total_loss += loss.item() * batch_size
            # for i, metric in enumerate(metric_fns):
            #     total_metrics[i] += metric(output, target) * batch_size

    outputs = torch.cat(outputs,dim=0)
    labels = torch.cat(labels,dim=0)
    labels = np.array(labels)
    my_data = np.array(outputs)
    # Normalize the data (this step is important for t-SNE to work effectively)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(my_data)

    # Initialize and fit t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_results = tsne.fit_transform(data_scaled)

    # Plot the t-SNE results
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', edgecolor='k')
    plt.title('t-SNE Plot')
    plt.xlabel('t-SNE component 1')
    plt.ylabel('t-SNE component 2')
    plt.show()
    # n_samples = len(data_loader.sampler)
    # log = {'loss': total_loss / n_samples}
    # log.update({
    #     met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    # })
    # logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument("-e", "--exp", type=int, required=True, help="Experiment id")
    config = ConfigParser.from_args(args)
    main(config)
