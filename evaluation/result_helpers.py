import ast
import os
import joblib
import sklearn
import torch
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm_notebook as tqdm
import torchvision

from advocacy_learning.data import dataset as ds


def process_filename(file):
    processed = {}
    characteristics = file.split(';')[:-1]
    processed['experiment_name'] = characteristics[0]
    for i in range(1, len(characteristics)):
        key, val = characteristics[i].split('=')
        try:
            processed[key] = ast.literal_eval(val)
        except:
            # messy
            processed[key] = val
    return processed


def load_and_process_results(file, name):
    path = '/data3/ifox/advocacy_learning/experiments/{}/'.format(name)
    try:
        tup = joblib.load(path+file)
        pred = np.concatenate(tup[0])
        true = np.concatenate(tup[1])
        pd = {'label_pred': pred, 'label_true': true}
    except EOFError as e:
        print('error occurred, skipping')
        return -1
    file_characteristics = process_filename(file)
    file_characteristics.update(pd)
    return file_characteristics


def onehotify(arr):
    ret = np.zeros((len(arr), 10))
    for i in range(len(arr)):
        ret[i, arr[i]-1] = 1
    return ret


def get_performance(res, binary):
    if binary:
        res['AUROC'] = sklearn.metrics.roc_auc_score(res['label_true'], res['label_pred'][:, 1])
        res['AUPR'] = sklearn.metrics.average_precision_score(res['label_true'], res['label_pred'][:, 1])
    else:
        res['AUROC'] = sklearn.metrics.roc_auc_score(onehotify(res['label_true']), res['label_pred'])
        res['AUPR'] = sklearn.metrics.average_precision_score(onehotify(res['label_true']), res['label_pred'])
    res['Acc'] = sklearn.metrics.accuracy_score(res['label_true'], np.argmax(res['label_pred'], axis=1))
    return res


def load_it_up(name, binary=False, retrain=False):
    directory = '/data3/ifox/advocacy_learning/experiments/{}/'.format(name)
    files = os.listdir(directory)
    if retrain:
        pd_files = list(filter(lambda x: 'pred_dict_retrain' in x, files))
    else:
        pd_files = list(filter(lambda x: 'pred_dict.' in x, files))

    list_files = Parallel(n_jobs=10, verbose=1)(delayed(load_and_process_results)(f, name) for f in pd_files)

    list_files_updated = []
    for i in tqdm(range(len(list_files))):
        res = get_performance(list_files[i], binary)
        list_files_updated.append(res)

    df = pd.DataFrame.from_dict(list_files_updated)
    return df, list_files_updated


def col_check(df):
    col_dict = {}
    for c in df.columns:
        try:
            col_dict[str(c)] = len(df[c].unique())
        except:
            continue
    return col_dict


def get_predictions(mdl, dat, device, attn_output=True):
    mdl.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for x, y in tqdm(dat, leave=False):
            labels.append(y.numpy())
            if attn_output:
                p, a = mdl(x.to(device))
            else:
                p = mdl(x.to(device))
            preds.append(p.cpu().numpy())
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
    mdl.train()
    return preds, labels


def get_data_params(data_type, data_path):
    if data_type == 'mnist':
        mimic = False
        custom_dataset = False
        dataset = torchvision.datasets.MNIST
        data_dir = '{}/mnist'.format(data_path)
        input_size = (1, 28, 28)
        num_class = 10
    elif data_type == 'fmnist':
        mimic = False
        custom_dataset = False
        dataset = torchvision.datasets.FashionMNIST
        data_dir = '{}/fashion_mnist'.format(data_path)
        input_size = (1, 28, 28)
        num_class = 10
    elif data_type == 'cifar10':
        mimic = False
        custom_dataset = False
        dataset = torchvision.datasets.CIFAR10
        data_dir = '{}/cifar_10'.format(data_path)
        input_size = (3, 32, 32)
        num_class = 10
    elif data_type == 'svhn':
        mimic = False
        custom_dataset = False
        dataset = torchvision.datasets.SVHN
        data_dir = '{}/svhn'.format(data_path)
        input_size = (1, 28, 28)
        num_class = 10
    elif data_type == 'mimic':
        mimic = True
        custom_dataset = True
        dataset = ds.MIMIC
        data_dir = '{}/advocacy_net/data3/mimic'.format(data_path)
        input_size = (76, 48, None)  # last dim for consistency
        num_class = 2
    elif data_type == 'mnist_letter':
        mimic = False
        custom_dataset = True
        dataset = ds.Anomaly
        data_dir = '{}/advocacy_net/data3/mnist_letter_anomaly'.format(data_path)
        input_size = (1, 28, 28)
        num_class = 10
    elif data_type == 'mimic_balance':
        mimic = True
        custom_dataset = True
        dataset = ds.BalancedMIMIC
        data_dir = '{}/advocacy_net/data3/mimic_balance'.format(data_path)
        input_size = (76, 48, None)
        num_class = 2
    elif data_type == 'mimic_downbalance':
        mimic = True
        custom_dataset = True
        dataset = ds.DownBalancedMIMIC
        data_dir = '{}/advocacy_net/data3/mimic_balance'.format(data_path)
        input_size = (76, 48, None)
        num_class = 2
    elif data_type == 'mnist_imbalance':
        mimic = False
        custom_dataset = True
        dataset = ds.ImbalancedMNIST
        data_dir = '{}/advocacy_net/data3/mnist_imbalance'.format(data_path)
        input_size = (1, 28, 28)
        num_class = 10
    elif data_type == 'synthetic':
        mimic = False
        custom_dataset = True
        dataset = ds.Synthetic
        data_dir = '{}/advocacy_net/data3/synthetic'.format(data_path)
        input_size = (1, 28, 28)
        num_class = 2
    elif data_type == 'mnist_label':
        mimic = False
        custom_dataset = True
        dataset = ds.MNISTLabel
        data_dir = '{}/mnist/processed'.format(data_path)
        input_size = (1, 28, 28)
        num_class = -1  # set depending on active_class
    elif data_type == 'fmnist_label':
        mimic = False
        custom_dataset = True
        dataset = ds.FMNISTLabel
        data_dir = '{}/fmnist/processed'.format(data_path)
        input_size = (1, 28, 28)
        num_class = -1  # set depending on active_class
    elif data_type == 'multi_mnist':
        mimic = False
        custom_dataset = True
        dataset = ds.MultiMNIST
        data_dir = '{}/advocacy_learning/data3/'.format(data_path)
        input_size = (1, 28, 28)
        num_class = 10  # set depending on active_class
    else:
        raise ValueError('No proper data_type given')

    return mimic, custom_dataset, dataset, data_dir, input_size, num_class


def model_settings(model_type):
    if model_type == 'random':
        advocate_model = True
        advocate_training = False
        include_advocates = False
        non_deceptive_advocates = False
        attention_type = 'multiply'
    elif model_type == 'attention':
        advocate_model = False
        advocate_training = False
        include_advocates = True
        non_deceptive_advocates = False
        attention_type = 'multiply'
    elif model_type == 'multi-attention':
        advocate_model = True
        advocate_training = False
        include_advocates = True
        non_deceptive_advocates = False
        attention_type = 'multiply'
    elif model_type == 'advocate':
        advocate_model = True
        advocate_training = True
        include_advocates = True
        non_deceptive_advocates = False
        attention_type = 'multiply'
    elif model_type == 'honest_advocate':
        advocate_model = True
        advocate_training = True
        include_advocates = True
        non_deceptive_advocates = True
        attention_type = 'multiply'
    elif model_type == 'embedding_advocate':
        advocate_model = True
        advocate_training = True
        include_advocates = True
        non_deceptive_advocates = False
        attention_type = 'embed'
    elif model_type == 'embedding_honest_advocate':
        advocate_model = True
        advocate_training = True
        include_advocates = True
        non_deceptive_advocates = False
        attention_type = 'embed'
    else:
        raise ValueError('No proper value for model_type given')
    return advocate_model, advocate_training, include_advocates, non_deceptive_advocates, attention_type
