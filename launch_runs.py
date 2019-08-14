import os
import itertools
from collections import OrderedDict

from advocacy_learning.training import experiments
from advocacy_learning.evaluation import result_helpers as rh


experiment_name = 'test'
root_path = '/insert/root/path/here'
data_path = '/insert/data/path/here'
# Available datasets
# mnist, mnist_imbalance, fmnist, mimic
data_type = 'fmnist'
save_path = '{}/experiments/{}'.format(root_path, experiment_name)
log_path = '{}/log/{}'.format(root_path, experiment_name)
model_path = '{}/saved_models/{}'.format(root_path, experiment_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)
    os.makedirs(log_path)
    os.makedirs(model_path)

(mimic,
 custom_dataset,
 dataset,
 data_dir,
 input_size,
 num_class) = rh.get_data_params(data_type=data_type, data_path=data_path)

seed_options = [i for i in range(5)]
train = True
evaluate = True

load_saved = False
overwrite = True
epoch_lim = 100
early_stopping_lim = 100
early_stopping_use_acc = False
batch_size = 512
validation_rate = 1
validation_percent = 0.1
learning_rate = 1e-4
weight_decay = 0
attention_reg = 10
# model_types:
# attention, multi-attention, advocate, honest_advocate
model_type_options = ['honest_advocate']
shared_encoder = True
max_size = None
# number of residual blocks, if None then basic architecture
advocate_capacity = None
judge_capacity = None
# only defined for mnist data_type
active_label = None

if active_label is not None:
    assert 'label' in data_type
    num_class = len(active_label)


config_arr = []
for setting in itertools.product(seed_options, model_type_options):
    seed, model_type = setting
    if use_input_noise_test:
        input_noise_test = input_noise_train
    else:
        input_noise_test = 0
    #judge_capacity = [judge_capacity_single, judge_capacity_single]
    #advocate_capacity = [advocate_capacity_single, advocate_capacity_single]
    name_args = OrderedDict({'seed': seed, 'model_type': model_type})

    name = '{}'.format(experiment_name)
    for key in name_args:
        name += ';{}={}'.format(key, name_args[key])
    name += ';'

    (advocate_model,
     advocate_training,
     include_advocates,
     non_deceptive_advocates,
     attention_type) = rh.model_settings(model_type)

    if loading_stopped:
        if os.path.exists('/data/ifox/advocacy_learning/experiments/{}/{}_pred_dict.pkl'.format(experiment_name, name)):
            print('already ran {}'.format(name))
            continue

    config_arr.append({'name': name, 'experiment_name': experiment_name, 'data_type': data_type, 'dataset': dataset,
                       'data_dir': data_dir, 'input_size': input_size, 'num_class': num_class,
                       'epoch_lim': epoch_lim,
                       'early_stopping_lim': early_stopping_lim, 'early_stopping_use_acc': early_stopping_use_acc,
                       'batch_size': batch_size, 'validation_rate': validation_rate,
                       'validation_percent': validation_percent, 'train': train,
                       'evaluate': evaluate, 'seed': seed, 'model_path': model_path,
                       'save_path': save_path, 'load_saved': load_saved, 'overwrite': overwrite,
                       'learning_rate': learning_rate, 'weight_decay': weight_decay,
                       'advocate_model': advocate_model, 'advocate_training': advocate_training,
                       'include_advocates': include_advocates, 'max_size': max_size,
                       'attention_reg': attention_reg, 'honest_advocates': non_deceptive_advocates,
                       'attention_type': attention_type,'custom_dataset': custom_dataset,
                       'active_label': active_label, 'advocate_capacity': advocate_capacity,
                       'judge_capacity': judge_capacity})
run_manager = experiments.RunManager(device_list=device_list, save_path=save_path)
for config in config_arr:
    run_manager.add_job(config)
run_manager.run_until_empty(10)

except:
    print('unlocking')
    if type(lock_file) == list:
        for lock_name in lock_file:
            os.rmdir('{}/{}'.format(lock_name, experiment_name))
            os.rmdir(lock_name)
    else:
        os.rmdir('{}/{}'.format(lock_file, experiment_name))
        os.rmdir(lock_file)
    print('finished {}'.format(experiment_name))
    raise

print('unlocking')
os.rmdir('{}/{}'.format(str(lock_file), experiment_name))
if type(lock_file) == list:
    for lock_name in lock_file:
        os.rmdir(lock_name)
else:
    os.rmdir(lock_file)
print('finished {}'.format(experiment_name))
