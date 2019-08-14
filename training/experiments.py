"""
Programatic interface for experiments
"""
import joblib
import os
import time
import multiprocessing as mp
import numpy as np
import itertools
import torch
import torchvision

from advocacy_learning.training.model import AdvocacyNet, AttentionNet, MIMICNet
from advocacy_learning.training import trainer


class RunConfig:
    def __init__(self, name, experiment_name, data_type, data_dir, dataset, input_size,
                 epoch_lim, early_stopping_lim, early_stopping_use_acc, batch_size,
                 validation_percent, validation_rate, train, evaluate, seed, model_path, save_path,
                 load_saved, overwrite, learning_rate, weight_decay, num_class,
                 advocate_model, advocate_training, include_advocates,
                 shared_encoder, input_noise_train, input_noise_test, label_noise, max_size, mimic,
                 retrain, retrain_type, attention_reg, non_deceptive_advocates, attention_type,
                 semi_supervised, supervised_number, supervised_rate, no_nine, custom_dataset, fancy_judge,
                 advocate_capacity, judge_capacity, active_label, multilabel):
        self.name = name
        self.experiment_name = experiment_name
        self.data_type = data_type
        self.data_dir = data_dir
        self.dataset = dataset
        self.input_size = input_size
        self.epoch_lim = epoch_lim
        self.early_stopping_lim = early_stopping_lim
        self.early_stopping_use_acc = early_stopping_use_acc
        self.batch_size = batch_size
        self.validation_percent = validation_percent
        self.validation_rate = validation_rate
        self.train = train
        self.evaluate = evaluate
        self.seed = seed
        self.model_path = model_path
        self.save_path = save_path
        self.load_saved = load_saved
        self.overwrite = overwrite
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_class = num_class
        self.advocate_model = advocate_model
        self.advocate_training = advocate_training
        self.include_advocates = include_advocates
        self.shared_encoder = shared_encoder
        self.input_noise_train = input_noise_train
        self.input_noise_test = input_noise_test
        self.label_noise = label_noise
        self.max_size = max_size
        self.mimic = mimic
        self.retrain = retrain
        self.retrain_type = retrain_type
        self.attention_reg = attention_reg
        self.non_deceptive_advocates = non_deceptive_advocates
        self.attention_type = attention_type
        self.semi_supervised = semi_supervised
        self.supervised_number = supervised_number
        self.supervised_rate = supervised_rate
        self.no_nine = no_nine
        self.custom_dataset = custom_dataset
        self.fancy_judge = fancy_judge
        self.advocate_capacity = advocate_capacity
        self.judge_capacity = judge_capacity
        self.active_label = active_label
        self.multilabel = multilabel


class RunManager:
    def __init__(self, device_list, save_path):
        self.device_names = device_list
        self.device_free = [True for _ in device_list]
        self.device_process = [None for _ in device_list]
        self.process_stack = []
        self.save_loc = save_path

    def add_job(self, run_config):
        load_path = os.path.join(self.save_loc, run_config['name'])+'.pkl'
        joblib.dump(run_config, load_path)
        self.process_stack.append(load_path)

    def done(self):
        free_device = sum(self.device_free)
        num_jobs = len(self.process_stack)
        print('------------------------------------')
        print('{}'.format(self.save_loc.split('/')[-1]))
        print('{}/{} devices free, {} tasks remain'.format(free_device, len(self.device_free), num_jobs))
        print('------------------------------------')
        return np.all(self.device_free) and len(self.process_stack) == 0

    def update_free(self):
        for i in range(len(self.device_process)):
            if self.device_process[i] is not None:
                if not self.device_process[i].is_alive():
                    print('job on device {} not alive'.format(self.device_names[i]))
                    self.device_process[i] = None
                    self.device_free[i] = True
            else:
                pass

    def launch_if_able(self):
        for i in range(len(self.device_names)):
            if self.device_free[i]:
                if len(self.process_stack) == 0:
                    print('No more jobs to launch')
                    break
                load_path = self.process_stack.pop()
                device = self.device_names[i]
                self.device_process[i] = mp.Process(target=run, args=(load_path, device))
                self.device_free[i] = False
                print('Launching job at {} on process {}'.format(load_path, device))
                self.device_process[i].start()

    def run_until_empty(self, sleep_interval=10):
        while not self.done():
            self.launch_if_able()
            time.sleep(sleep_interval)
            self.update_free()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.kaiming_normal_(m.weight)
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_dataset(config, type):
    if type == 'train':
        if config.custom_dataset:
            data = config.dataset(mode='trainval')
        elif config.data_type == 'svhn':
            train_transform = torchvision.transforms.Compose([torchvision.transforms.Grayscale(),
                                                              torchvision.transforms.Resize(size=[28, 28]),
                                                              torchvision.transforms.ToTensor()])
            data = config.dataset(root=config.data_dir, split='train', transform=train_transform)
            # TODO: could add in extra train split
        else:
            train_transform = torchvision.transforms.ToTensor()
            data = config.dataset(root=config.data_dir, transform=train_transform)
    elif type == 'test':
        if config.custom_dataset:
            data = config.dataset(mode='test')
        elif config.data_type == 'svhn':
            train_transform = torchvision.transforms.Compose([torchvision.transforms.Grayscale(),
                                                              torchvision.transforms.Resize(size=[28, 28]),
                                                              torchvision.transforms.ToTensor()])
            data = config.dataset(root=config.data_dir, split='test', transform=train_transform)
        else:
            test_transform = torchvision.transforms.ToTensor()
            data = config.dataset(root=config.data_dir, transform=test_transform, train=False)
    else:
        raise ValueError('No valid type given')
    return data


def get_model_and_optim(config, device, advocate_training, include_judge, include_advocates):
    if config.advocate_model:
        num_advocates = config.num_class
    else:
        num_advocates = 1
    model = AdvocacyNet(input_size=config.input_size,
                        num_class=config.num_class,
                        num_advocates=num_advocates,
                        shared_encoder=config.shared_encoder,
                        ts=config.mimic,
                        attention_type=config.attention_type,
                        fancy_judge=config.fancy_judge,
                        advocate_capacity=config.advocate_capacity,
                        judge_capacity=config.judge_capacity).to(device)
    judge_optimizer, advocate_optimizers = None, None
    if not advocate_training:
        if include_advocates:
            judge_params = model.parameters()
        else:
            judge_params = model.decision_module.parameters()
        judge_optimizer = torch.optim.Adam(params=judge_params,
                                           weight_decay=config.weight_decay, lr=config.learning_rate)
    else:
        advocate_optimizers = [torch.optim.Adam(params=model.advocates[i].parameters(),
                                                weight_decay=config.weight_decay,
                                                lr=config.learning_rate) for i in range(num_advocates)]
        if include_judge:
            judge_optimizer = torch.optim.Adam(params=model.decision_module.parameters(),
                                               weight_decay=config.weight_decay, lr=config.learning_rate)
    return advocate_optimizers, judge_optimizer, model


def run(load_path, device_id):
    torch.cuda.set_device(device_id)
    config_dict = joblib.load(load_path)
    config = RunConfig(**config_dict)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    print('seed: {}'.format(config.seed))
    device = torch.device('cuda:{}'.format(device_id))
    if config.multilabel:
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction='none')

    if config.train:
        print('training')
        data_tr = get_dataset(config=config, type='train')
        if config.active_label is not None:
            data_tr.set_active(config.active_label)
        (advocate_optimizers,
         judge_optimizer,
         model) = get_model_and_optim(config=config, device=device, advocate_training=config.advocate_training,
                                      include_judge=True, include_advocates=config.include_advocates)
        model_trainer = trainer.ExperimentTrainer(model=model,
                                                  optimizer=judge_optimizer,
                                                  advocate_optimizers=advocate_optimizers,
                                                  criterion=criterion,
                                                  name=config.name, experiment_name=config.experiment_name,
                                                  device=device,
                                                  load_saved=config.load_saved, overwrite=config.overwrite,
                                                  train_advocate=config.advocate_training, max_size=config.max_size,
                                                  label_noise_level=config.label_noise,
                                                  input_noise_level_train=config.input_noise_train,
                                                  input_noise_level_test=config.input_noise_test,
                                                  num_class=config.num_class, attention_reg=config.attention_reg,
                                                  honest_advocates=config.non_deceptive_advocates,
                                                  retrain=False, multilabel=config.multilabel)
        if config.semi_supervised:
            (supervised_ind,
             unsupervised_ind,
             validation_ind) = joblib.load('/data/ifox/advocacy_learning/data/{}_indices_{}.pkl'.format(config.data_type,
                                                                                                        config.supervised_number))
            improvements = model_trainer.semi_train_loop(epoch_lim=config.epoch_lim, data=data_tr,
                                                         supervised_ind=supervised_ind,
                                                         unsupervised_ind=unsupervised_ind,
                                                         validation_ind=validation_ind,
                                                         supervised_rate=config.supervised_rate,
                                                         early_stopping_lim=config.early_stopping_lim,
                                                         batch_size=config.batch_size,
                                                         validation_rate=config.validation_rate)
        else:
            improvements = model_trainer.train_loop(epoch_lim=config.epoch_lim, data=data_tr,
                                                    validation_percent=config.validation_percent,
                                                    early_stopping_lim=config.early_stopping_lim,
                                                    early_stopping_use_acc=config.early_stopping_use_acc,
                                                    batch_size=config.batch_size,
                                                    validation_rate=config.validation_rate, no_nine=config.no_nine)
        joblib.dump(improvements, '{}/{}_training_improvements.pkl'.format(config.save_path, config.name), compress=3)
        print('made model')
    if config.evaluate:
        print('evaluating')
        data_test = get_dataset(config, type='test')
        if config.active_label is not None:
            data_test.set_active(config.active_label)
        (_, _, model) = get_model_and_optim(config=config, device=device, advocate_training=config.advocate_training,
                                            include_judge=True, include_advocates=False)
        model.load_state_dict(torch.load('{}/{}/bsf.pt'.format(config.model_path, config.name)))
        model_tester = trainer.ExperimentTrainer(model=model, optimizer=None, advocate_optimizers=None,
                                                 criterion=criterion, name=config.name,
                                                 experiment_name=config.experiment_name, device=device,
                                                 load_saved=True, overwrite=False,
                                                 train_advocate=config.advocate_training,
                                                 label_noise_level=config.label_noise,
                                                 input_noise_level_train=config.input_noise_train,
                                                 input_noise_level_test=config.input_noise_test,
                                                 num_class=config.num_class, attention_reg=config.attention_reg,
                                                 honest_advocates=config.non_deceptive_advocates,
                                                 max_size=None, retrain=False, multilabel=config.multilabel)
        prediction_dict = model_tester.get_predictions(data=data_test, batch_size=config.batch_size)
        joblib.dump(prediction_dict, '{}/{}_pred_dict.pkl'.format(config.save_path, config.name), compress=3)
    if config.retrain:
        print('retraining')
        data_tr = get_dataset(config=config, type='train')
        if config.active_label is not None:
            data_tr.set_active(config.active_label)
        assert config.advocate_training
        if config.retrain_type == 'ete':
            retrain_advocate_training = False
            include_judge = True
            include_advocates = True
        elif config.retrain_type == 'judge':
            retrain_advocate_training = False
            include_judge = True
            include_advocates = False
        elif config.retrain_type == 'advocate':
            retrain_advocate_training = True
            include_judge = False
            include_advocates = True
        else:
            raise ValueError('No proper retrain_type given')
        (advocate_optimizers,
         judge_optimizer,
         model) = get_model_and_optim(config=config, device=device, advocate_training=retrain_advocate_training,
                                      include_judge=include_judge, include_advocates=include_advocates)
        model_trainer = trainer.ExperimentTrainer(model=model,
                                                  optimizer=judge_optimizer,
                                                  advocate_optimizers=advocate_optimizers,
                                                  criterion=criterion,
                                                  name=config.name, experiment_name=config.experiment_name,
                                                  device=device,
                                                  load_saved=config.load_saved, overwrite=config.overwrite,
                                                  train_advocate=retrain_advocate_training, max_size=config.max_size,
                                                  label_noise_level=config.label_noise,
                                                  input_noise_level_train=config.input_noise_train,
                                                  input_noise_level_test=config.input_noise_test,
                                                  num_class=config.num_class, retrain=True,
                                                  attention_reg=config.attention_reg,
                                                  honest_advocates=config.non_deceptive_advocates,
                                                  multilabel=config.multilabel)
        improvements = model_trainer.train_loop(epoch_lim=config.epoch_lim, data=data_tr,
                                                validation_percent=config.validation_percent,
                                                early_stopping_lim=config.early_stopping_lim,
                                                early_stopping_use_acc=config.early_stopping_use_acc,
                                                batch_size=config.batch_size,
                                                validation_rate=config.validation_rate, no_nine=config.no_nine)
        joblib.dump(improvements,
                    '{}/{}_training_improvements_retrain.pkl'.format(config.save_path, config.name),
                    compress=3)

        print('re-evaluating')
        data_test = get_dataset(config=config, type='test')
        if config.active_label is not None:
            data_test.set_active(config.active_label)
        (_, _, model) = get_model_and_optim(config=config, device=device, advocate_training=retrain_advocate_training,
                                            include_judge=True, include_advocates=False)
        model.load_state_dict(torch.load('{}/{}/bsf_retrain.pt'.format(config.model_path, config.name)))
        model_tester = trainer.ExperimentTrainer(model=model, optimizer=None, advocate_optimizers=None,
                                                 criterion=criterion, name=config.name,
                                                 experiment_name=config.experiment_name, device=device,
                                                 load_saved=True, overwrite=False,
                                                 train_advocate=config.advocate_training,
                                                 label_noise_level=config.label_noise,
                                                 input_noise_level_train=config.input_noise_train,
                                                 input_noise_level_test=config.input_noise_test,
                                                 num_class=config.num_class, retrain=True,
                                                 attention_reg=config.attention_reg,
                                                 honest_advocates=config.non_deceptive_advocates,
                                                 max_size=None, multilabel=config.multilabel)
        prediction_dict = model_tester.get_predictions(data=data_test, batch_size=config.batch_size)
        joblib.dump(prediction_dict, '{}/{}_pred_dict_retrain.pkl'.format(config.save_path, config.name), compress=3)
    print('finished')


def flatten_list(l):
    return np.concatenate([item for sublist in l for item in sublist])


if __name__ == '__main__':
    print('todo')
