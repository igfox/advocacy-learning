"""
Implements training schemes with logging
"""
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, sampler
from tensorboardX import SummaryWriter
from tqdm import tqdm

DATA_DIR = '/insert/data/dir/here'


class ExperimentTrainer:
    """
    Class which implements simple training scheme. Defaults to an autoencoder loss.
    """
    def __init__(self, model, optimizer, criterion, name, experiment_name, device, num_class,
                 advocate_training, honest_advocates, advocate_optimizers, max_size, attention_reg):
        self.model = model
        self.criterion = criterion
        if advocate_training:
            self.judge_optimizer = optimizer
            self.advocate_optimizers = advocate_optimizers
        else:
            self.optimizer = optimizer
        self.advocate_training = advocate_training
        self.honest_advocates = honest_advocates
        self.num_class = num_class
        self.max_size = max_size
        self.attention_reg = attention_reg
        self.name = name
        self.device = device
        self.model_dir = '{}/saved_models/{}/{}'.format(DATA_DIR, experiment_name, self.name)
        self.log_dir = '{}/log/{}/{}'.format(DATA_DIR, experiment_name, self.name)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def train_loop(self, epoch_lim, data, validation_percent, early_stopping_lim, batch_size, validation_rate):
        if early_stopping_lim is None:
            early_stopping_lim = epoch_lim
        if self.max_size is None:
            train_sampler = sampler.SubsetRandomSampler(list(range(0, int(len(data) * (1 - validation_percent)))))
        else:
            train_sampler = sampler.SubsetRandomSampler(list(range(0, self.max_size)))
        validation_sampler = sampler.SubsetRandomSampler(list(range(int(len(data) * (1 - validation_percent)),
                                                                        len(data))))
        data_train = DataLoader(data, batch_size=batch_size, sampler=train_sampler, drop_last=True)
        data_validation = DataLoader(data, batch_size=batch_size, sampler=validation_sampler)
        step = 0

        bsf_loss = np.inf
        epochs_without_improvement = 0
        improvements = []
        for epoch in range(epoch_lim):
            if epochs_without_improvement > early_stopping_lim:
                print('Exceeded early stopping limit, stopping')
                break
            if epoch % validation_rate == 0:
                validation_loss = self.validation(data_validation, step)
                (bsf_loss,
                 epochs_without_improvement,
                 improvements) = self.manage_early_stopping(bsf_loss, early_stopping_lim,
                                                            epochs_without_improvement,
                                                            validation_loss, validation_rate, improvements)
            running_train_loss = 0
            for dat, label in tqdm(data_train):
                step += 1
                x = dat.to(self.device)
                y = label.to(self.device)
                pred, attention = self.model(x=x)
                running_train_loss = self.train_update(pred=pred, attention=attention,
                                                       running_train_loss=running_train_loss, y=y)
            train_size = len(data_train) * len(dat)
            running_train_loss = running_train_loss / train_size
            self.writer.add_scalar(tag='train_loss',
                                   scalar_value=running_train_loss,
                                   global_step=step)
            torch.save(self.model.state_dict(), '{}/final.pt'.format(self.model_dir))
        return improvements

    def train_update(self, pred, attention, running_train_loss, y):
        if self.advocate_training:
            return self.train_update_advocate(pred, attention, running_train_loss, y)
        else:
            return self.train_update_attention(pred, attention, running_train_loss, y)

    def train_update_advocate(self, pred, attention, running_train_loss, y):
        judge_loss = self.criterion(pred, y).sum()
        running_train_loss += judge_loss.item()
        judge_loss.backward(retain_graph=True)
        self.judge_optimizer.step()
        self.judge_optimizer.zero_grad()
        for i in range(len(self.advocate_optimizers)):
            advocate_y = torch.ones_like(y, dtype=torch.long).to(self.device) * i
            advocate_loss = self.criterion(pred, advocate_y)
            if self.honest_advocates:
                advocate_mask = (y == advocate_y).to(dtype=torch.float)
            else:
                advocate_mask = torch.ones_like(advocate_loss)
            advocate_loss = (advocate_loss * advocate_mask).sum()
            total_loss = advocate_loss + self.attention_reg * attention.mean()
            total_loss.backward(retain_graph=True)
            self.advocate_optimizers[i].step()
            self.advocate_optimizers[i].zero_grad()
        return running_train_loss

    def train_update_attention(self, pred, attention, running_train_loss, y):
        loss = self.criterion(pred, y).sum()
        total_loss = loss + self.attention_reg * attention.mean()
        running_train_loss += total_loss.item()
        total_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return running_train_loss

    def manage_early_stopping(self, bsf_loss, early_stopping_lim, epochs_without_improvement, valid_loss,
                              validation_rate, improvements):
        if valid_loss < bsf_loss:
            print('improved validation loss from {:.3f} to {:.3f}'.format(bsf_loss, valid_loss))
            bsf_loss = valid_loss
            improvements.append(epochs_without_improvement)
            epochs_without_improvement = 0
            print('saving to {}/bsf.pt'.format(self.model_dir))
            torch.save(self.model.state_dict(), '{}/bsf.pt'.format(self.model_dir))
        else:
            epochs_without_improvement += validation_rate
            print('Validation loss of {} did not improve on {}'.format(valid_loss, bsf_loss))
            print('Early stopping at {}/{}'.format(epochs_without_improvement, early_stopping_lim))
        return bsf_loss, epochs_without_improvement, improvements

    def validation(self, dataloader, step):
        with torch.no_grad():
            self.model.eval()
            running_valid_loss = 0
            running_correct = 0
            running_size = 0
            for dat, label in dataloader:
                x = dat.to(self.device)
                y = label.to(self.device)
                pred, attention = self.model(x=x)
                valid_loss = self.criterion(pred, y).sum()
                running_valid_loss += valid_loss.item()
                running_correct += (torch.argmax(pred, dim=1) == y).sum().item()
                running_size += pred.shape[0]
            validation_size = len(dataloader) * len(dat)
            running_valid_loss = running_valid_loss / validation_size
            running_accuracy = running_correct / running_size
            print('validation loss: {:.3f}, {:.3f}'.format(running_valid_loss, self.attention_reg*attention.mean()))
            print('validation Acc: {:.3f}'.format(running_accuracy))
            self.writer.add_scalar(tag='valid_total_loss',
                                   scalar_value=running_valid_loss,
                                   global_step=step)
            self.model.train()
        return running_valid_loss

    def get_predictions(self, data, batch_size):
        dataloader = DataLoader(data, batch_size=batch_size)
        with torch.no_grad():
            self.model.eval()
            y = []
            pred = []
            for dat, label in dataloader:
                x = dat.to(self.device)
                y.append(label.numpy())
                p, a = self.model(x=x)
                pred.append(p.cpu().numpy())
        return pred, y
