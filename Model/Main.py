import os
import sys
import torch
import random
import warnings
import torch.onnx
import numpy as np
import torch.nn as nn
import scipy.io as io
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix
from torchsummary import summary
from thop import profile, clever_format
from Utils import prepareData
from TemporalAttention import TemporalAttention
from ChannelAttention import ChannelAttention

warnings.filterwarnings('ignore')

class Model(nn.Module):
    def __init__(self, lr):
        super(Model, self).__init__()
        self.TA_Block_1 = TemporalAttention()
        self.TA_Block_2 = TemporalAttention()
        self.TA_Block_3 = TemporalAttention()
        self.TA_Block_4 = TemporalAttention()
        self.TA_Block_5 = TemporalAttention()


        self.BN_ELU_1 = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ELU(),
            nn.Dropout(0.3)
        )
        self.BN_ELU_2 = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ELU(),
            nn.Dropout(0.3)
        )

        self.BN_ELU_3 = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ELU(),
            nn.Dropout(0.3)
        )
        self.BN_ELU_4 = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ELU(),
            nn.Dropout(0.3)
        )

        self.BN_ELU_5 = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ELU(),
            nn.Dropout(0.3)
        )


        self.CA_Block = ChannelAttention()

        self.Avg_pool_1 = nn.Sequential(
            nn.BatchNorm2d(5),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            #nn.Dropout(0.3)
        )
        self.DW_PW_Conv = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=5, kernel_size=(16, 1), groups=5, padding='same'),
            nn.Conv2d(in_channels=5, out_channels=1, kernel_size=(1, 1), padding='same'),
            nn.BatchNorm2d(1),
            nn.ELU()
        )


        self.classification_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=375, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=10),
            nn.ELU(),
            nn.Linear(in_features=10, out_features=2),
            nn.ELU()
        )

        self.criterion = nn.MSELoss()
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = []
        self.acc = []
        self.train_acc = []
        pass

    def forward(self, input):
        delta = input[:, 0:1, :, :]
        theta = input[:, 1:2, :, :]
        alpha = input[:, 2:3, :, :]
        beta = input[:, 3:4, :, :]
        gamma = input[:, 4:5, :, :]

        TA_Block_1 = self.TA_Block_1(delta)
        TA_Block_2 = self.TA_Block_2(theta)
        TA_Block_3 = self.TA_Block_3(alpha)
        TA_Block_4 = self.TA_Block_4(beta)
        TA_Block_5 = self.TA_Block_5(gamma)

        scale_TA_Block_1 = delta * TA_Block_1
        scale_TA_Block_2 = theta * TA_Block_2
        scale_TA_Block_3 = alpha * TA_Block_3
        scale_TA_Block_4 = beta * TA_Block_4
        scale_TA_Block_5 = gamma * TA_Block_5

        BN_ELU_1 = self.BN_ELU_1(scale_TA_Block_1)
        BN_ELU_2 = self.BN_ELU_2(scale_TA_Block_2)
        BN_ELU_3 = self.BN_ELU_3(scale_TA_Block_3)
        BN_ELU_4 = self.BN_ELU_4(scale_TA_Block_4)
        BN_ELU_5 = self.BN_ELU_5(scale_TA_Block_5)

        concat_TA_Blocks = torch.cat([BN_ELU_1,
                                      BN_ELU_2,
                                      BN_ELU_3,
                                      BN_ELU_4,
                                      BN_ELU_5
                                      ],
                                     dim=1)

        CA_Block = self.CA_Block(concat_TA_Blocks)
        scale_CA_Block = concat_TA_Blocks * CA_Block
        Avg_pool_1 = self.Avg_pool_1(scale_CA_Block)
        DW_PW_Conv = self.DW_PW_Conv(Avg_pool_1)

        output = self.classification_layer(DW_PW_Conv)

        return output
        pass

    def train_model(self, input, target):
        self.optimizer.zero_grad()
        output = self(input)
        loss = self.criterion(output, target)

        if not torch.isnan(loss):
            loss.backward()
            self.optimizer.step()
            self.loss.append(loss.item())
            self.train_acc.append((((output.argmax(dim=1) == target.argmax(dim=1)).sum()/len(output)).item())*100)
        pass

    def eval_model(self, testDataloader):
        with torch.no_grad():
            correct = []
            total = []
            self.outs = []
            self.real_targs = []

            for data, targets in testDataloader:
                data, targets = data.to('cuda'), targets.to('cuda')
                outputs = self(data).argmax(dim=1)
                targs = targets.argmax(dim=1)

                self.outs.extend(list((outputs.to('cpu').numpy())))
                self.real_targs.extend(list(targs.to('cpu').numpy()))

                correct.append((outputs == targs).sum().item())
                total.append(len(outputs))

                pass

            self.acc.append(sum(correct)*100/sum(total))

            pass
        pass
    pass


def parse_results(preds, targets):
    accuracy = accuracy_score(targets, preds)*100
    recall = recall_score(targets, preds)*100
    f_score = f1_score(targets, preds)*100
    kappa = cohen_kappa_score(targets, preds)*100
    confusionMatrix = confusion_matrix(targets, preds)

    results = {'Accuracy':accuracy,
               'Recall':recall,
               'F_score':f_score,
               'Kappa':kappa,
               'Confusion_matrix':confusionMatrix}

    return results

    pass


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32
    epochs = 40
    learning_rate = 0.0001
    fs = 100
    model = Model(lr=learning_rate).to(device)

    Data = prepareData('../Path to Dataset', fs=fs)
    results_filename = "../Path for Saving Results"


    idx = np.arange(len(Data))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, shuffle=True)
    testDataloader = DataLoader(Data, sampler=test_idx, batch_size=batch_size)

    results = {}
    model.eval_model(testDataloader)

    for epoch in range(epochs):
        random.shuffle(train_idx)
        trainDataloader = DataLoader(Data, sampler=train_idx, batch_size=batch_size)
        progress_bar = tqdm(trainDataloader, leave=False)

        for data, labels in progress_bar:
            data, labels = data.to(device), labels.to(device)
            model.train_model(data, labels)

            progress_bar.set_description(
                f'\033[92m({device} accelerated)\033[0m '
                f'epoch {epoch + 1}/{epochs} '
                f'loss:{model.loss[-1]:.6f}, '
                f'valid. accuracy:{model.acc[-1]:.4f}%'
            )
            pass

        if epoch == 0:
            plt.ion()
            plt.figure()


        model.eval_model(testDataloader)
        plt.plot(model.acc, 'r-o')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy(%)')
        plt.title('Validation Accuracy Trend')
        plt.ylim([model.acc[0], 100])
        plt.xlim([0, epochs])
        plt.xticks(list(range(0, epochs+1)))
        plt.grid()
        plt.pause(0.1)
        pass

    results['num_epochs'] = epochs
    results['loss'] = model.loss
    results['training_acc'] = model.train_acc
    results['validation_acc'] = model.acc
    results['Final_Results'] = parse_results(model.outs, model.real_targs)

    io.savemat(results_filename, results)

    plt.ioff()
    plt.show()
    pass
