import pandas as pd
from transformers import AutoModel, AutoTokenizer
import pickle
import os
from os import listdir
from os.path import isfile, join
import sys
import pickle
import pandas as pd
import json
from tqdm import trange
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
import re
import csv
from datetime import datetime
import sys
import numpy as np
import string
import argparse
import csv
import random
import pandas as pd
from tqdm import trange
import pickle
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import sys
import gensim
from random import randrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torchvision.transforms as transforms
from torch.autograd import Variable
import re
from torch.utils.data import DataLoader, Sampler
from torchvision import datasets, transforms
from nltk.stem import SnowballStemmer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_model = AutoModel.from_pretrained('bert-base-uncased')


def findMax(result):
    index = 0
    if result[1] - result[0] > 0:
        index = 1
    return index


def findAccuracy(solution, fact):
    correct = 0
    for index, answer in enumerate(solution):
        if fact[index] == answer:
            correct += 1
    return float(correct) / len(solution)


# 为什么要返回truth呢
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, dataLen, truth):
        self.dataLen = dataLen
        self.truth = truth.clone().detach()
        self.data = data.clone().detach()

    def __getitem__(self, index: int):
        return self.data[index].cuda(), self.truth[index].cuda()

    def __len__(self):
        return self.dataLen


class ModifiedBert(nn.Module):
    def __init__(self, size, bert):
        super().__init__()
        self.bert = bert
        self.fc1 = nn.Linear(size, 450)
        self.fc2 = nn.Linear(450, 15)
        self.fc3 = nn.Linear(15, 2)

    def forward(self, x):
        bert_output = self.bert(x[0:768])
        pooled_output = bert_output['pooler_output'].cpu().detach().numpy()
        hidden_shape = np.concatenate((pooled_output, x[768:]), axis=1)
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(hidden_shape)))))


def train_nn(trainSet, trainTruth, testSet, testTruth, qtype, lr, epoch):
    learning_rate = lr
    epochs = epoch
    batch_size = 256
    model_name = "nn_old_approach_50_" + qtype + "_best.pt"

    # 更改
    trainSet = torch.from_numpy(trainSet).double().cuda()
    trainTruth = torch.tensor(trainTruth).cuda()
    testSet = torch.from_numpy(testSet).double().cuda()
    testTruth = torch.tensor(testTruth).cuda()
    train_dataLen = len(trainSet)
    test_dataLen = len(testSet)
    trainAccuracies = []
    testAccuracies = []
    epochLists = []

    net = ModifiedBert(size=len(trainSet[0]), bert=bert_model).cuda()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()

    trainDataset = Dataset(data=trainSet, dataLen=train_dataLen, truth=trainTruth)
    trainLoader = DataLoader(trainDataset, batch_size=batch_size)
    testDataset = Dataset(data=testSet, dataLen=test_dataLen, truth=testTruth)
    testLoader = DataLoader(testDataset, batch_size=batch_size)

    for i in trange(epochs):
        for train, target in trainLoader:
            optimizer.zero_grad()
            score = net(train.float())
            loss = criterion(score, target.view(len(target)))
            loss.backward()
            optimizer.step()

        if i % 5 == 0 or i == epochs - 1:

            train_solution = []
            for j in trainSet:
                j = j.float()
                score = net(j).tolist()
                oneResult = findMax(score)
                train_solution.append(oneResult)
            train_accuracy = accuracy_score(trainTruth.cpu(), train_solution)
            train_f1 = f1_score(trainTruth.cpu(), train_solution)
            train_precision = precision_score(trainTruth.cpu(), train_solution)
            train_recall = recall_score(trainTruth.cpu(), train_solution)
            # print("Train accuracy:", train_accuracy, " at epoch：{}".format(i + 1))
            print("Train F1：", train_f1, " at epoch：{}".format(i + 1))
            print("Train precision", train_precision, " at epoch：{}".format(i + 1))
            print("Train recall", train_recall, " at epoch：{}\n".format(i + 1))

            test_solution = []
            for j in testSet:
                j = j.float()
                score = net(j).tolist()
                oneResult = findMax(score)
                test_solution.append(oneResult)
            # test_accuracy = findAccuracy(test_solution, testTruth)
            test_accuracy = accuracy_score(testTruth.cpu(), test_solution)
            test_f1 = f1_score(testTruth.cpu(), test_solution)
            test_precision = precision_score(testTruth.cpu(), test_solution)
            test_recall = recall_score(testTruth.cpu(), test_solution)
            # print("Test accuracy:", test_accuracy, " at epoch: {}".format(i + 1))
            print("Test F1：", test_f1, " at epoch: {}".format(i + 1))
            print("Test precision：", test_precision, " at epoch: {}".format(i + 1))
            print("Test recall：", test_recall, " at epoch: {}\n".format(i + 1))

            trainAccuracies.append(train_accuracy)
            testAccuracies.append(test_accuracy)
            epochLists.append(i)
            torch.save(net.state_dict(), model_name)

    print("Model stored to: {}".format(model_name))
    torch.save(net.state_dict(), model_name)
