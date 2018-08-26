#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-07-25 20:43:20
# @Author  : ${menzec} (${menzc@outlook.com})
# @Link    : http://example.org

import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import datetime
import matplotlib.pyplot as plt


def intToString(intnum, lenstring):
    strtem = str(intnum)
    if len(strtem) <= lenstring:
        strtem = (lenstring - len(strtem)) * '0' + strtem
    else:
        strtem = strtem[:lenstring]
    return strtem


def datestr(cur_datetime):
    return (intToString(cur_datetime.year, 4) + intToString(cur_datetime.month, 2) + intToString(cur_datetime.day, 2) +
            intToString(cur_datetime.hour, 2) + intToString(cur_datetime.minute, 2) + intToString(cur_datetime.second, 2))


# -----------------ready the dataset--------------------------
#pic_num = 0
def default_loader(path):
    # global pic_num
    # pic_num += 1
    # print('%5d is trained'%pic_num)
    return Image.open(path)


class MyDataset(Dataset):

    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        self.root = txt[:txt.rfind('\\') + 1]
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(self.root + fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

#-----------------create the Net and training------------------------


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()  # 256,256,3
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3,
                            stride=1, padding=1, bias=True),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(4)  # 128,128,64
        )
        self.conv2 = torch.nn.Sequential(  # 12
            torch.nn.Conv2d(64, 32, kernel_size=3,
                            stride=1, padding=1, bias=True),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 16, kernel_size=3,
                            stride=1, padding=1, bias=True),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(4096, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 21)
        )
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=0.01, momentum=0.9)

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        res = conv3_out.view(conv1_out.size(0), -1)
        out = self.dense(res)
        return out


########################################################################
# 4. Train the network
def train_net(net, train_dataset, rate):
    '''返回保存的每个阶段训练的损失函数值的一个 列表'''
    record_info = []
    for epoch in range(30):  # loop over the dataset multiple times
        running_loss = 0.0
        index1 = 0
        for i, data in enumerate(train_dataset, 0):
            # get the inputs
            inputs, labels = data
            # zero the parameter gradients
            net.optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = net.loss_func(outputs, labels)
            loss.backward()
            net.optimizer.step()
            # print statistics
            running_loss += loss.item()
            # print every## mini-batches
            index1 += 1
            if index1 > rate:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss))
                record_info.append(
                    [epoch + 1, i + 1, running_loss])
                running_loss = 0.0
                index1 = 0
    return record_info


# 保存模型
def save_model(net, root, record_info):
    '''返回保存的模型文件名'''
    trainFinsh = datetime.datetime.now()
    torch.save(net.state_dict(), root + 'state_dict' +
               datestr(trainFinsh) + '.pkl')  # 只保存参数，自己添加模型的结构
    torch.save(net, root + datestr(trainFinsh) + '.pkl')  # 保存参数和模型的结构信息
    filename = root + 'state_dict' + datestr(trainFinsh) + '.txt'
    record_fn = open(filename, 'w')
    print(net, file=record_fn)
    record_fn.write('Loss:\n')
    print(record_info, file=record_fn)
    del record_info[:]
    record_fn.close()
    return filename


def test_model(net, test_dataset):
    '''返回保存的测试结果的信息--列表'''
    eval_loss = 0.
    eval_acc = 0.
    test_info = []
    classifcation_results = {}
    for batch_x, batch_y in test_dataset:
        # batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        out = net(batch_x)
        # loss = loss_func(torch.from_numpy(out), batch_y)
        loss = net.loss_func(out, batch_y)
        eval_loss += loss.item()
        pred = torch.max(out, 1)[1]  # max(,1)返回每一行中最大的，并且返回索引
        classifcation_result = (pred == batch_y)
        num_correct = classifcation_result.sum()
        eval_acc += num_correct.item()
        batch_ynp = batch_y.numpy()
        for i, x in enumerate(batch_ynp):
            if classifcation_results.__contains__(str(x)):
                classifcation_results[
                    str(x)][0] += classifcation_result[i].item()
                classifcation_results[str(x)][1] += 1
            else:
                classifcation_results[
                    str(x)] = [classifcation_result[i].item(), 1]
    test_info.append(eval_acc / len(test_dataset))
    i = 0
    while i < len(classifcation_results):
        test_info.append(classifcation_results[str(i)][
                         0] / classifcation_results[str(i)][1])
        i = i + 1
    del classifcation_results
    return test_info


def main():

    start_time = datetime.datetime.now()
    print('开始时间：%s' % start_time)
    root = 'D:\\data\\UCMerced\\training\\'
    train_dataset = root + 'train.txt'
    test_dataset = root + 'test.txt'
    # 准备数据集
    batch_size = 20
    train_data = MyDataset(train_dataset, transform=transforms.ToTensor())
    test_data = MyDataset(test_dataset, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_data, batch_size=20, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=20)
    classes = ['agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings', 'chaparral',
               'denseresidential', 'forest', 'freeway', 'golfcourse', 'harbor', 'intersection',
               'mediumresidential', 'mobilehomepark', 'overpass', 'parkinglot', 'readme.txt', 'river',
               'runway', 'sparseresidential', 'storagetanks', 'tenniscourt']
    # 建立网络
    net = Net()
    print(net)
    # 训练和测试
    train_info = train_net(net, train_loader, 0.3 * len(test_data) / batch_size)
    print('Finished Training')
    modelname = save_model(net, root, train_info)
    print('Model save!')
    net.eval()
    test_info = test_model(net, test_loader)
    fn = open(modelname, 'a')
    fn.write('Test result:\n')
    print(test_info, file=fn)
    end_time = datetime.datetime.now()
    print('start_time:', start_time, '\nend_time', end_time,
          '\ncost time:', end_time - start_time, file=fn)
    #print(end_time,file = fn)
    print(test_info)
    fn.close()

main()
