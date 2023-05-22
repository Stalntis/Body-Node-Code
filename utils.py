import torch
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import csv
import codecs
import pandas as pd
from datetime import datetime




def train_and_val(epochs, model, train_loader, len_train,val_loader, len_val,criterion, optimizer,device):

    torch.cuda.empty_cache()
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    best_acc = 0

    model.to(device)
    fit_time = time.time()
    for e in range(epochs):
        since = time.time()
        running_loss = 0
        training_acc = 0
        with tqdm(total=len(train_loader)) as pbar:
            for image, label in train_loader:

                model.train()
                optimizer.zero_grad()
                image = image.to(device)
                label = label.to(device)
                # forward
                output = model(image)
                loss = criterion(output, label)
                predict_t = torch.max(output, dim=1)[1]

                # backward
                loss.backward()
                optimizer.step()  # update weight

                running_loss += loss.item()
                training_acc += torch.eq(predict_t, label).sum().item()
                pbar.update(1)

        model.eval()
        val_losses = 0
        validation_acc = 0
        # validation loop
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as pb:
                for image, label in val_loader:
                    image = image.to(device)
                    label = label.to(device)
                    output = model(image)

                    # loss
                    loss = criterion(output, label)
                    predict_v = torch.max(output, dim=1)[1]

                    val_losses += loss.item()
                    validation_acc += torch.eq(predict_v, label).sum().item()
                    pb.update(1)

            # calculatio mean for each batch
            train_loss.append(running_loss / len_train)
            val_loss.append(val_losses / len_val)

            train_acc.append(training_acc / len_train)
            val_acc.append(validation_acc / len_val)




            torch.save(model, "./weight/last.pth")
            if best_acc <(validation_acc / len_val):
                torch.save(model, "./weight/best.pth")


            print("Epoch:{}/{}..".format(e + 1, epochs),
                  "Train Acc: {:.6f}..".format(training_acc / len_train),
                  "Val Acc: {:.6f}..".format(validation_acc / len_val),
                  "Train Loss: {:.6f}..".format(running_loss / len_train),
                  "Val Loss: {:.6f}..".format(val_losses / len_val),
                  "Time: {:.6f}s".format((time.time() - since)))

            Time = "%s"%datetime.now()#获取当前时间

            Epoch = "Epoch[%d]" % e

            t_loss1 = (running_loss / len_train)
            t_loss = "%f"%t_loss1

            t_acc1 = (training_acc / len_train)
            t_acc = "%g"%t_acc1
            # 将数据保存在一维列表
            list1 = [Time, Epoch,t_acc,t_loss]
            # 由于DataFrame是Pandas库中的一种数据结构，它类似excel，是一种二维表，所以需要将list以二维列表的形式转化为DataFrame
            data1 = pd.DataFrame([list1])
            data1.to_csv('./weight/train_acc.csv', mode='a', header=False,
                        index=False)  # mode设为a,就可以向csv文件追加数据了

            v_loss1 = (val_losses / len_val)
            v_loss = "%f" % v_loss1

            v_acc1 = (validation_acc / len_val)
            v_acc = "%g" % v_acc1
            # 将数据保存在一维列表
            list2 = [Time, Epoch, v_acc, v_loss]
            # 由于DataFrame是Pandas库中的一种数据结构，它类似excel，是一种二维表，所以需要将list以二维列表的形式转化为DataFrame
            data2 = pd.DataFrame([list2])
            data2.to_csv('./weight/validation_acc.csv', mode='a', header=False,
                         index=False)  # mode设为a,就可以向csv文件追加数据了




    history = {'train_loss': train_loss, 'val_loss': val_loss ,'train_acc': train_acc, 'val_acc': val_acc}
    print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))





    return history


def plot_loss(x, history):
    plt.plot(x, history['train_loss'], label='train', marker='')
    plt.plot(x, history['val_loss'], label='test', marker='')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(), plt.grid()
    plt.savefig('./weight/loss.png')
    # plt.show()


def plot_acc(x, history):
    plt.plot(x, history['train_acc'], label='train', marker='')
    plt.plot(x, history['val_acc'], label='tset', marker='')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(), plt.grid()
    plt.savefig('./weight/acc.png')
   # plt.show()

# def plot_acc(x, history):
#     plt.plot(x, history['train_acc'], label='training accuracy',color='#993432', marker='')
#     plt.plot(x, history['val_acc'], label='validation accuracy',color='#4465a7', marker='')
#     plt.plot(x, history['train_loss'], label='training loss',color='#d39775', marker='')
#     plt.plot(x, history['val_loss'], label='validation loss',color='#b5cce1', marker='')
#     plt.title('Accuracy and Loss')
#     plt.ylabel('Accuracy')
#     plt.xlabel('Epochs')
#     plt.legend(), plt.grid()
#     plt.savefig('./weight/1.png')
#     # plt.show()

