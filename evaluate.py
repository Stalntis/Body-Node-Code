import torch
from sklearn.metrics import accuracy_score
from model import LeNet
from torchvision import transforms, datasets
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import time





if __name__ == '__main__':
    model = torch.load("./weight/best.pth")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




    BATCH_SIZE = 32
    data_transform = transforms.Compose(
        [
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_dataset = datasets.ImageFolder("./data/test/", transform=data_transform)  # 测试集数据
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                                 num_workers=2)  # 加载数据


    #classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10')  # 定义之后顺序是对的

    classes = test_dataset.classes
    n = len(classes)
    print(n)




    class_correct = [0.] * n  # for in in range(num_classes)
    class_total = [0.] * n
    y_test, y_pred = [], []
    X_test = []




    with torch.no_grad():
        for images, labels in test_loader:
            since = time.time()
            X_test.extend([_ for _ in images])
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.cpu()
            c = (predicted == labels).squeeze()
            # for i, label in enumerate(labels):
            #      class_correct[label] += c[i].item()
            #      class_total[label] += 1
            y_pred.extend(predicted.numpy())
            y_test.extend(labels.cpu().numpy())

    # for i in range(n):
    #     print(f"Acuracy of {classes[i]:5s}: {100 * class_correct[i] / class_total[i]:2.0f}%")





    ac = accuracy_score(y_test, y_pred,normalize=True, sample_weight=None)

    #rs = recall_score(y_test, y_pred, labels=None, pos_label=1, average=None, sample_weight=None)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=classes,digits=4)
    #roc_curve(y_test, y_score, pos_label=None, sample_weight=None, drop_intermediate=True)
    #print("Accuracy is :", ac)
    print(cm)
    print(cr)
    print("Accuracy is ",ac)

    print("Time: {:.2f}s".format((time.time() - since)))

