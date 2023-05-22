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






    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns


    def make_confusion_matrix(cf,
                              group_names=None,
                              categories='auto',
                              count=True,
                              percent=True,
                              cbar=True,
                              xyticks=True,
                              xyplotlabels=True,
                              sum_stats=True,
                              figsize=None,
                              cmap='Blues',
                              title=None):
        # CODE TO GENERATE TEXT INSIDE EACH SQUARE
        blanks = ['' for i in range(cf.size)]

        if group_names and len(group_names) == cf.size:
            group_labels = ["{}\n".format(value) for value in group_names]
        else:
            group_labels = blanks

        if count:
            group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
        else:
            group_counts = blanks

        if percent:
            group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
        else:
            group_percentages = blanks

        box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
        box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

        # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
        if sum_stats:
            # Accuracy is sum of diagonal divided by total observations
            accuracy = np.trace(cf) / float(np.sum(cf))

            # if it is a binary confusion matrix, show some more stats
            if len(cf) == 2:
                # Metrics for Binary Confusion Matrices
                precision = cf[1, 1] / sum(cf[:, 1])
                recall = cf[1, 1] / sum(cf[1, :])
                f1_score = 2 * precision * recall / (precision + recall)
                stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                    accuracy, precision, recall, f1_score)
            else:
                stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
        else:
            stats_text = ""

        # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
        if figsize == None:
            # Get default figure size if not set
            figsize = plt.rcParams.get('figure.figsize')

        if xyticks == False:
            # Do not show categories if xyticks is False
            categories = False

        # MAKE THE HEATMAP VISUALIZATION
        sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)

        if xyplotlabels:
            plt.ylabel('True label')
            plt.xlabel('Predicted label' + stats_text)
        else:
            plt.xlabel(stats_text)

        if title:
            plt.title(title)


    #roc
    from sklearn.metrics import precision_score, f1_score
    from sklearn.preprocessing import label_binarize

    nb_classes = n
    # Binarize the output
    y_testss = label_binarize(y_test, classes=[i for i in range(nb_classes)])
    pred_digitss = label_binarize(y_pred, classes=[i for i in range(nb_classes)])

    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc
    from scipy import interp
    from itertools import cycle
    import numpy as np

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(nb_classes):
        fpr[i], tpr[i], _ = roc_curve(y_testss[:, i], pred_digitss[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_testss.ravel(), pred_digitss.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nb_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(nb_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= nb_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    #
    plt.figure(figsize=(9, 9))
    # plt.subplot(1, 2, 1)

    from sklearn.metrics import confusion_matrix  # 导入混淆矩阵函数
    import matplotlib.pyplot as plt

    # 混淆矩阵图1
    confusion = confusion_matrix(y_pred, y_test)
    make_confusion_matrix(confusion,
                          group_names=None,
                          categories='auto',
                          count=False,
                          percent=False,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None)

    plt.subplot(1, 1, 1)
    plt.plot(fpr["micro"], tpr["micro"],
             label='ROC curve (area = {0:0.4f})'
                   ''.format(roc_auc["micro"]),
             color='g', linewidth = 2)
    #
    '''
    plt.plot(fpr["macro"], tpr["macro"],
      label='macro-average ROC curve (area = {0:0.2f})'
      ''.format(roc_auc["macro"]),
      color='r', linewidth=2)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(nb_classes), colors):
     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
      label='ROC curve of class {0} (area = {1:0.2f})'
      ''.format(i, roc_auc[i]))
    '''

    plt.plot([0, 1], [0, 1], 'k', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig('./weight/ROC curve.png')
    # plt.show()






    #heatmap

    labels = pd.DataFrame(cm).applymap(lambda v: f"{v}" if v != 0 else f"")
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=labels, fmt='s', xticklabels=classes, yticklabels=classes, linewidths=0.1,cmap = 'Blues')


    plt.ylabel('True Label')
    plt.xlabel('Predicted Label\nAccuracy is :{:.6f}'.format(ac))
    plt.tight_layout()
    plt.savefig('./weight/heatmap.png')
    # plt.show()












