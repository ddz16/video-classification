from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import label_binarize
import os


def compute_metrics(true_label, pred_label):
    '''
        常用指标: 精度, 查准率, 召回率, F1-Score
    '''
    # 精度，准确率， 预测正确的占所有样本种的比例
    accuracy = accuracy_score(true_label, pred_label)
    print("精度: ", accuracy)

    # 查准率P（准确率），precision(查准率)=TP/(TP+FP)

    precision = precision_score(true_label, pred_label, labels=None, pos_label=1, average='macro')  # 'micro', 'macro', 'weighted'
    print("查准率P: ", precision)

    # 查全率R（召回率），原本为对的，预测正确的比例；recall(查全率)=TP/(TP+FN)
    recall = recall_score(true_label, pred_label, average='macro')  # 'micro', 'macro', 'weighted'
    print("召回率: ", recall)

    # F1-Score
    f1 = f1_score(true_label, pred_label, average='macro')  # 'micro', 'macro', 'weighted'
    print("F1 Score: ", f1)


def plot_confusion_matrix(true_label, pred_label, label_dict, save_path='results/'):
    """
    示例用法: 
        true_label = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        pred_label = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        label_dict = {'A': 0, 'B': 1, 'C': 2}
        save_path = 'confusion_matrix.pdf'
        plot_confusion_matrix(true_label, pred_label, label_dict, save_path)
    """
    # 计算混淆矩阵
    cm = confusion_matrix(true_label, pred_label)
    # 获取类别列表
    labels = list(label_dict.keys())
    # 绘制混淆矩阵
    plt.figure(figsize=(len(labels), len(labels)))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.annotate(str(cm[i][j]), xy=(j, i),
                         horizontalalignment='center',
                         verticalalignment='center')
        
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    # 保存混淆矩阵到pdf文件
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    plt.savefig(os.path.join(save_path, 'confusion.pdf'))
    plt.close()


def plot_roc_curve(true_label, pred_label, label_dict, save_path='results/'):
    """
    示例用法:
        true_label = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        pred_label : B*C
        label_dict = {'A': 0, 'B': 1, 'C': 2}
        plot_roc_curve(true_label, pred_label, label_dict)
    """
    fig, ax = plt.subplots()

    new_label_dict = {value: key for key, value in label_dict.items()}

    # 遍历每个类别
    for class_label in new_label_dict:
        # 将真实标签和预测标签转换为二进制形式，以便计算ROC曲线
        true_binary = [1 if label == class_label else 0 for label in true_label]
        pred_binary = pred_label[:, class_label].tolist()

        # 计算ROC曲线的假阳率、真阳率和阈值
        fpr, tpr, thresholds = roc_curve(true_binary, pred_binary)

        # 计算AUC值
        roc_auc = auc(fpr, tpr)

        # 绘制ROC曲线
        ax.plot(fpr, tpr, label=f"{new_label_dict[class_label]} (AUC = {roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], 'k--')

    # 添加图例
    ax.legend(loc="lower right")

    # 设置x轴和y轴标签
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    # 设置图表标题
    ax.set_title("Multi-class receiver operating characteristic")

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    plt.savefig(os.path.join(save_path, 'roc.pdf'))
    plt.close()
