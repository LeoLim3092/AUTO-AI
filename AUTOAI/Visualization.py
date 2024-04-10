import matplotlib.pyplot as plt
import matplotlib
import os
import seaborn as sb
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns

cwd = os.getcwd()
save_dir = cwd + "/Save_Models"
save_Figures = cwd + "/Figures"
fontsz = 20

def plot_ROC_cruve(fpr, tpr, roc_auc, title=None, n_classes=None, plot_class=None, savepth=None):

    """
    plot binary or multi-classes ROC, for plotting multi-classes (n_classes>=3), use plot_class
    to indicate which class ROC (oneVsRest) or plot "macro" average
    :param fpr: (list, dict) False positive rate
    :param tpr: (list, dict) True positive rate
    :param roc_auc: (list, dict) roc_auc
    :param title: (str) figures title
    :param n_classes: (int) number of classes
    :param plot_class: (int,"macro", "micro") which classes to plot or using macro/micro average for multiclasses
    :return: ROC figures save in "../Output/Figures"
    """
    font = {'weight' : 'bold',
            'size'   : 64}
    matplotlib.rc("font", **font)
    
    lw = 3
    plt.style.use("ggplot")
    plt.figure(figsize=(10, 10), dpi=200, facecolor='w', edgecolor='k')

    if isinstance(fpr, list):
        print("input is list")
        plt.plot(fpr, tpr, color='darkorange', label='RandomForest (area = %0.2f)' % roc_auc)

    elif isinstance(fpr, dict):
        print("input is dict")
        for fpr_values, tpr_values, roc_auc_values, methods in zip(
                fpr.values(), tpr.values(), roc_auc.values(), fpr.keys()):
            if n_classes == 2:
                plt.plot(fpr_values, tpr_values, label=str(methods) + " : %0.2f" % roc_auc_values)
            else:
                if plot_class == "macro":
                    plt.plot(fpr_values["macro"], tpr_values["macro"],
                             label=str(methods) + " : %0.2f" % roc_auc_values["macro"])

                elif plot_class == "micro":
                    plt.plot(fpr_values["micro"], tpr_values["micro"],
                             label=str(methods) + " : %0.2f" % roc_auc_values["micro"])

                else:
                    plt.plot(fpr_values[0][plot_class-1], tpr_values[0][plot_class-1],
                             label=str(methods) + " : %0.2f" % roc_auc_values[0][plot_class-1])
    else:
        if n_classes == 2:
            plt.plot(fpr, tpr, color='darkorange', label='23 features (area = %0.2f)' % roc_auc)
        else:
            plt.plot(fpr[plot_class-1], tpr[plot_class-1],
                     color='darkorange', label='23 features (area = %0.2f)' % roc_auc[plot_class-1])

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('1-Specificity', fontsize=fontsz, fontweight="bold")
    plt.ylabel('Sensitivity', fontsize=fontsz, fontweight="bold")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.09),
          fancybox=True, shadow=True, ncol=1, fontsize=fontsz-2)

    plt.title(title, fontsize=fontsz)
    
    if savepth:
        plt.savefig(savepth, bbox_inches="tight")
    else:
        plt.show()
    
    plt.close()
    


def plot_pecision_recall_curve(precision, recall, average_precision, title=None, n_classes=None, plot_class=None):
    """
    plot binary or multi-classes Precision-Recall Curve
    :param precision: (list, dict) list or dict of precision
    :param recall: (list, dict) list or dict of recall
    :param average_precision: (list, dict) average precision(auc)
    :param title: (str) title and save name of the figures
    :param n_classes: (int) number of class
    :param plot_class: (int, "micro") plotting class
    :return:
    """

    font = {'weight': 'bold',
            'size': 14}

    plt.style.use("ggplot")
    matplotlib.rc('font', **font)
    plt.figure(figsize=(10, 10), dpi=100, facecolor='w', edgecolor='k')

    f_scores = np.linspace(0.1, 0.9, num=9)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    if isinstance(precision, list):
        print("input is list")
        if n_classes == 2:
            plt.step(recall[0], precision[0], color='darkorange', label="RandomForest : %0.2f " % average_precision[0],
                     where="mid")
        else:
            plt.step(recall[plot_class-1], precision[plot_class-1], color='darkorange',
                     label="RandomForest AP: %0.2f " % average_precision[plot_class-1],
                     where="mid")

    elif isinstance(precision, dict):
        print("input is dict")
        for p_values, r_values, ap_values, methods in zip(
                precision.values(), recall.values(), average_precision.values(), precision.keys()):
            if n_classes == 2:
                plt.step(r_values[0], p_values[0], label=str(methods) + " : %0.2f " % ap_values[0], where="mid")
            else:
                if plot_class == "micro":
                    plt.step(r_values[0]["micro"], p_values[0]["micro"],
                             label=str(methods) + " : %0.2f " % ap_values[0]["micro"], where="mid")

                # elif plot_class == "macro":
                #    plt.step(r_values[0][plot_class - 1], p_values[0][plot_class - 1],
                #             label=str(methods) + " : %0.2f " % ap_values[0][plot_class - 1], where="mid")

                else:
                    plt.step(r_values[0][plot_class-1], p_values[0][plot_class-1],
                             label=str(methods) + " : %0.2f " % ap_values[0][plot_class-1], where="mid")
    else:
        print("input is None")

    plt.xlabel('Recall', fontsize=18, fontweight="bold")
    plt.ylabel('Precision', fontsize=18, fontweight="bold")
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.legend(bbox_to_anchor=(0., -.302, 0., .102), loc=3, ncol=2, borderaxespad=0.)

    if title is not None:
        plt.title('AUPRC%s\n' % (title), fontsize=26)
        plt.savefig("../Output/Figures/%s_PRC.tif" % title, bbox_inches="tight")

    else:
        plt.title('PR Curve')
        plt.show()


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues, ax=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    if ax is None:
        fig, ax = plt.subplots()

    im = ax.imshow(cm, cmap=cmap)

    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xlim=(-0.5, cm.shape[1]-0.5),
           ylim=(cm.shape[0]-0.5, -0.5),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           xlabel='Predicted label',
           ylabel='True label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.grid(False)
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    return ax, cm


def plot_feature_selection(sfs, features_name, title=""):
    order_ls = []
    score_ls = []

    for k, v in sfs.subsets_.items():
        if k <= len(sfs.k_feature_idx_):
            order_ls.append(list(set(v["feature_idx"]) - set(order_ls))[0])
            score_ls.append(v["avg_score"])

    all_result_ls = []
    all_idx_ls = []

    for sfs in sfs.all_results_ls:
        for idxs in sfs["idx"]:
            all_idx_ls.append(list(idxs))
        for scores in sfs["avg_scores"]:
            all_result_ls.append(scores)

    best_features = features_name[order_ls]

    fig, ax = plt.subplots((2), figsize=(15, 15))
    print(best_features, score_ls, np.arange(0, len(best_features)))

    sns.barplot(np.arange(0, len(best_features)), np.array(score_ls) * 100, color="b", ax=ax[0])
    ax[0].set_ylim([0, 110])
    ax[0].set_xticklabels([bf if i == 0 else "... + " + bf for i, bf in enumerate(list(best_features))],
                          rotation=90, fontsize=18)

    for i, score in enumerate(score_ls):
        ax[0].text(i - 0.2, score * 100 + 2, "%.2f" % np.dot(score, 100), fontsize=14)

    ax[0].set_ylabel("Accuracy, %")
    ax[0].set_title("{} Sequential Forward Selection Best Features".format(title))

    ax[1].plot(np.array(all_result_ls) * 100)
    ax[1].set_ylim([0, 100])
    # ax[1].set_xticks([], [])
    ax[1].set_ylabel("Accuracy, %")
    ax[1].set_title("{} Sequential Forward Selection All results".format(title))

    inax = ax[1].inset_axes([0.3, 0.1, 0.7, 0.7])
    inax.plot(np.array(all_result_ls[:len(features_name)]) * 100)
    inax.set_ylim([0, 100])
    inax.set_xticks(np.arange(0, len(features_name)))
    inax.set_xticklabels(features_name, rotation=90, fontsize=14)

    ax[1].indicate_inset_zoom(inax, lw=5)

    plt.subplots_adjust(hspace=1.2)
    plt.show()

    return all_result_ls, all_idx_ls
