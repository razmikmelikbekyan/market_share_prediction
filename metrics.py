from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.collections import QuadMesh
from scipy import stats
from sklearn.metrics import (confusion_matrix, roc_curve, auc, average_precision_score,
                             precision_recall_curve, mean_squared_error, explained_variance_score,
                             r2_score)


def bernoulli_conf_interval(p: float, n: int, confidence: float):
    """
    Calculates confidence interval for n i.i.d. bernoulli(p) random variables.
    It is using CLT: we approximating the average of n i.i.d. bernoulli(p) distributed random
    variables with normal distribution.

        confidence interval: p ± z * (p(1-p) / n)^(1/2)
        alpha = 1 - confidence
        z = 1 - alpha / 2 quantile for standard normal distribution


    :param p: the probability of 1
    :param n: number of i.i.d. bernoulli(p) random variables
    :param confidence: confidence value (0 < confidence < 1)
    :return: tuple, confidence interval
    """
    alpha = 1 - confidence  # target error rate
    z = stats.norm.ppf(1 - alpha / 2)  # 1-alpha/2 - quantile of a standard normal distribution
    se = z * np.sqrt(p * (1 - p) / n)  # standard error
    return p - se, p + se


def confusion_matrix_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Calculates confusion matrix related metrics and plots confusion matrix.
    :param y_true: true label of the data, with shape (n_samples,), binary array
    :param y_pred: prediction of the data, with shape (n_samples,), binary array

    :return: dict of metrics
    """
    metrics = {}
    cm = confusion_matrix(y_true, y_pred)

    # true negatives (TN): we predicted N, and they don't have the disease (actual N)
    # false positives (FP): we predicted Y, but they don't have the disease (actual N)
    # false negatives (FN): we predicted N, but they do have the disease (actual Y)
    # true positives (TP): we predicted Y and they do have the disease (actual Y)
    tn, fp, fn, tp = cm.ravel()
    metrics.update({'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp})

    # normalzing matrix - getting rates
    cm_norm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])

    # tnr (specificity): probability that a test result will be negative when the disease is not present (tn / actual_N)
    # fpr: probability that a test result will be positive when the disease is not present (fp / actual_N)
    # fnr: probability that a test result will be negative when the disease is present (fn / actual_Y)
    # tpr (recall): probability that a test result will be positive when the disease is present (tp / actual_Y)
    tnr, fpr, fnr, tpr = cm_norm.ravel()
    metrics.update({'tpr': tpr, 'fnr': fnr, 'tnr': tnr, 'fpr': fpr})

    # ppv (precision): probability that the disease is present when the test is positive (tp / (tp + fp))
    # npv: probability that the disease is not present when the test is  negative (tn / (tn + fn))
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    ppv_conf_interval = bernoulli_conf_interval(ppv, tp + fn, 0.95)
    npv_conf_interval = bernoulli_conf_interval(npv, tn + fn, 0.95)
    metrics.update(
        {'ppv': ppv, 'npv': npv, 'PPV 95% CI': ppv_conf_interval, 'NPV 95% CI': npv_conf_interval})

    # Overall, how often is the classifier correct?
    accuracy = (tp + tn) / (tn + fp + fn + tp)
    metrics.update({'accuracy': accuracy})

    # The weighted average of recall and precision.
    f_score = 2 * tpr * ppv / (tpr + ppv)
    metrics.update({'f_score': f_score, 'cm': cm_norm})

    return metrics


def plot_confusion_matrix(confusion_matrix_metrics: Dict,
                          label: str = '',
                          class_labels: List[str] = ('N', 'Y'),
                          figsize: Tuple = (5, 5),
                          fsize: int = 12,
                          image_path: str = None):
    """
    Plots confusion matrix
    :param confusion_matrix_metrics: the output of "confusion_matrix_metrics" function
    :param class_labels: list of labels, first should be the name of negative classes
    :param figsize: the tuple, specifying figure size of plot
    :param fsize: font size for plot
    :param image_path: the path were image will be saved
    """

    cm_norm = confusion_matrix_metrics['cm']

    tn = confusion_matrix_metrics['tn']
    tnr = confusion_matrix_metrics['tnr']
    npv = confusion_matrix_metrics['npv']

    fp = confusion_matrix_metrics['fp']
    fpr = confusion_matrix_metrics['fpr']

    fn = confusion_matrix_metrics['fn']
    fnr = confusion_matrix_metrics['fnr']

    tp = confusion_matrix_metrics['tp']
    tpr = confusion_matrix_metrics['tpr']
    ppv = confusion_matrix_metrics['ppv']

    accuracy = confusion_matrix_metrics['accuracy']
    f_score = confusion_matrix_metrics['f_score']

    # annotation for heatmap
    annot = np.empty_like(cm_norm).astype(str)
    annot[0, 0] = 'TN={} \n\nTNR={:.2f}% \n\nNPV={:.2f}%'.format(tn, tnr * 100, npv * 100)
    annot[0, 1] = 'FP={} \n\nFPR={:.2f}%'.format(fp, fpr * 100)
    annot[1, 0] = 'FN={} \n\nFNR={:.2f}%'.format(fn, fnr * 100)
    annot[1, 1] = 'TP={} \n\nTPR={:.2f}% \n\nPPV={:.2f}%'.format(tp, tpr * 100, ppv * 100)

    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(pd.DataFrame(cm_norm, index=class_labels, columns=class_labels),
                     annot=annot,
                     annot_kws={"size": fsize, 'color': 'w', 'fontstyle': 'oblique'},
                     linewidths=0.1,
                     ax=ax,
                     cbar=False,
                     linecolor='w',
                     fmt='')

    # face colors list
    quadmesh = ax.findobj(QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()

    facecolors[0] = [0.35, 0.8, 0.55, 1.0]  # green
    facecolors[3] = [0.35, 0.8, 0.55, 1.0]  # green
    facecolors[1] = [0.65, 0.1, 0.1, 1.0]  # red
    facecolors[2] = [0.65, 0.1, 0.1, 1.0]  # red

    # set ticklabels rotation
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=25, fontsize=12)

    # set labels
    ax.axes.set_title("Model: {} \n Accuracy={:.2f}%, f_score={:.2f}".format(label,
                                                                             accuracy * 100,
                                                                             f_score),
                      fontsize=fsize)
    ax.set_xlabel("Predicted label", fontsize=15)
    ax.set_ylabel("Actual label", fontsize=15)
    plt.tight_layout()

    if image_path:
        plt.savefig(image_path)

    plt.show()


def roc_curve_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict:
    """
    Calculates ROC curve related metrics: AUC and fpr's and tpr's for thresholds.
    :param y_true: true label of the data, with shape (n_samples,), binary array
    :param y_score: target scores, can either be probability estimates of the positive class,
                    confidence values, or non-thresholded measure of decisions
    :return: dict of metrics
    """
    fprs, tprs, _ = roc_curve(y_true, y_score)
    auc_score = auc(fprs, tprs)
    return {'fprs': fprs, 'tprs': tprs, 'auc': auc_score}


def plot_roc_curve(roc_curve_metrics: Dict,
                   label: str = None,
                   image_path: str = None):
    """
    Plots RUC curve.
    :param roc_curve_metrics: the output of "roc_curve_metrics" function
    :param label: the name of line
    :param image_path: the path were image will be saved
    """
    plt.figure(figsize=(10, 10))

    temp = 'ROC (AUC = {:.2f})'.format(roc_curve_metrics['auc'])
    label = '{}: {}'.format(label, temp) if label else temp

    plt.plot(roc_curve_metrics['fprs'], roc_curve_metrics['tprs'], lw=3, alpha=0.3, label=label)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

    if image_path:
        plt.savefig(image_path)

    plt.show()


def precision_recall_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict:
    """
    Calculates precision-recall curve related metrics: average_precision_score and
    precision's and recall's for thresholds.
    :param y_true: true label of the data, with shape (n_samples,), binary array
    :param y_score: target scores, can either be probability estimates of the positive class,
                    confidence values, or non-thresholded measure of decisions
    :return:dict of metrics
    """
    average_precision = average_precision_score(y_true, y_score)
    precisions, recalls, _ = precision_recall_curve(y_true, y_score)
    return {'ap': average_precision, 'precisions': precisions, 'recalls': recalls}


def plot_precision_recall_curve(precision_recall_metrics: Dict,
                                label: str = None,
                                image_path: str = None):
    """
    Plots Precision-Recall curve.
    :param precision_recall_metrics: the output of "precision_recall_metrics" function
    :param label: the name of line
    :param image_path: the path were image will be saved
    """
    precisions = precision_recall_metrics['precisions']
    recalls = precision_recall_metrics['recalls']
    ap = precision_recall_metrics['ap']

    plt.figure(figsize=(10, 10))

    plt.step(recalls, precisions, color='b', alpha=0.2, where='post', label=label)
    plt.fill_between(recalls, precisions, step='post', alpha=0.2, color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve: AP={0:0.2f}'.format(ap))

    if image_path:
        plt.savefig(image_path)

    plt.show()


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Calculates regression metrics: RMSE, R2 and explained variance
    :param y_true: true value of the data, with shape (n_samples,), continuous array
    :param y_pred: predicted value of the data, with shape (n_samples,), continuous array
    :return: dict of metrics
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    explained_variance = explained_variance_score(y_true, y_pred)
    r_2 = r2_score(y_true, y_pred)
    return {'rmse': rmse, 'r_2': r_2, 'explained_variance': explained_variance}


def plot_actual_vs_pred(y_true: np.ndarray, y_pred: np.ndarray, label: str = None):
    """
    Plots actual vs predicted values scatter plot.
    :param y_true: true value of the data, with shape (n_samples,), continuous array
    :param y_pred: predicted value of the data, with shape (n_samples,), continuous array
    :param label: the name of the line
    """
    """Plots model predictions vs their actual values."""
    plt.figure(figsize=(10, 10))
    plt.scatter(y_true, y_pred, label=label)
    plt.title('Actual values vs Predicted values.', fontsize=18)
    plt.ylabel('Predicted', fontsize=14)
    plt.xlabel('Actual', fontsize=14)
    plt.legend(fontsize=18)
    plt.show()
