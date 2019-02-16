from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn import clone
from sklearn.model_selection import KFold
from tabulate import tabulate

from metrics import (confusion_matrix_metrics, roc_curve_metrics, precision_recall_metrics,
                     regression_metrics, plot_actual_vs_pred, plot_confusion_matrix)

TO_VERBOSE = ('rmse', 'r_2', 'accuracy', 'tpr', 'fpr', 'ppv', 'npv', 'f_score', 'auc', 'ap')


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> Dict:
    """
    Calculates Regression and Classification related all metrics.
    :param y_true: true value of the data, with shape (n_samples,), continuous array
    :param y_pred: predicted value of the data, with shape (n_samples,), continuous array
    :param threshold: the threshold for transforming scores to binary arrays
    :return: dict of metrics
    """
    y_true_binary = y_true > threshold
    y_pred_binary = y_pred > threshold

    return {
        **regression_metrics(y_true, y_pred),
        **confusion_matrix_metrics(y_true_binary, y_pred_binary),
        **roc_curve_metrics(y_true_binary, y_pred),
        **precision_recall_metrics(y_true_binary, y_pred),
    }


def print_metrics(models_metrics: Dict[str, Dict]):
    """
    Prints metrics.
    :param models_metrics: the dict: {model_name: model_metrics}, where each model_metrics is the
                           dict itself and is the output of calculate_metrics
    """
    models, metrics = zip(*models_metrics.items())

    tabular_format = [[k] + [item[k] for item in metrics] for k in TO_VERBOSE]
    tabular_format = tabulate(
        tabular_format,
        headers=['metric_name'] + list(models),
        tablefmt="fancy_grid",
        floatfmt=",.3f")
    print(tabular_format)


def plot_model_performance(y_true: np.ndarray, y_pred: np.ndarray, metrics: Dict, model_name: str):
    plot_actual_vs_pred(y_true, y_pred, label=model_name)
    print()

    plot_confusion_matrix(metrics, label=model_name)
    print()


def plot_multiple_roc_curves(models_metrics: Dict[str, Dict], image_path: str = None):
    plt.figure(figsize=(15, 10))

    for model, metrics in models_metrics.items():
        label = '{}: AUC = {:.3f}'.format(model, metrics['auc'])
        plt.plot(metrics['fprs'], metrics['tprs'], lw=2, alpha=0.6, label=label)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=3, color='r', label='Chance', alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.title('ROC curve', fontsize=18)
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.legend(loc="lower right", fontsize=18)

    if image_path:
        plt.savefig(image_path)

    plt.show()


def plot_multiple_precision_recall_curves(models_metrics: Dict[str, Dict], image_path: str = None):
    plt.figure(figsize=(15, 10))

    for model, metrics in models_metrics.items():
        plt.step(metrics['recalls'], metrics['precisions'], color='b', alpha=0.2, where='post')
        plt.fill_between(metrics['recalls'], metrics['precisions'], step='post', alpha=0.2,
                         label='{}: AP={:.3f}'.format(model, metrics['ap']))

    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.xlabel('Recall', fontsize=18)
    plt.ylabel('Precision', fontsize=18)
    plt.title('Precision-Recall curve', fontsize=18)
    plt.legend(loc="lower right", fontsize=18, shadow=True)

    if image_path:
        plt.savefig(image_path)

    plt.show()


def test_model(model,
               model_name,
               predictors: np.ndarray,
               y_true: np.ndarray,
               threshold: float,
               plot: bool = True,
               verbose: bool = True):
    """
    Tests model and returns results.
    :return: model, predicted_values, test_metrics
    """
    y_pred = model.predict(predictors)
    metrics = calculate_metrics(y_true, y_pred, threshold)

    if plot:
        plot_model_performance(y_true, y_pred, metrics, model_name)

    if verbose:
        print_metrics({model_name: metrics})

    return model, y_pred, metrics


def train_model(model,
                model_name,
                predictors: np.ndarray,
                y_true: np.ndarray,
                threshold: float,
                plot: bool = True,
                verbose: bool = True):
    """
    Tests model and returns results.
    :return: model, predicted_values, test_metrics
    """
    model.fit(predictors, y_true)
    return test_model(model, model_name, predictors, y_true, threshold, plot=plot, verbose=verbose)


def cv_mean_metrics(folds_metrics: List[Dict]):
    """Calculates model mean metrics based on list of each folds metrics."""
    if not folds_metrics:
        return
    data = {k: [x[k] for x in folds_metrics] for k in TO_VERBOSE}
    return {k: (np.mean(v), np.std(v)) for k, v in data.items()}


def cv_model(model, model_name, predictors, y_true, threshold, n_folds=5, verbose=True):
    """Helper function for doing cross validation and collecting metrics."""

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    training_metrics, validation_metrics = [], []
    for i, (train_ind, valid_ind) in enumerate(kf.split(predictors)):
        model_, _, t_metrics = train_model(clone(model), model_name,
                                           predictors.iloc[train_ind],
                                           y_true.iloc[train_ind],
                                           threshold, plot=False, verbose=False)
        _, _, v_metrics = test_model(model_, model_name,
                                     predictors.iloc[valid_ind],
                                     y_true.iloc[valid_ind],
                                     threshold, plot=False, verbose=False)

        tabular_metrics = [[k, t_metrics[k], v_metrics[k]] for k in TO_VERBOSE]

        if verbose:
            print('\nfold: {}\n'.format(i + 1))
            print(tabulate(tabular_metrics,
                           headers=['metric_name', 'training_set', 'validation_set'],
                           tablefmt="fancy_grid",
                           floatfmt=",.3f"))

        training_metrics.append(t_metrics)
        validation_metrics.append(v_metrics)

    mean_t_metrics = cv_mean_metrics(training_metrics)
    mean_v_metrics = cv_mean_metrics(validation_metrics)

    tabular_mean_metrics = [
        [k, mean_t_metrics[k][0], mean_t_metrics[k][1], mean_v_metrics[k][0], mean_v_metrics[k][1]]
        for k in mean_t_metrics.keys()]

    tabular_mean_metrics = tabulate(
        tabular_mean_metrics,
        headers=['metric_name', 'train: mean', 'train: std', 'valid: mean', 'valid: std'],
        tablefmt="fancy_grid",
        floatfmt=",.3f")
    return tabular_mean_metrics, mean_t_metrics, mean_v_metrics


if __name__ == '__main__':
    threshold = 0.007
    y_true = np.random.exponential(1 / 100, 50)
    y_pred = np.random.exponential(1 / 100, 50)
    metrics_1 = calculate_metrics(y_true, y_pred, threshold)

    threshold = 0.007
    y_true = np.random.exponential(1 / 100, 50)
    y_pred = np.random.exponential(1 / 100, 50)
    metrics_2 = calculate_metrics(y_true, y_pred, threshold)

    threshold = 0.007
    y_true = np.random.exponential(1 / 100, 50)
    y_pred = np.random.exponential(1 / 100, 50)
    metrics_3 = calculate_metrics(y_true, y_pred, threshold)

    k = {'a': metrics_1, 'b': metrics_2, 'c': metrics_3}

    print_metrics(k)
    plot_multiple_roc_curves(k)
    plot_multiple_precision_recall_curves(k)

    from sklearn.linear_model import LinearRegression

    m = LinearRegression()

    l, p, metr = train_model(m, 'linear', y_pred.reshape(-1, 1), y_true, threshold)

    # binary_y_true = y_true > threshold
    #
    # plot_confusion_matrix(a, ['N', 'Y'], figsize=(5, 5), fsize=12, image_path='a.png')
    # plot_roc_curve(b, label='my model', image_path='b.png')
    # plot_precision_recall_curve(c, label='my model', image_path='c.png')
