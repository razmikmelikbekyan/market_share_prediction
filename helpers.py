import itertools
import matplotlib.pyplot as plt
import numpy as np
import time

from sklearn.metrics import mean_squared_error, roc_auc_score, confusion_matrix
from scipy import stats



def confusion_matrix_rates(confusion_matrix):
    """
    This function calculates several rates using confusion matrix.
    """
    # true positives (TP): We predicted yes and they do have the disease.
    # true negatives (TN): We predicted no, and they don't have the disease.
    # false positives (FP): We predicted yes, but they don't have the disease.
    # false negatives (FN): We predicted no, but they do have the disease.
    tn, fp, fn, tp = confusion_matrix.ravel()

    # correct labels
    total = tn + fp + fn + tp
    actual_yes = fn + tp
    actual_no = total - actual_yes

    # Overall, how often is the classifier correct?
    accuracy = (tp + tn) / total

    # Overall, how often is it wrong?
    misclassification_rate = (fp + fn) / total

    # When it is actually yes, how often does it predict yes?
    recall = tp / actual_yes

    # When it is actually no, how often does it predict yes?
    fp_rate = fp / actual_no

    # When it is actually no, how often does it predict no?
    specificity = tn / actual_no

    # When it predicts yes, how often is it correct?
    precision = tp / (fp + tp)

    # The weighted average of recall and precision.
    f_score = 2 * recall * precision / (recall + precision)

    return (accuracy, misclassification_rate, recall,
            fp_rate, specificity, precision, f_score)


def normalize_confusion_matrix(confusion_matrix):
    """
    :param confusion_matrix: numpy array of confusion matrix
    :return: normalized numpy array of confusion matrix
    """
    confusion_matrix = (confusion_matrix.astype('float') /
                        confusion_matrix.sum(axis=1)[:, np.newaxis])
    return confusion_matrix


def plot_confusion_matrix(confusion_matrix, classes, cmap=plt.cm.Reds):
    """
    This function plots the normalized confusion matrix.

    :param confusion_matrix: numpy array of confusion matrix to be plotted
    :param classes: list of class names
    :param cmap: colormap colors
    """
    confusion_matrix = normalize_confusion_matrix(confusion_matrix)

    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title('Normalized Confusion matrix', fontsize=17)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=15)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    fmt = '.2f'
    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]),
                                  range(confusion_matrix.shape[1])):
        plt.text(j, i, format(confusion_matrix[i, j], fmt),
                 horizontalalignment="center", fontsize=17,
                 color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)
    plt.show()
    
    
def evaluate(y_true, y_pred, data_type, model_name):
    
    print()
    print('*****EVALUATION METRICS*****')
    
    print('Model: {}'.format(model_name))
    print('DataType: {}'.format(data_type))
    
    # calculating RMSE
    print('RMSE: {:.3f}'.format(np.sqrt(mean_squared_error(y_true, y_pred))))
    print()
    
    # plotting predicted values vs actual ones
    plt.figure(figsize=(10, 10))
    plt.scatter(y_true, y_pred, )
    plt.title('Actual values vs Predicted values on {} data'.format(data_type))
    plt.ylabel('Predicted', fontsize=12)
    plt.xlabel('Actual', fontsize=12)
    plt.show()
    
    # calculating binary predictions, for classification
    threshold = 0.007
    threshold_transformed = stats.boxcox(threshold, 0.25)

    binary_prediction = y_pred > threshold_transformed
    binary_target = y_true > threshold_transformed   
    
    # calculating classification metrics
    roc_auc = roc_auc_score(binary_target, binary_prediction)
    conf_matrix = confusion_matrix(binary_target, binary_prediction)
    
    (accuracy, misclassification_rate, recall, 
     fp_rate, specificity, precision, f_score) = confusion_matrix_rates(conf_matrix) 
    
    print()
    print('**ClassficationMetrics**')
    print('ROC AUC:        {:.3f}'.format(roc_auc))
    print('Accuracy:       {:.3f} (Overall, how often is the classifier correct?)'.format(accuracy))
    print('MissClass rate: {:.3f} (Overall, how often is it wrong?)'.format(misclassification_rate))
    print('Recall:         {:.3f} (When it is actually Y, how often does it predict Y?)'.format(recall))
    print('FalseP rate:    {:.3f} (When it is actually N, how often does it predict Y?)'.format(fp_rate))
    print('Precision:      {:.3f} (When it predicts Y, how often is it correct?)'.format(precision))
    print('F Score:        {:.3f} (The weighted average of recall and precision.)'.format(f_score))
    print()
    
    
    plot_confusion_matrix(conf_matrix, ['Failure', 'Success'])
    time.sleep(1)
 

 

def evaluate_train_test(model, train_data, train_y, test_data=None, test_y=None):
    # fitting model before prediction
    model.fit(train_data, train_y)
    
    # predicition
    train_y_pred = model.predict(train_data)
    
    # evaluation
    evaluate(train_y, train_y_pred, 'Train', model.__class__.__name__)
    
    if test_data is not None:
        # test prediction
        test_y_pred = model.predict(test_data)
    
        # evaluation
        evaluate(test_y, test_y_pred, 'Test', model.__class__.__name__)

