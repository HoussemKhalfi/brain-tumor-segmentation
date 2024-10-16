import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def dice_coefficient(y_true, y_pred):
    """
    Calculate the Dice coefficient between two binary masks.

    Parameters:
    y_true (numpy array): Ground truth binary mask.
    y_pred (numpy array): Predicted binary mask.

    Returns:
    float: Dice coefficient score.
    """
    # Flatten the arrays to ensure they are 1D
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Calculate the intersection (True Positives)
    intersection = np.sum(y_true_flat * y_pred_flat)

    # Calculate the Dice coefficient
    dice = (2. * intersection) / (np.sum(y_true_flat) + np.sum(y_pred_flat))

    return dice

def create_confusion_matrix(y_true, y_pred, classes=None, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    Creates and displays a confusion matrix.

    Args:
        y_true: Array of true labels.
        y_pred: Array of predicted labels.
        classes: Array of class labels. If None, uses unique labels from y_true.
        normalize: If True, normalizes the confusion matrix.
        title: Title for the plot.
        cmap: Colormap to use for the plot.
    """

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if classes is None:
      classes = unique_labels(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.4f' if normalize else 'd'
    thresh = cm.max() / 4.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    return ax, cm