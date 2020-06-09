import numpy as np
from ai.domain_adaptation.datasets import image_index
from ai.domain_adaptation.utils import np_utils
from IPython.display import display, Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def load_data_for_vis(prob_path, target_domain_file, dataset_dir):
    domain_info = image_index.parse_domain_file(target_domain_file, dataset_dir)
    yhat_info = np_utils.parse_predictions_from_pickle(prob_path)

    return domain_info, yhat_info


def visulize_confidence(prob_path, target_domain_file, dataset_dir, cls_id):
    domain_info, yhat_info = load_data_for_vis(prob_path, target_domain_file, dataset_dir)
    vis_confident_predictions(cls_id, None, domain_info, yhat_info)


def vis_confident_predictions(cls_id, top_k=20, domain_info=None, yhat_info=None):
    sorted_id_indices = np_utils.retrieve_sorted_indices_for_one_cls(cls_id, yhat_info)

    for ith, example_id in enumerate(sorted_id_indices):
        filename, label = domain_info.image_path_label_tuples[example_id]
        print(f'{domain_info.label_description_dict[label]}, P {yhat_info.prob[example_id, cls_id]:.3}')
        img = Image(filename=filename, width=150, height=150)
        display(img)
        if top_k is not None and ith > top_k:
            break


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
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
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # np.set_printoptions(precision=3)

    fig, ax = plt.subplots(figsize=(20, 20))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # ax.figure.colorbar(im, ax=ax)
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
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(f'./plots/confusion_matrix{title}.pdf')
    return ax
