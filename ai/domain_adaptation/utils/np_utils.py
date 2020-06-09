import numpy as np
import collections

YHatInfo = collections.namedtuple('YHatInfo', 'prob y_hat y_hat_to_indices_dict confidence')


def parse_predictions_from_pickle(prob_path):
    prob = np.load(prob_path, allow_pickle=True)
    y_hat = np.argmax(prob, axis=1)
    confidence = np.amax(prob, axis=1)
    y_hat_to_indices_dict = build_pseudo_label_dict(y_hat)
    yhat_info = YHatInfo(prob=prob, y_hat=y_hat, y_hat_to_indices_dict=y_hat_to_indices_dict, confidence=confidence)
    return yhat_info


def build_pseudo_label_dict(y_hat):
    label_dict = collections.OrderedDict()
    for i, label in enumerate(y_hat):
        if label not in label_dict:
            label_dict[label] = [i]
        else:
            label_dict[label].append(i)
    return label_dict


def compute_entropy(prob):
    entropy = [-p * np.log(p + 1e-8) for p in prob]
    entropy = sum(entropy)
    return entropy


def retrieve_sorted_indices_for_one_cls(cls_id, yhat_info):
    cls_probs = yhat_info.prob[yhat_info.y_hat_to_indices_dict[cls_id], cls_id]
    sorted_id_indices_in_this_cls = np.argsort(cls_probs)[::-1]
    sorted_id_indices_for_the_dataset = [
        yhat_info.y_hat_to_indices_dict[cls_id][i] for i in sorted_id_indices_in_this_cls]
    return sorted_id_indices_for_the_dataset
