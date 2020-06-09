import numpy as np
import pandas as pd
from ai.domain_adaptation.utils import vis
from sklearn.metrics import classification_report


def analyze_predictions(cls_id, sorted_id_indices_for_the_dataset, domain_info):
    label_description = domain_info.unique_label_descriptions[cls_id]
    ground_truth = np.array([domain_info.label_list[i] for i in sorted_id_indices_for_the_dataset])
    true_positive = np.sum(ground_truth == cls_id)
    predicted_positive = len(sorted_id_indices_for_the_dataset)
    all_positive = len(domain_info.label_to_path_dict[cls_id])
    precision = true_positive / predicted_positive
    recall = true_positive / all_positive
    return {
        'class': label_description,
        'PP': predicted_positive,
        'TP': true_positive,
        'precision': precision,
        'recall': recall
    }


def get_eval_df(prob_path, source_domain_file, target_domain_file, dataset_dir, name=''):
    domain_info, yhat_info = vis.load_data_for_vis(prob_path, target_domain_file, dataset_dir)
    source_domain_info, yhat_info = vis.load_data_for_vis(prob_path, source_domain_file, dataset_dir)
    source_cls_size = source_domain_info.class_sample_size + [0, 0, 0]

    vis.plot_confusion_matrix(domain_info.label_list, yhat_info.y_hat, classes=domain_info.unique_label_descriptions,
                              title=f'Confusion matrix {name}')
    report = classification_report(domain_info.label_list, yhat_info.y_hat, output_dict=True)
    report = {(int(k) if k.isdigit() else k): v for k, v in report.items()}
    report_df = pd.DataFrame.from_dict(report)
    report_df = report_df.transpose().astype({'support': int}).add_suffix(name)
    report_df = report_df.rename(columns={f'support{name}': 'target size'})
    report_df[f'source size'] = source_cls_size

    report_df = report_df.astype({'source size': int})

    return report_df


