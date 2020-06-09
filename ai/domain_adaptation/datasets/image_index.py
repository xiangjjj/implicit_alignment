from os import path
import random
import collections
import numpy as np

DomainInfo = collections.namedtuple('DomainInfo',
                                    """image_path_label_tuples img_path_list label_list label_description_dict 
                                    unique_labels unique_label_descriptions label_to_path_dict 
                                    num_unique_labels class_sample_size label_one_hot""")


def parse_domain_file(domain_file, dataset_dir):
    image_path_label_tuples = get_image_filenames_and_labels(domain_file, dataset_dir)
    img_path_list, label_list = zip(*image_path_label_tuples)
    label_list = np.array(label_list)
    label_description_dict = parse_label_descriptions(image_path_label_tuples)
    unique_labels = np.array(sorted(list(set(label_list))))
    num_unique_labels = len(unique_labels)
    unique_label_descriptions = [label_description_dict[i] for i in unique_labels]
    label_to_path_dict = parse_label_dict(image_path_label_tuples)
    class_sample_size = [len(label_to_path_dict[i]) for i in range(num_unique_labels)]
    label_one_hot = np.zeros((label_list.shape[0], len(unique_labels)))
    label_one_hot[np.arange(label_list.shape[0]), label_list] = 1
    return DomainInfo(
        image_path_label_tuples=image_path_label_tuples, img_path_list=img_path_list, label_list=label_list,
        label_description_dict=label_description_dict, unique_labels=unique_labels,
        unique_label_descriptions=unique_label_descriptions, label_to_path_dict=label_to_path_dict,
        num_unique_labels=num_unique_labels, class_sample_size=class_sample_size, label_one_hot=label_one_hot
    )


def get_image_files(images_file_path, args, is_train):
    if args.train_loss == 'classifier_loss':
        get_img_files_fn = get_image_files_for_classification
    else:
        get_img_files_fn = get_image_files_for_domain_adaptation

    image_files = get_img_files_fn(images_file_path, args, is_train)
    return image_files


def get_image_files_for_domain_adaptation(images_file_path, args, is_train):
    image_files = get_image_filenames_and_labels(images_file_path, args.datasets_dir)
    return image_files


def parse_label_descriptions(image_path_label_tuples):
    label_descriptions = [(path.split('/')[-2], label_id) for path, label_id in image_path_label_tuples]
    label_description_dict = {}
    for label_name, label_id in label_descriptions:
        label_description_dict[label_id] = label_name
    return label_description_dict


def parse_label_dict(image_path_label_tuples):
    label_dict = {}
    for img_path, y in image_path_label_tuples:
        if y not in label_dict:
            label_dict[y] = [img_path]
        else:
            label_dict[y].append(img_path)
    return label_dict


def get_image_files_for_classification(images_file_path, args, is_train):
    image_files = get_image_filenames_and_labels(images_file_path, args.datasets_dir)
    return image_files


def get_complement_image_files(all_files, subset_files):
    image_files = [img for img in all_files if img not in subset_files]
    return image_files


def get_image_filenames_and_labels(images_file_path, datasets_dir):
    image_files = open(images_file_path).readlines()
    image_files = [path.join(datasets_dir, img_file) for img_file in image_files]
    image_files = list(map(parse_filname_and_label, image_files))
    return image_files


def parse_filname_and_label(img_index):
    img_path, label_id = img_index.split()
    label_id = int(label_id)
    return img_path, label_id


def filter_image_files_by_class(image_files, filter_classes_str):
    start_id, end_id = filter_classes_str.split(',')
    start_id, end_id = int(start_id), int(end_id)

    def reindex_label_ids(img_index):
        img_path, label_id = img_index
        label_id = int(label_id) - start_id
        return img_path, label_id

    image_files = list(
        filter(
            lambda img_file: start_id <= img_file[1] < end_id,
            image_files))

    image_files = list(
        map(reindex_label_ids, image_files)
    )
    return image_files


def down_sample_images(image_files, k_shot):
    img_dict = group_images_into_dict(image_files)
    img_dict = sample_images_from_dict(img_dict, k_shot)
    image_files = imgdict_to_imglist(img_dict)
    return image_files


def group_images_into_dict(image_files):
    img_dict = {}
    for img, label in image_files:
        if label not in img_dict:
            img_dict[label] = [img]
        else:
            img_dict[label].append(img)
    return img_dict


def sample_images_from_dict(img_dict, sample_size):
    for k in img_dict:
        random.shuffle(img_dict[k])
        img_dict[k] = img_dict[k][:sample_size]
    return img_dict


def imgdict_to_imglist(img_dict):
    img_list = []
    for label_id in img_dict:
        for img_path in img_dict[label_id]:
            img_file = img_path, label_id
            img_list.append(img_file)
    return img_list


def extract_img_info(img_file):
    img_path, label_id = img_file.split()
    label_id = int(label_id)
    return img_path, label_id
