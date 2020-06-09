from torch.utils.data import Sampler
from collections import OrderedDict
from ai.domain_adaptation.trainer.losses import get_entropy_loss
import random
import torch


class Singleton(type):
    """
    Define an Instance operation that lets clients access its unique
    instance.
    """

    def __init__(cls, name, bases, attrs, **kwargs):
        super().__init__(name, bases, attrs)
        cls._instance = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


class TaskSampler(metaclass=Singleton):
    def __init__(self, unique_classes, args):
        self.unique_classes = sorted(unique_classes)
        self.args = args
        self.k_shot = args.k_shot
        self.n_way = args.n_way
        self.counter = 0
        self.sampled_classes = None
        self.sample_freq = self.get_sampling_frequency()

    def get_sampling_frequency(self):
        # because of the Singleton and two dataloaders: source and target
        # need to make sure sampled classes align between souce and target
        if self.args.self_train is True:
            freq = 2
        else:
            freq = 1
        return freq

    def sample_N_classes_as_a_task(self):
        # mod-2 because both the source and target domains are using the same sampler
        # sometimes need to make sure they sample the same set of classes
        if self.counter % self.sample_freq == 0:
            self.sampled_classes = random.sample(self.unique_classes, self.n_way)

        self.counter += 1
        return self.sampled_classes


class N_Way_K_Shot_BatchSampler(Sampler):
    def __init__(self, y, max_iter, task_sampler):
        self.y = y
        self.max_iter = max_iter
        self.task_sampler = task_sampler
        self.label_dict = self.build_label_dict()
        self.unique_classes_from_y = sorted(set(self.y))

    def build_label_dict(self):
        label_dict = OrderedDict()
        for i, label in enumerate(self.y):
            if label not in label_dict:
                label_dict[label] = [i]
            else:
                label_dict[label].append(i)
        return label_dict

    def sample_examples_by_class(self, cls):
        if cls not in self.label_dict:
            return []

        if self.task_sampler.k_shot <= len(self.label_dict[cls]):
            sampled_examples = random.sample(self.label_dict[cls],
                                             self.task_sampler.k_shot)  # sample without replacement
        else:
            sampled_examples = random.choices(self.label_dict[cls],
                                              k=self.task_sampler.k_shot)  # sample with replacement
        return sampled_examples

    def __iter__(self):
        for _ in range(self.max_iter):
            batch = []
            classes = self.task_sampler.sample_N_classes_as_a_task()
            if len(batch) == 0:
                for cls in classes:
                    samples_for_this_class = self.sample_examples_by_class(cls)
                    batch.extend(samples_for_this_class)

            yield batch

    def __len__(self):
        return self.max_iter


class SelfTrainingBaseSampler(Sampler):
    def __init__(self, max_iter, task_sampler, args):
        self.max_iter = max_iter
        self.args = args
        self.task_sampler = task_sampler
        self.probs, self.y_hat, self.y_prob = None, None, None
        self.pseudo_label_dict = None

    def update_predicted_probs(self, probs):
        self.probs = probs
        self.y_prob, self.y_hat = self.probs.max(1)
        if self.args.confidence_threshold is not None:
            self.y_hat[self.y_prob < self.args.confidence_threshold] = -1
        self.pseudo_label_dict = self.build_pseudo_label_dict()

    def build_pseudo_label_dict(self):
        label_dict = OrderedDict()
        for i, label in enumerate(self.y_hat):
            label = label.item()
            if label not in label_dict:
                label_dict[label] = [i]
            else:
                label_dict[label].append(i)

        # make sure there is no missing label
        for i in range(self.args.class_num):
            if i not in label_dict:
                label_dict[i] = []
        return label_dict

    def remove_list_of_examples_from_pseudo_label_dict(self, cls, data_indices):
        for d in data_indices:
            self.remove_one_example_from_pseudo_label_dict(cls, d)

    def remove_one_example_from_pseudo_label_dict(self, cls, data_index):
        if cls in self.pseudo_label_dict:
            self.pseudo_label_dict[cls].remove(data_index)
        else:
            raise ValueError(f'class {cls} not present in pseudo label dictionary')

    def __iter__(self):
        for _ in range(self.max_iter):
            batch = []
            classes = self.task_sampler.sample_N_classes_as_a_task()
            if len(batch) == 0:
                for cls in classes:
                    if cls not in self.pseudo_label_dict or not self.pseudo_label_dict[cls]:
                        continue
                    samples_for_this_class = self.sample_examples_by_class(cls)
                    batch.extend(samples_for_this_class)

            if len(batch) < self.task_sampler.args.batch_size:
                random_samples = random.sample(range(len(self.y_hat)), self.task_sampler.args.batch_size - len(batch))
                batch.extend(random_samples)

            yield batch

    def __len__(self):
        return self.max_iter


class SelfTrainingVannilaSampler(SelfTrainingBaseSampler):
    def __init__(self, max_iter, task_sampler, args):
        super().__init__(max_iter, task_sampler, args)

    def sample_examples_by_class(self, cls):
        if self.task_sampler.k_shot <= len(self.pseudo_label_dict[cls]):
            sampled_examples = random.sample(self.pseudo_label_dict[cls],
                                             self.task_sampler.k_shot)  # sample without replacement
        else:
            sampled_examples = random.choices(self.pseudo_label_dict[cls],
                                              k=self.task_sampler.k_shot)  # sample with replacement
        return sampled_examples


class SelfTrainingConfidentSampler(SelfTrainingBaseSampler):
    def __init__(self, max_iter, task_sampler, args):
        super().__init__(max_iter, task_sampler, args)

    def sample_examples_by_class(self, cls):
        prob = self.probs[self.pseudo_label_dict[cls], cls]
        top_k_indices_for_this_cls = torch.topk(prob, self.task_sampler.k_shot)[1].cpu().numpy().tolist()
        top_k_indices = [self.pseudo_label_dict[cls][i] for i in top_k_indices_for_this_cls]
        return top_k_indices


class SelfTrainingUnConfidentSampler(SelfTrainingBaseSampler):
    def __init__(self, max_iter, task_sampler, args):
        super().__init__(max_iter, task_sampler, args)

    def sample_examples_by_class(self, cls):
        prob = self.probs[self.pseudo_label_dict[cls], cls]
        top_k_indices_for_this_cls = torch.topk(1 - prob, self.task_sampler.k_shot)[1].cpu().numpy().tolist()
        top_k_indices = [self.pseudo_label_dict[cls][i] for i in top_k_indices_for_this_cls]
        return top_k_indices


class SelfTrainingMedianConfidentSampler(SelfTrainingBaseSampler):
    def __init__(self, max_iter, task_sampler, args):
        super().__init__(max_iter, task_sampler, args)

    def sample_examples_by_class(self, cls):
        prob = self.probs[self.pseudo_label_dict[cls], cls]
        median_index_for_this_cls = [torch.median(prob, dim=0)[1].item()]
        median_index = [self.pseudo_label_dict[cls][i] for i in median_index_for_this_cls]
        return median_index


class SelfTrainingEpistemicSampler(SelfTrainingBaseSampler):
    def __init__(self, max_iter, task_sampler, args):
        super().__init__(max_iter, task_sampler, args)

    def sample_examples_by_class(self, cls):
        prob = self.probs[self.pseudo_label_dict[cls]]
        entropies = get_entropy_loss(prob=prob, element_wise=True)
        top_k_indices_for_this_cls = torch.topk(entropies, self.task_sampler.k_shot)[1].cpu().numpy().tolist()
        top_k_indices = [self.pseudo_label_dict[cls][i] for i in top_k_indices_for_this_cls]
        return top_k_indices
