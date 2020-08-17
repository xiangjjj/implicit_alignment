from torchvision import datasets


class CustomClassSubset:
    def __init__(self):
        self.unique_labels_list = list(range(10))

    def build_label_index_dict(self):
        self.label_index = {}
        for index, label in enumerate(self.labels):
            if label not in self.label_index:
                self.label_index[label] = [index]
            else:
                self.label_index[label].append(index)

    def get_subset_indices(self, p=None):
        if p is None:
            return None

        self.build_label_index_dict()
        all_indices = []
        for i, label in enumerate(self.unique_labels_list):
            label_indices = [self.label_index[label][index] for index in range(p[i])]
            all_indices.extend(label_indices)
        return all_indices


class mnist(datasets.MNIST, CustomClassSubset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=True,
                 unbalance_type='mild_unbalance'):
        datasets.MNIST.__init__(self, root, train=train, transform=transform, target_transform=target_transform,
                                download=download)
        CustomClassSubset.__init__(self)
        self.labels = self.targets.numpy()
        subset_dict_train = {
            'mild_unbalance': [542, 1084, 1626, 2168, 2710, 3252, 3794, 4336, 4878, 5421],
            'extreme_unbalance': [107,  164,  253,  389,  601,  930, 1441, 2237, 3479, 5421]
        }
        subset_test = [89, 178, 267, 356, 446, 535, 624, 713, 802, 892]

        if train is True:
            subset_indices = subset_dict_train[unbalance_type]
        else:
            subset_indices = self.get_subset_indices(p=subset_test)
        self.targets = self.labels = self.labels[subset_indices]
        self.data = self.data[subset_indices]

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return index, img, target


class svhn(datasets.SVHN, CustomClassSubset):
    def __init__(self, root, train, transform=None, target_transform=None, download=True,
            unbalance_type = 'mild_unbalance'):
        CustomClassSubset.__init__(self)
        if train is True:
            split = 'train'
        else:
            split = 'test'
            print('SVHN test loader...')
        datasets.SVHN.__init__(self, root=root, split=split, transform=transform, target_transform=target_transform,
                               download=download)

        # use `reversed` to simulate RS-UT
        subset_dict_train = {
            'mild_unbalance': list(reversed([465,  931, 1397, 1863, 2329, 2795, 3261, 3727, 4193, 4659])),
            'extreme_unbalance': list(reversed([92,  141,  217,  335,  517,  799, 1238, 1922, 2990, 4659]))
        }
        subset_test = list(reversed([159,  319,  478,  638,  797,  957, 1116, 1276, 1435, 1595]))

        if train is True:
            subset_indices = self.get_subset_indices(subset_dict_train[unbalance_type])
        else:
            subset_indices = self.get_subset_indices(p=subset_test)
        self.targets = self.labels = self.labels[subset_indices]
        self.data = self.data[subset_indices]

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return index, img, target
