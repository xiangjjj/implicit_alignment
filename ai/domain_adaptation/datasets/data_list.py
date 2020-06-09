from PIL import Image
import random
import numpy as np


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    # from torchvision import get_image_backend
    # if get_image_backend() == 'accimage':
    #    return accimage_loader(path)
    # else:
    return pil_loader(path)


class ImageList(object):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        img_items (list): List of (image path, class_index) tuples
    """

    def __init__(self, image_list, labels=None, transform=None, target_transform=None,
                 loader=default_loader):
        if len(image_list) == 0:
            raise (RuntimeError("Found 0 images in the dataset"))

        self.img_items = image_list
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.img_files, self.labels = zip(*self.img_items)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.img_items[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, img, target

    def __len__(self):
        return len(self.img_items)


class TripletImageList(object):
    def __init__(self, image_list, transform=None, target_transform=None, loader=default_loader):
        if len(image_list) == 0:
            raise (RuntimeError("Found 0 images in the dataset"))

        self.img_items = image_list
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.img_files, self.labels = zip(*self.img_items)
        self.unique_labels = list(set(self.labels))

        self.label_index = {}
        self.build_label_index()

    def build_label_index(self):
        for filename, label in zip(self.img_files, self.labels):
            if label not in self.label_index:
                self.label_index[label] = [filename]
            else:
                self.label_index[label].append(filename)

    def sample_triplet(self):
        pos_cls, neg_cls = self.sample_positive_negative_classes()
        positives = self.sample_example_from_cls(pos_cls, 2)
        negative = self.sample_example_from_cls(neg_cls, 1)
        return positives[0], positives[1], negative[0]

    def sample_positive_negative_classes(self):
        return np.random.choice(self.unique_labels, size=2, replace=False)

    def sample_example_from_cls(self, cls_id, num_sample):
        return np.random.choice(self.label_index[cls_id], size=num_sample, replace=False)

    def transform_img(self, path):
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __getitem__(self, index):
        anchor, positive, negative = self.sample_triplet()
        anchor, positive, negative = map(self.transform_img, (anchor, positive, negative))
        return anchor, positive, negative

    def __len__(self):
        return len(self.img_items)
