import torch
from torch import nn
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier


def restore_data_order(data, original_index):
    restored_data = torch.zeros_like(data)
    restored_data[original_index] = data
    return restored_data


class KNN:
    def __init__(self, args, encoder=None):
        self.args = args
        self.encoder = encoder
        self.metric = Metrics(args)

    def predict_from_dataloader(self, onepass_source_loader, onepass_target_loader):
        neigh = KNeighborsClassifier(n_neighbors=3, weights='uniform', n_jobs=-1)
        # neigh = RadiusNeighborsClassifier()
        with torch.no_grad():
            features_source, labels_source = self.get_features_from_dataloader(onepass_source_loader)
            features_target, _ = self.get_features_from_dataloader(onepass_target_loader)
        neigh.fit(features_source, labels_source)
        prob = neigh.predict_proba(features_target)
        prob = torch.FloatTensor(prob)
        return prob

    def get_features_from_dataloader(self, dataloader):
        features = []
        labels = []
        indices = []
        for index, x, y in dataloader:
            features.append(self.encoder(x.cuda()))
            labels.append(y)
            indices.extend(index)
        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)
        features_restored = restore_data_order(features, indices).cpu().numpy()
        labels_restored = restore_data_order(labels, indices).cpu().numpy()
        return features_restored, labels_restored


class ProtoNet:
    def __init__(self, args, encoder=None):
        self.args = args
        self.encoder = encoder
        self.metric = Metrics(args)

    def predict_from_dataloader(self, onepass_source_loader, onepass_target_loader):
        prob = []
        order = []
        y_hats = []
        with torch.no_grad():
            prototypes = self.construct_prototypes_from_dataloader(onepass_source_loader)
            for indices, x, y in onepass_target_loader:
                y_hat, p_y = self.predict(prototypes, x.cuda())
                prob.append(p_y)
                order.append(indices)
                y_hats.append(y_hat)
        probs = torch.cat(prob, dim=0)
        order = torch.cat(order, dim=0).cuda()

        # restore the order
        probs = restore_data_order(probs, order)
        return probs

    def construct_prototypes_from_dataloader(self, dataloader):
        num_batch = len(dataloader)
        for _, x, y in dataloader:
            proto_for_this_batch = self.get_prototypes(x.cuda(), y.cuda())
            if 'prototypes' not in dir():
                prototypes = proto_for_this_batch
            else:
                prototypes += proto_for_this_batch
        prototypes = prototypes / num_batch
        return prototypes

    def get_prototypes(self, x, y):
        z = self.forward(x)
        z_proto = self.get_prototype_for_one_task(z, y)
        return z_proto

    def get_prototype_for_one_class(self, z, y, class_id):
        indices = y == class_id
        if indices.sum() == 0:
            return torch.zeros(z.shape[1]).cuda()
        z_in_this_class = z[indices]
        prototype = z_in_this_class.mean(0)
        return prototype

    def forward(self, x):
        x = self.encoder(x)

        if self.args.normalize_metric_space is True:
            x = x / x.norm(p=2, dim=-1).unsqueeze(1)

        return x

    def get_prototype_for_one_task(self, z, y):
        prototypes_list = [self.get_prototype_for_one_class(z, y, class_id=cls_id) for cls_id in
                           range(self.args.class_num)]
        prototypes_tensor = torch.stack(prototypes_list)
        return prototypes_tensor

    def predict(self, prototype, query):
        z_proto = prototype
        z_query = self.forward(query)

        dists = self.metric.get_distance(z_query, z_proto)
        p_y = F.softmax(-dists, dim=-1)
        p_hat, y_hat = p_y.max(-1)
        return y_hat, p_y


class Metrics:
    def __init__(self, args):
        self.args = args
        self.distance_fn = getattr(self, args.distance)

    def get_distance(self, query, prototype):
        prototype = prototype.unsqueeze(0)
        query = query.unsqueeze(1)
        distances = self.distance_fn(prototype, query)
        return distances

    def manhattan(self, x, y):
        dist = torch.abs(x - y).sum(2)
        return dist

    def euclidean(self, x, y):
        dist = torch.pow(x - y, 2).sum(2)
        return dist

    def fractional(self, x, y, p=0.5):
        dist = torch.pow(x - y, p).sum(2).pow(1 / p)
        return dist

    def cosine(self, x, y):
        x = x / x.norm(dim=-1, keepdim=True)
        y = y / y.norm(dim=-1, keepdim=True)
        dist = x * y
        dist = dist.sum(dim=-1)
        return dist


def get_pairwise_distance(data, references):
    distances = [get_distance_with_each_reference(data[i, :], references)
                 for i in range(data.shape[0])]
    distances = torch.stack(distances)
    return distances


def get_distance_with_each_reference(data_i, references):
    dist = ((data_i - references) ** 2).sum(dim=1).sqrt()
    return dist


class TripletNet(nn.Module):
    def __init__(self, encoder, margin=0.01):
        super(TripletNet, self).__init__()
        self.embeddingnet = encoder
        self.margin = margin
        self.criterion = nn.MarginRankingLoss(margin=margin)

    def get_loss(self, anchor, positive, negative):
        dist_positives, dist_negatives = self.forward(anchor, positive, negative)
        target = -torch.ones(dist_positives.shape[0]).cuda()
        loss = self.criterion(dist_positives, dist_negatives, target)
        return loss

    def forward(self, anchor, positive, negative):
        embed_anchor = self.embeddingnet(anchor)
        embed_positive = self.embeddingnet(positive)
        embed_negative = self.embeddingnet(negative)
        dist_positives = F.pairwise_distance(embed_anchor, embed_positive, 2)
        dist_negatives = F.pairwise_distance(embed_anchor, embed_negative, 2)

        return dist_positives, dist_negatives
