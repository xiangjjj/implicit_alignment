import torch
import torch.nn.functional as F


def get_proximal_loss(source_param, target_param, scale=0.00001):
    loss = 0
    for s, t in zip(source_param.parameters(), target_param.parameters()):
        loss += torch.norm(s - t, 2)
    return loss * scale


def get_entropy_loss(logits=None, prob=None, temperature=1):
    if logits is not None:
        prob = F.softmax(logits / temperature, dim=1)
    entropy = - prob * torch.log(prob + 1e-8)
    entropy = entropy.sum(dim=-1)
    entropy = entropy.mean()
    return entropy
