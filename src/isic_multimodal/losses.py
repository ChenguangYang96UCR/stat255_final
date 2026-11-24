import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.register_buffer("weight", weight)
        self.gamma = gamma

    def forward(self, logits, targets):
        log_probability = F.log_softmax(logits, dim=1)
        log_pt = log_probability.gather(1, targets[:, None]).squeeze(1)
        loss = -((1.0 - log_pt.exp()) ** self.gamma) * log_pt
        if self.weight is not None:
            loss = loss * self.weight[targets]
        return loss.mean()


def contrastive_alignment(image_embeddings, metadata_embeddings, temperature=0.1):
    similarity = image_embeddings @ metadata_embeddings.T / temperature
    targets = torch.arange(similarity.size(0), device=similarity.device)
    return 0.5 * (F.cross_entropy(similarity, targets) + F.cross_entropy(similarity.T, targets))


def class_weights(labels, num_classes=8):
    counts = torch.bincount(torch.as_tensor(labels, dtype=torch.long), minlength=num_classes).float()
    weights = counts.sum() / (num_classes * counts.clamp_min(1.0))
    return weights


def build_loss(name, weights=None, focal_gamma=2.0):
    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=weights)
    if name == "focal":
        return FocalLoss(weight=weights, gamma=focal_gamma)
    raise ValueError(f"Unknown loss: {name}")

