import torch.nn.functional as F
import torch

def cross_entropy(output, target, weight = None):
    return F.cross_entropy(output, target, weight=weight)

# cross_entropy = torch.nn.CrossEntropyLoss(weight = torch.FloatTensor([1.2, 0.2, 1.2, 0.2]))

def nll_loss(output, target):
    return F.nll_loss(output, target)

bce_logits_loss = torch.nn.BCELoss()
bce_logits_loss = torch.nn.BCEWithLogitsLoss()