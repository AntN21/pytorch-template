import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        # print(pred.shape,target.shape)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def multilabel_accuracy(output, target):
    with torch.no_grad():
        pred = 1.*(torch.sigmoid(output) > .5)
        # print(pred.shape,target.shape)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / target.numel()

def multilabel_recall(output, target):
    with torch.no_grad():
        pred = 1.*(torch.sigmoid(output) > .5)
        # print(pred.shape,target.shape)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum((1.*(pred == target)) * (1.*(target == 1))).item()
    return correct /  torch.sum(1.*(target == 1))

def top_k_acc(output, target, k=2):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
