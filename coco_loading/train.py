import torch
import torchvision


def coco_collate(batch):
    """Puts each data field into a tensor with outer dimension batch size, or put collade recursively for dict"""
    if isinstance(batch[0], tuple):
        # if each batch element is not a tensor, then it should be a tuple
        # of tensors; in that case we collate each element in the tuple
        transposed = zip(*batch)
        return [coco_collate(samples) for samples in transposed]
    return batch


def train():
    train_set = torchvision.datasets.CocoDetection(
        root='/data/public/MSCOCO/train2017', annFile='/data/public/MSCOCO/annotations/instances_train2017.json')
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=32, shuffle=True, num_workers=2, collate_fn=coco_collate)
    for batch_idx, (inputs, target) in enumerate(train_loader):
        break


if __name__ == '__main__':
    train()
