import torch
import torchvision

trainset = torchvision.datasets.CocoDetection(
    root='/data/public/MSCOCO', annFile='/data/public/MSCOCO/annotations/instances_train2017.json')
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=32, shuffle=True, num_workers=2)