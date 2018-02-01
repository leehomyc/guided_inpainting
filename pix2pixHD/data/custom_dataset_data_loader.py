import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt):
    dataset = None
    from data.aligned_dataset import AlignedDataset
    from data.inpainting_dataset import InpaintingDataset
    from data.inpainting_grid_dataset import InpaintingGridDataset
    from data.inpainting_dataset_guided_style_transfer import InpaintingDatasetGuidedStyleTransfer

    if opt.model == 'inpainting':
        dataset = InpaintingDataset()
    elif opt.model == 'inpainting_grid':
        dataset = InpaintingGridDataset()
    elif opt.model == 'inpainting_guided_style_transfer':
        dataset = InpaintingDatasetGuidedStyleTransfer()
    elif opt.model == 'inpainting_object':
        from data.inpainting_dataset_object import InpaintingDatasetObject
        dataset = InpaintingDatasetObject()
    elif opt.model == 'inpainting_color':
        from data.inpainting_dataset_guided_color import InpaintingDatasetColor
        dataset = InpaintingDatasetColor()
    elif opt.model == 'inpainting_guided':
        from data.inpainting_dataset_guided import InpaintingDatasetGuided
        dataset = InpaintingDatasetGuided()
    else:
        dataset = AlignedDataset()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads),
            drop_last=True)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
