import torch.utils.data


def CreateDataset(opt):
    dataset = None
    if opt.model == 'inpainting_guided':
        from inpainting.data.inpainting_dataset_guided import InpaintingDatasetGuided
        dataset = InpaintingDatasetGuided()
    elif opt.model == 'inpainting':
        from inpainting.data.inpainting_dataset import InpaintingDataset
        dataset = InpaintingDataset()
    elif opt.model == 'inpainting_test' or opt.model == 'harmonization_test':
        from inpainting.data.inpainting_dataset_test import InpaintingDatasetTest
        dataset = InpaintingDatasetTest()
    elif opt.model == 'inpainting_unguided':
        from inpainting.data.inpainting_dataset_unguided import InpaintingDatasetUnguided
        dataset = InpaintingDatasetUnguided()
    elif opt.model == 'inpainting_ade20k':
        from inpainting.data.inpainting_dataset_ADE20k import InpaintingDatasetADE20k
        dataset = InpaintingDatasetADE20k()
    elif opt.model == 'inpainting_unguided_general':
        from inpainting.data.inpainting_dataset_unguided_general import InpaintingDatasetUnguidedGeneral
        dataset = InpaintingDatasetUnguidedGeneral()
    elif opt.model == 'inpainting_unguided_teaser':
        from inpainting.data.inpainting_dataset_unguided_teaser import InpaintingDatasetUnguidedTeaser
        dataset = InpaintingDatasetUnguidedTeaser()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader:
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        self.opt = opt
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
