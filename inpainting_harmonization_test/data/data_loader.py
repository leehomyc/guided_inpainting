
def CreateDataLoader(opt):
    from inpainting_harmonization_test.data.custom_dataset_data_loader import CustomDatasetDataLoader  # noqa 501
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader
