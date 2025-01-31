from dataset import SynthText, TotalText

def get_dataset(dataset_name, train = False, transform = None):
    if dataset_name == "TotalText":
        return TotalText(train=train, transform=transform)
    elif dataset_name == "SynthText":
        return SynthText(transform=transform)
    else:
        raise ValueError("Dataloader not found in the list of dataloaders.")