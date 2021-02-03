import torch
import torchvision
from torchvision import transforms


def get_dataloaders(data_root: str, test_frac: float = 0.20, batch_size: int = 128):
    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset = torchvision.datasets.ImageFolder(data_root, transform=image_transforms)

    test_size = int(len(dataset) * test_frac)
    lengths = [len(dataset) - test_size, test_size]

    train, test = torch.utils.data.random_split(dataset, lengths)

    return {
        "train": torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True),
        "valid": torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True),
    }


