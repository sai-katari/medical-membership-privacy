from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(image_size: int = 224) -> dict:
    """Minimal baseline transforms — no augmentation.

    Augmentation affects prediction confidence and therefore membership
    leakage. It is only introduced as an explicit regularisation experiment.
    """
    t = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return {"train": t, "val": t, "test": t}
