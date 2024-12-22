import albumentations as A
from albumentations.pytorch import ToTensorV2

# Dataset configuration
DATASET_MEAN = (0.4914, 0.4822, 0.4465)
DATASET_STD = (0.2470, 0.2435, 0.2616)

# Training configuration
DEVICE = "cuda"
LEARNING_RATE = 0.005
BATCH_SIZE = 128
NUM_EPOCHS = 100
NUM_CLASSES = 10

# Data augmentation
train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=15, p=0.5),
    A.OneOf([
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    ], p=0.5),
    A.CoarseDropout(
        max_holes=8,
        max_height=8,
        max_width=8,
        min_holes=1,
        min_height=1,
        min_width=1,
        fill_value=[int(x * 255) for x in DATASET_MEAN],
        p=0.5
    ),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
    ToTensorV2()
])

test_transforms = A.Compose([
    A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
    ToTensorV2()
]) 