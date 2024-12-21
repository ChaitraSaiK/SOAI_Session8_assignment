import albumentations as A
from albumentations.pytorch import ToTensorV2

# Dataset configuration
DATASET_MEAN = (0.4914, 0.4822, 0.4465)
DATASET_STD = (0.2470, 0.2435, 0.2616)

# Training configuration
DEVICE = "cuda"
LEARNING_RATE = 0.001
BATCH_SIZE = 128
NUM_EPOCHS = 50
NUM_CLASSES = 10

# Data augmentation
train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.CoarseDropout(
        max_holes=1,
        max_height=16,
        max_width=16,
        min_holes=1,
        min_height=16,
        min_width=16,
        fill_value=[int(x * 255) for x in DATASET_MEAN],
        p=0.5
    ),
    A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
    ToTensorV2()
])

test_transforms = A.Compose([
    A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
    ToTensorV2()
]) 