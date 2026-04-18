from torch.utils.data import DataLoader
from dataset import BiscuitDataset, get_train_transform

dataset = BiscuitDataset(
    csv_path="data/processed/train.csv",
    transform=get_train_transform()
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)

images, labels = next(iter(loader))

print("Batch shape:", images.shape)
print("Labels shape:", labels.shape)
print("Labels:", labels)
