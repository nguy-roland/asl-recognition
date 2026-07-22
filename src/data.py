
import os
import re
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def extract_index(filename):
    """Extract the integer from a filename like 'A1000.jpg' -> 1000."""
    return int(re.search(r'\d+', filename).group())


class ASLDataset(Dataset):
    def __init__(self, root, split, train_ratio=0.8, transform=None, max_per_class=None):
        self.transform = transform
        self.classes = sorted(os.listdir(root))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = []

        for class_name in self.classes:
            class_dir = os.path.join(root, class_name)
            files = sorted(os.listdir(class_dir), key=extract_index)

            if max_per_class:
                files = files[:max_per_class]

            # Temporal split: early frames for training, later frames for validation
            cutoff = int(len(files) * train_ratio)
            selected = files[:cutoff] if split == 'train' else files[cutoff:]

            for f in selected:
                path = os.path.join(class_dir, f)
                self.samples.append((path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label