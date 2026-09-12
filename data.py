from functools import lru_cache

import numpy as np
from datasets import load_from_disk
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import BATCH_SIZE, FRACTIONS, IMAGE_SIZE, PACS_PATH


normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225],)

train_tf = transforms.Compose([
    transforms.Resize((144, 144)),
    transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

eval_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    normalize,
])


@lru_cache(maxsize=1)
def load_pacs_data():
    pacs = load_from_disk(str(PACS_PATH))

    if hasattr(pacs, "keys"):
        return pacs[list(pacs.keys())[0]]

    return pacs


class PACSDataset(Dataset):
    def __init__(self, indices, transform):
        self.pacs_data = load_pacs_data()

        self.indices = [int(index) for index in indices]

        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        dataset_index = int(self.indices[int(index)])

        row = self.pacs_data[dataset_index]

        image = row["image"].convert("RGB")

        return (self.transform(image), int(row["label"]),)


@lru_cache(maxsize=1)
def build_data_state():
    pacs_data = load_pacs_data()

    domains = sorted(set(pacs_data["domain"]))
    domain_array = np.array(pacs_data["domain"])
    label_array = np.array(pacs_data["label"])

    splits = {}

    for domain in domains:
        domain_indices = np.where(domain_array == domain)[0]

        train_indices, test_indices = train_test_split(
            domain_indices,
            test_size=0.20,
            stratify=label_array[domain_indices],
            random_state=42,
        )

        train_indices, validation_indices = train_test_split(
            train_indices,
            test_size=0.125,
            stratify=label_array[train_indices],
            random_state=42,
        )

        splits[domain] = {"train": train_indices, "val": validation_indices, "test": test_indices,}

    # Nested subsets: 10% contains 5%, 25% contains 10%, and so on.
    target_subsets = {}

    for domain in domains:
        training_pool = splits[domain]["train"]
        rng = np.random.default_rng(42)
        indices_by_class = []

        for label in np.unique(label_array):
            class_indices = training_pool[label_array[training_pool] == label].copy()

            rng.shuffle(class_indices)
            indices_by_class.append(class_indices)

        target_subsets[domain] = {
            fraction: np.concatenate([
                class_indices[: max(1, round(len(class_indices) * fraction))]
                for class_indices in indices_by_class
            ])
            for fraction in FRACTIONS
        }

    return {
        "pacs_data": pacs_data,
        "domains": domains,
        "splits": splits,
        "target_subsets": target_subsets,
        "label_array": label_array,
    }


def make_loader(indices, train=False, batch_size=BATCH_SIZE):
    dataset = PACSDataset(indices, train_tf if train else eval_tf,)

    return DataLoader(dataset, batch_size = batch_size, shuffle = train,)