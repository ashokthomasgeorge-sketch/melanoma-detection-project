import os
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from pandas import read_csv


class DermaDataset(Dataset):
    """
    Custom PyTorch Dataset for dermatology image classification.
    Loads image files and their labels from metadata CSV.
    """

    IMAGE_EXTENSION = ".jpg"
    IMAGE_COLUMN = "maibi_id"
    LABEL_COLUMN = "dx"
    classes = ['bkl', 'nv', 'df', 'mel', 'vasc', 'bcc', 'akiec', 'other']

    LABEL_MAPPING = {
        # Nevus variations → nv
        'nv': 'nv',
        'nevus': 'nv',
        'nevi': 'nv',
        'nev': 'nv',
        'n': 'nv',

        # Melanoma variations → mel
        'mel': 'mel',
        'melanoma': 'mel',
        'mela': 'mel',
        'm': 'mel',
        'mn': 'mel',

        # Standard labels (no change)
        'bkl': 'bkl',
        'bcc': 'bcc',
        'akiec': 'akiec',
        'vasc': 'vasc',
        'df': 'df'
    }

    def __init__(self, image_directory: str, meta_data_path: str, transform=None):
        if not os.path.exists(image_directory):
            raise FileNotFoundError(f"Image directory not found: {image_directory}")
        if not os.path.exists(meta_data_path):
            raise FileNotFoundError(f"Metadata CSV not found: {meta_data_path}")

        self.image_directory = image_directory
        all_files = os.listdir(self.image_directory)
        self.image_files = [f for f in all_files if f.endswith(self.IMAGE_EXTENSION)]

        unindexed_meta_data = read_csv(meta_data_path)
        if self.IMAGE_COLUMN not in unindexed_meta_data.columns or \
           self.LABEL_COLUMN not in unindexed_meta_data.columns:
            raise ValueError(
                f"CSV must contain '{self.IMAGE_COLUMN}' and '{self.LABEL_COLUMN}' columns."
            )

        self.meta_data = unindexed_meta_data.set_index([self.IMAGE_COLUMN])
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file_name = self.image_files[idx]
        label = self.get_integer_label_for_image_name(image_file_name)
        image_path = os.path.join(self.image_directory, image_file_name)
        image = default_loader(image_path)
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_integer_label_for_image_name(self, image_name: str) -> int:
        """
        Converts a text label (e.g., 'mel') into an integer index
        according to the class order defined in self.classes.
        """
        text_label = self.get_text_label_for_image_name(image_name)
        if text_label in self.classes:
            return self.classes.index(text_label)
        else:
            return self.classes.index("other")

    def get_text_label_for_image_name(self, image_name: str) -> str:
        """
        Retrieves and normalizes the text label for a given image name
        using LABEL_MAPPING. Returns 'other' if missing or unrecognized.
        """
        no_extension = image_name[:-len(self.IMAGE_EXTENSION)]
        if no_extension not in self.meta_data.index:
            return "other"

        text_label = str(self.meta_data.loc[no_extension][self.LABEL_COLUMN]).strip().lower()
        return self.LABEL_MAPPING.get(text_label, "other")


#Test runner for VS Code
if __name__ == "__main__":
    print("🔍 Testing DermaDataset locally...")

    #Replace these paths with your actual ones
    image_dir = r"C:\Users\ashok\Desktop\Barco\legacy"
    csv_path = r"C:\Users\ashok\Desktop\Barco\metadata_legacy.csv"

    if not os.path.exists(image_dir) or not os.path.exists(csv_path):
        print("  Please update the image_dir and csv_path in the code before running.")
    else:
        dataset = DermaDataset(image_directory=image_dir, meta_data_path=csv_path)
        print(f"Loaded {len(dataset)} images from {image_dir}")
        image, label = dataset[0]
        print(f"First label index: {label}, class name: {dataset.classes[label]}")

        # INSERT WeightedRandomSampler CODE HERE
        from torch.utils.data import DataLoader, WeightedRandomSampler
        import numpy as np
        import torch

        # Compute sampling weights for each class
        targets = [dataset.get_integer_label_for_image_name(f) for f in dataset.image_files]
        class_counts = np.bincount(targets, minlength=len(dataset.classes))
        class_weights = 1. / (class_counts + 1e-6)
        samples_weight = np.array([class_weights[t] for t in targets])
        samples_weight = torch.from_numpy(samples_weight).double()

        sampler = WeightedRandomSampler(
            weights=samples_weight,
            num_samples=len(samples_weight),
            replacement=True
        )

        # Create a DataLoader that uses the sampler
        train_loader = DataLoader(dataset, batch_size=32, sampler=sampler, num_workers=4)

        # Verify
        print("WeightedRandomSampler applied successfully!")
        print(f"Class counts: {class_counts}")
        print(f"Class weights: {class_weights}")
