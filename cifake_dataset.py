from os import listdir
from os.path import exists, isdir, join
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class CIFAKEDataset(Dataset):
    def __init__(self, dataset_path, split, resolution=224, norm_mean=IMAGENET_DEFAULT_MEAN, norm_std=IMAGENET_DEFAULT_STD):
        assert isdir(dataset_path), f"got {dataset_path}"
        self.dataset_path = dataset_path
        assert split in {"train", "test"}, f"got {split}"
        self.split = split
        
        # parses metas from the datasets
        self.items = self.parse_dataset()
        
        # sets up the preprocessing options
        assert isinstance(resolution, int) and resolution >= 1, f"got {resolution}"
        self.resolution = resolution
        assert len(norm_mean) == 3
        self.norm_mean = norm_mean
        assert len(norm_std) == 3
        self.norm_std = norm_std

    def parse_dataset(self):
        def is_image(filename):
            for extension in ["jpg", "png", "jpeg"]:
                if filename.lower().endswith(extension):
                    return True
            return False
        
        split_path = join(self.dataset_path, self.split)
        items = [{
            "image_path":  join(split_path, "REAL", image_path),
            "is_real": True
            } for image_path in listdir(join(split_path, "REAL")) if is_image(image_path)] + [{
            "image_path":  join(split_path, "FAKE", image_path),
            "is_real": False
            } for image_path in listdir(join(split_path, "FAKE")) if is_image(image_path)]
        return items

    def __len__(self):
        return len(self.items)

    def read_image(self, path):
        image = Image.open(path).convert('RGB')
        image = T.Compose([
            T.Resize(self.resolution + self.resolution // 8, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(self.resolution),
            T.ToTensor(),
            # T.Normalize(mean=self.norm_mean, std=self.norm_std),
        ])(image)
        return image
    
    def __getitem__(self, i):
        sample = {
            "image_path": self.items[i]["image_path"],
            "image": self.read_image(self.items[i]["image_path"]),
            "is_real": torch.as_tensor([1 if self.items[i]["is_real"] is True else 0]),
        }
        return sample
    
    @staticmethod
    def _plot_image(image):
        import matplotlib.pyplot as plt
        import einops
        plt.imshow(einops.rearrange(image, "c h w -> h w c"))
        plt.show()
        plt.close()
        
    def _plot_labels_distribution(self, save_path=None):
        import matplotlib.pyplot as plt
        
        # Count the occurrences of each label
        label_counts = {"Real": 0, "Fake": 0}
        for item in self.items:
            if item["is_real"]:
                label_counts["Real"] += 1
            else:
                label_counts["Fake"] += 1
        
        # Data for plotting
        labels = list(label_counts.keys())
        counts = list(label_counts.values())
        
        # Creating the bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(labels, counts, color=['blue', 'orange'])
        
        plt.xlabel('Label')
        plt.ylabel('Count')
        plt.title(f'[DFFD] Distribution of labels for split {self.split}')
        plt.xticks(labels)
        plt.yticks(range(0, max(counts) + 1, max(counts) // 10))
        
        for i, v in enumerate(counts):
            plt.text(i, v + 0.5, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

if __name__=="__main__":
    import random
    dataset_path = "../../datasets/cifake"

    for split in ["train", "test"]:
        dataset = CIFAKEDataset(dataset_path=dataset_path, split=split, resolution=224)
        print(f"sample keys:", {k: (type(v) if not isinstance(v, torch.Tensor) else v.shape) for k, v in dataset[0].items()})
        dataset._plot_image(dataset[random.randint(0, len(dataset))]["image"])
        dataset._plot_labels_distribution(save_path=f"_{split}_labels_cifake.png")