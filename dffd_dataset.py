from os import listdir
from os.path import exists, isdir, join
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class DFFDDataset(Dataset):
    def __init__(self, dataset_path, split, resolution=224, norm_mean=IMAGENET_DEFAULT_MEAN, norm_std=IMAGENET_DEFAULT_STD):
        assert isdir(dataset_path), f"got {dataset_path}"
        self.dataset_path = dataset_path
        assert split in {"train", "val", "test"}, f"got {split}"
        self.split = split
        
        # parses metas from the datasets
        self.items = self.parse_datasets()
        
        # sets up the preprocessing options
        assert isinstance(resolution, int) and resolution >= 1, f"got {resolution}"
        self.resolution = resolution
        assert len(norm_mean) == 3
        self.norm_mean = norm_mean
        assert len(norm_std) == 3
        self.norm_std = norm_std

    def parse_datasets(self):
        def is_image(filename):
            for extension in ["jpg", "png", "jpeg"]:
                if filename.lower().endswith(extension):
                    return True
            return False
        
        def parse_celeba_metas(filename):
            if self.split == "train":
                split_index = 0
            elif self.split == "val":
                split_index = 1
            else:
                split_index = 2
            # reads the .txt line by line, separating each word by the space
            filenames = []
            with open(filename) as f:
                for line in f.readlines():
                    line = line.split()
                    if int(line[1]) != split_index:
                        continue
                    filenames.append(line[0])
            return filenames
        
        subdatasets = [
            {
                "name": subdataset,
                "is_real": True,
            } for subdataset in ["img_align_celeba", "ffhq"]
        ] + [
            {
                "name": subdataset,
                "is_real": False,
            } for subdataset in ["pggan_v1", "pggan_v2", "stylegan_ffhq", "stylegan_celeba", "faceapp", "stargan"]
        ]
        data = []
        for subdataset_item in subdatasets:
            subdataset_path = join(self.dataset_path, subdataset_item["name"])
            if not isdir(subdataset_path):
                raise f"subdataset {subdataset_path} not found"
            is_real = subdataset_item["is_real"]
            
            if subdataset_item["name"] == "img_align_celeba":
                # reads the metas
                celeba_metas = parse_celeba_metas(join(subdataset_path, "list_eval_partition.txt"))
                for filename in sorted(celeba_metas):
                    data.append({
                        "image_path": join(subdataset_path, filename),
                        "is_real": True,
                    })
            else:
                subdataset_split_path = join(subdataset_path, "validation" if self.split == "val" else self.split)
                assert f"no split folder found in {subdataset_path}"
                for filename in sorted(listdir(subdataset_split_path)):
                    if not is_image(filename):
                        continue
                    data.append({
                        "image_path": join(subdataset_split_path, filename),
                        "is_real": is_real,
                    })
                    assert exists(data[-1]["image_path"]), f"{data[-1]['image_path']} does not exists"
        return data

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
    dataset_path = "../../datasets/dffd"

    for split in {"train", "val", "test"}:
        dataset = DFFDDataset(dataset_path=dataset_path, split=split, resolution=224)
        print(f"sample keys:", {k: (type(v) if not isinstance(v, torch.Tensor) else v.shape) for k, v in dataset[0].items()})
        dataset._plot_image(dataset[random.randint(0, len(dataset))]["image"])
        dataset._plot_labels_distribution(save_path=f"_{split}_labels_dffd.png")