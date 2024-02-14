from os import listdir
from os.path import exists, isdir, join
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
from PIL import Image

class COCOFakeDataset(Dataset):
    def __init__(self, coco2014_path, coco_fake_path, split, mode="single", resolution=224):
        assert isdir(coco2014_path), f"got {coco2014_path}"
        assert isdir(coco_fake_path), f"got {coco_fake_path}"
        self.coco2014_path = coco2014_path
        self.coco_fake_path = coco_fake_path
        assert split in {"train", "val"}, f"got {split}"
        self.split = split
        
        # parses metas from both COCO and COCO-Fake
        self.metas = self.parse_datasets()

        assert mode in {"single", "couple"}, f"got {mode}"
        self.mode = mode
        if mode == "single":
            # tuples here have shape:
            # {
            #     "image_path": ...,
            #     "is_real": ...,
            # }
            real_images_paths = sorted(list({self.metas[i]["real_image_path"] for i in range(len(self.metas))}))
            fake_images_paths = sorted([self.metas[i]["fake_image_path"] for i in range(len(self.metas))])
            self.items = [{"image_path": image_path, "is_real": True} for image_path in real_images_paths] + [{"image_path": image_path, "is_real": False} for image_path in fake_images_paths]
        elif mode == "couple":
            # tuples here have shape:
            # {
            #     "fake_image_path": ...,
            #     "real_image_path": ...,
            # }
            self.items = self.metas

        assert isinstance(resolution, int) and resolution >= 1, f"got {resolution}"
        self.resolution = resolution

    def parse_datasets(self):
        data = []
        split_path = join(self.coco_fake_path, f"{self.split}2014")
        for folder in sorted(listdir(split_path)):
            for filename in sorted(listdir(join(split_path, folder))):
                if not filename.lower().endswith(".jpg"):
                    continue
                data.append({
                    "fake_image_path": join(split_path, folder, filename),
                    "real_image_path": join(self.coco2014_path, f"{self.split}2014", f"{folder}.jpg"),
                })
                assert exists(data[-1]["fake_image_path"]), f"{data[-1]['fake_image_path']} does not exists"
                assert exists(data[-1]["real_image_path"]), f"{data[-1]['real_image_path']} does not exists"
        return data

    def __len__(self):
        return len(self.items)

    def read_image(self, path):
        image = Image.open(path).convert('RGB')
        if self.split == "train":
            transforms = T.Compose([
                T.Resize(self.resolution + self.resolution // 8, interpolation=T.InterpolationMode.BILINEAR),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomChoice([
                    T.RandomRotation(degrees=(-90, -90)),
                    T.RandomRotation(degrees=(90, 90)),
                    ], p=[0.5, 0.5]),
                T.RandomCrop(self.resolution),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                ])
        else:
            transforms = T.Compose([
                T.Resize(self.resolution + self.resolution // 8, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop(self.resolution),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
            ])
        image = transforms(image)
        return image
    
    def __getitem__(self, i):
        if self.mode == "single":
            sample = {
                "image_path": self.items[i]["image_path"],
                "image": self.read_image(self.items[i]["image_path"]),
                "is_real": torch.as_tensor([1 if self.items[i]["is_real"] is True else 0]),
            }
        elif self.mode == "couple":
            sample = {
                "fake_image_path": self.items[i]["fake_image_path"],
                "fake_image": self.read_image(self.items[i]["fake_image_path"]),
                "real_image_path": self.items[i]["real_image_path"],
                "real_image": self.read_image(self.items[i]["real_image_path"]),
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
        if self.mode == "single":
            for item in self.items:
                if item["is_real"]:
                    label_counts["Real"] += 1
                else:
                    label_counts["Fake"] += 1
        else:
            label_counts["Fake"] = len(self.items)
            label_counts["Real"] = len({item["real_image_path"] for item in self.items})
        
        # Data for plotting
        labels = list(label_counts.keys())
        counts = list(label_counts.values())
        
        # Creating the bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(labels, counts, color=['blue', 'orange'])
        
        plt.xlabel('Label')
        plt.ylabel('Count')
        plt.title(f'[COCO-Fake] Distribution of labels for split {self.split}')
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
    
    coco2014_path = "../../datasets/coco2014"
    coco_fake_path = "../../datasets/fake_coco"
    for split in ["train", "val"]:
        for mode in ["single", "couple"]:
            dataset = COCOFakeDataset(coco2014_path=coco2014_path, coco_fake_path=coco_fake_path, split=split, mode=mode, resolution=224)
            print(f"sample keys for mode {mode}:", {k: (type(v) if not isinstance(v, torch.Tensor) else v.shape) for k, v in dataset[0].items()})
        dataset._plot_image(dataset[random.randint(0, len(dataset))]["fake_image"])
        dataset._plot_labels_distribution(save_path=f"_{split}_labels_cocofake.png")