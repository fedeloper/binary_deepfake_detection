from os import listdir
from os.path import exists, isdir, join
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

class COCOFakeDataset(Dataset):
    def __init__(self, coco2014_path, coco_fake_path, split, mode="single", resolution=224, norm_mean=[0.5, 0.5, 0.5], norm_std=[0.5, 0.5, 0.5]):
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
        assert len(norm_mean) == 3
        self.norm_mean = norm_mean
        assert len(norm_std) == 3
        self.norm_std = norm_std

    def parse_datasets(self):
        data = []
        split_path = join(self.coco_fake_path, f"{self.split}2014")
        for folder in sorted(listdir(split_path)):
            for filename in sorted(listdir(join(split_path, folder))):
                if not filename.endswith(".jpg"):
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
        image = T.Compose([
            T.Resize(self.resolution + self.resolution // 8, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(self.resolution),
            T.ToTensor(),
            T.Normalize(mean=self.norm_mean, std=self.norm_std),
        ])(image)
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
        plt.imshow(image.transpose(0, 2))
        plt.show()
        plt.close()


if __name__=="__main__":
    coco2014_path = "../../datasets/coco2014"
    coco_fake_path = "../../datasets/fake_coco"
    for mode in ["single", "couple"]:
        dataset = COCOFakeDataset(coco2014_path=coco2014_path, coco_fake_path=coco_fake_path, split="train", mode=mode, resolution=224)
        print(f"sample keys for mode {mode}:", {k: (type(v) if not isinstance(v, torch.Tensor) else v.shape) for k, v in dataset[0].items()})
    dataset._plot_image(dataset[0]["fake_image"])