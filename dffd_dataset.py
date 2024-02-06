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
        self.metas = self.parse_datasets()
        print(len(self.metas))
        raise
        # real_images_paths = sorted(list({self.metas[i]["real_image_path"] for i in range(len(self.metas))}))
        # fake_images_paths = sorted([self.metas[i]["fake_image_path"] for i in range(len(self.metas))])
        # self.items = [{"image_path": image_path, "is_real": True} for image_path in real_images_paths] + [{"image_path": image_path, "is_real": False} for image_path in fake_images_paths]

        assert isinstance(resolution, int) and resolution >= 1, f"got {resolution}"
        self.resolution = resolution
        assert len(norm_mean) == 3
        self.norm_mean = norm_mean
        assert len(norm_std) == 3
        self.norm_std = norm_std

    def parse_datasets(self):
        def is_image(filename):
            ok = False
            for extension in ["jpg", "png", "jpeg"]:
                if filename.lower().endswith(extension):
                    ok = True
                    break
            return ok
        
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
                continue
                # raise NotImplementedError()
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
        print(len(data))
        raise
        # split_path = join(self.coco_fake_path, f"{self.split}2014")
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
        image = T.Compose([
            T.Resize(self.resolution + self.resolution // 8, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(self.resolution),
            T.ToTensor(),
            T.Normalize(mean=self.norm_mean, std=self.norm_std),
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
        plt.imshow(image.transpose(0, 2))
        plt.show()
        plt.close()


if __name__=="__main__":
    dataset_path = "../../datasets/dffd"

    dataset = DFFDDataset(dataset_path=dataset_path, split="train", resolution=224)
    print(f"sample keys:", {k: (type(v) if not isinstance(v, torch.Tensor) else v.shape) for k, v in dataset[0].items()})
    dataset._plot_image(dataset[0]["fake_image"])