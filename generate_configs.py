import argparse
from os import makedirs
from os.path import isdir, join
import yaml


def args_func():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        help="The path where to save the configs.",
        default="./configs",
    )
    args = parser.parse_args()
    return args


def update_dict_recursively(original, updates):
    for key, value in updates.items():
        if (
            key in original
            and isinstance(original[key], dict)
            and isinstance(value, dict)
        ):
            update_dict_recursively(original[key], value)
        else:
            original[key] = value


if __name__ == "__main__":
    args = args_func()
    if not isdir(args.path):
        makedirs(args.path)
    # creates a .gitkeep file
    with open(join(args.path, ".gitkeep"), "w") as fp:
        fp.write("")

    # define the template here
    configs_template = {
        "dataset": {
            "name": "coco_fake",
            "coco2014_path": "../../datasets/coco2014",
            "coco_fake_path": "../../datasets/fake_coco",
            "cifake_path": "../../datasets/cifake",
            "dffd_path": "../../datasets/dffd",
            "labels": 2,
        },
        "model": {
            "add_magnitude_channel": False,
            "add_fft_channel": False,
            "add_lbp_channel": False,
            "freeze_backbone": True,
            "backbone": "BNext-T",
        },
        "train": {
            "batch_size": 32,
            "accumulation_batches": 4,
            "mixed_precision": True,
            "epoch_num": 5,
            "limit_train_batches": 1.0,
            "limit_val_batches": 1.0,
            "resolution": 224,
            "seed": 5,
        },
        "test": {
            "weights_path": "./weights",
            "batch_size": 128,
            "mixed_precision": True,
            "limit_test_batches": 1.0,
            "resolution": 224,
            "seed": 5,
        },
    }

    configs = []
    # ablation configs
    configs = [
        {"name": name, "changes": changes}
        for name, changes in [
            ("ablation_baseline", {"model": {}}),
            (
                "ablation_mag",
                {
                    "model": {
                        "add_magnitude_channel": True,
                    }
                },
            ),
            (
                "ablation_fft",
                {
                    "model": {
                        "add_fft_channel": True,
                    }
                },
            ),
            (
                "ablation_lbp",
                {
                    "model": {
                        "add_lbp_channel": True,
                    }
                },
            ),
            (
                "ablation_mag_fft",
                {
                    "model": {
                        "add_magnitude_channel": True,
                        "add_fft_channel": True,
                    }
                },
            ),
            (
                "ablation_mag_lbp",
                {
                    "model": {
                        "add_magnitude_channel": True,
                        "add_lbp_channel": True,
                    }
                },
            ),
            (
                "ablation_fft_lbp",
                {
                    "model": {
                        "add_fft_channel": True,
                        "add_lbp_channel": True,
                    }
                },
            ),
            (
                "ablation_mag_fft_lbp",
                {
                    "model": {
                        "add_magnitude_channel": True,
                        "add_fft_channel": True,
                        "add_lbp_channel": True,
                    }
                },
            ),
        ]
    ]
    # results
    for dataset in ["coco_fake", "dffd", "cifake"]:
        for model_name in ["T", "S", "M", "L"]:
            for freeze_backbone in [True, False]:
                configs += [
                    {
                        "name": f"results_{dataset}_{model_name}{'_unfrozen' if not freeze_backbone else ''}",
                        "changes": {
                            "dataset": {"name": dataset},
                            "model": {
                                "add_fft_channel": True,
                                "add_lbp_channel": True,
                                "backbone": f"BNext-{model_name}",
                                "freeze_backbone": freeze_backbone,
                            },
                            "train": {
                                "accumulation_batches": 4,
                                "batch_size": 32,
                                "epoch_num": 5 if dataset in {"coco_fake", "dffd"} else 10,
                            }
                        },
                    }
                ]
    for config in configs:
        with open(join(args.path, f"{config['name']}.cfg"), "w") as fp:
            from copy import deepcopy

            current_config = deepcopy(configs_template)
            update_dict_recursively(current_config, config["changes"])
            documents = yaml.dump(current_config, fp)
