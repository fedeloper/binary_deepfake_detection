# Faster Than Lies: Real-time Deepfake Detection using Binary Neural Networks
|[arXiv](https://arxiv.org/abs/2406.04932)|[Proceedings](https://openaccess.thecvf.com/content/CVPR2024W/DFAD/html/Lanzino_Faster_Than_Lies_Real-time_Deepfake_Detection_using_Binary_Neural_Networks_CVPRW_2024_paper.html)|[Weights](https://drive.google.com/drive/folders/1rYtfozcq5eXK1a8tP8ouXrBFZs1e72dV?usp=drive_link)|
| ------------- | ------------- | ------------- |

Welcome to the repository of the CVPRW24 paper.
The authors are Romeo Lanzino, Federico Fontana, Anxhelo Diko, Marco Raoul Marini, Luigi Cinque, from Sapienza University of Rome, Italy.

Here you can also find the presentation held on June 17th 2024, alongside [the prompt](https://chatgpt.com/share/983d40c4-cc5b-498d-acea-4e02643d49a2) used to generate it.

## Setup

- Install Anaconda or Miniconda
- Create a new `conda` environment using `conda create --name dfad`
- Install the dependencies of the project using `pip install -r requirements.txt`
- Clone the BNext using ```git clone https://github.com/hpi-xnor/BNext.git```
- Download the BNext pretrained models from [here](https://github.com/hpi-xnor/BNext/tree/main). Place them in the `pretrained` folder, prefixing the filenames with "tiny_", "small_", "middle_", or "large_" accordingly.

## Running the code
To train a model, create a new configuration copying and editing the default ones in the `configs` folder. When you have created a new configuration, run the training using `python train.py --cfg <PATH_TO_YOUR_CONFIGURATION>`

To test the model, do the same as before but use the `test.py` script: `python test.py --cfg <PATH_TO_YOUR_CONFIGURATION>`. Weights can be downloaded from [here](https://drive.google.com/drive/folders/1rYtfozcq5eXK1a8tP8ouXrBFZs1e72dV?usp=drive_link).

## Citing
If you want to cite our paper, please do it using the following format:

```bibtex
@InProceedings{Lanzino_2024_CVPR,
    author    = {Lanzino, Romeo and Fontana, Federico and Diko, Anxhelo and Marini, Marco Raoul and Cinque, Luigi},
    title     = {Faster Than Lies: Real-time Deepfake Detection using Binary Neural Networks},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
    pages     = {3771-3780}
}
```
