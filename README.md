# Binary Deepfake Detection

## Setup

- Install Anaconda or Miniconda
- Create a new `conda` environment using `conda create --name dfad`
- Install the dependencies of the project using `pip install -r requirements.txt`
- Clone the BNext using ```git clone https://github.com/hpi-xnor/BNext.git```
- Download the BNext pretrained models from [here](https://github.com/hpi-xnor/BNext/tree/main). Place them in the `pretrained` folder, prefixing the filenames with "tiny_", "small_", "middle_", or "large_" accordingly.

## Running the code
To train a model, create a new configuration copying and editing the default ones in the `configs` folder. When you have created a new configuration, run the training using `python train.py --cfg <PATH_TO_YOUR_CONFIGURATION>`

To test the model, do the same as before but use the `test.py` script: `python test.py --cfg <PATH_TO_YOUR_CONFIGURATION>`
