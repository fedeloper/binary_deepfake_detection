**Model Setup**

- Download the BNext pretrained models from [here](https://github.com/hpi-xnor/BNext/tree/main). Place them in the `pretrained` folder, prefixing the filenames with "tiny_", "small_", "middle_", or "large_" accordingly.

## Data Preparation

### Using FF++ Dataset

1. **Dataset Acquisition**: Download FF++ dataset from [FF++](https://github.com/ondyari/FaceForensics). Store it in the `./data` directory.

    ```
    .
    └── data
        └── FaceForensics++
            ├── original_sequences
            │   └── youtube
            │       └── raw
            │           └── videos
            │               └── *.mp4
            ├── manipulated_sequences
            │   ├── Deepfakes
            │       └── raw
            │           └── videos
            │               └── *.mp4
            │   ├── Face2Face
            │       ...
            │   ├── FaceSwap
            │       ...
            │   ├── NeuralTextures
            │       ...
            │   ├── FaceShifter
            │       ...
    ```

2. **Landmark Detector**: Obtain the landmark detector from [this link](https://github.com/codeniko/shape_predictor_81_face_landmarks) and place it in the `./lib` folder.

3. **Frame Extraction**: Execute the following script to extract frames from FF++ videos. The frames should be saved in `./train_images` or `./test_images`, based on their categorization in the original dataset.

    ```
    python3 lib/extract_frames_ldm_ff++.py
    ```

**Training Command**

- To initiate training, run:

    ```
    python3 train.py --cfg ./configs/bin_caddm_train.cfg
    ```
