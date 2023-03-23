# Deep-learning model associating lateral cervical radiographic features with Cormack-Lehane grade 3 or 4 glottic view

## Overview

This repository contains two Python scripts for a deep-learning model:

- `train.py`: A script for training the model on a dataset.
- `test.py`: A script for testing the trained model on a DICOM file and generating a class activation map (CAM) with an overlay on the input image.

## Dependencies

This project requires the following dependencies to be installed:

- Python 3.x
- OpenCV
- Numpy
- Tensorflow
- argparse

## Usage

### `train.py`

This script trains the model on a given dataset.

#### Usage

```
python train.py --path <path_to_dataset> --input_size <input_size> <input_size> --batch_size <batch_size> --total_epoch <total_epoch> --save_path <save_path> --save_interval <save_interval> --learning_rate <learning_rate> --load_path <load_path>
```

#### Parameters

| Parameter | Description | Default Value |
| --------- | ----------- | ------------- |
| `--path` | Path to the dataset | `./dataset` |
| `--input_size` | Input size of the images | `256 256` |
| `--batch_size` | Batch size for training | `15` |
| `--total_epoch` | Total number of epochs | `100` |
| `--save_path` | Path to save the trained model | `./weights` |
| `--save_interval` | Interval between model checkpoints | `1` |
| `--learning_rate` | Learning rate for the optimizer | `1e-3` |
| `--load_path` | Path to load the weights of a pre-trained model | `None` |

### `test.py`

This script tests the trained model on a DICOM file and generates a class activation map (CAM) with an overlay on the input image.

#### Usage

```
python test.py <dicom_file_path> --input_size <input_size> <input_size> --load_path <load_path> --overlay_weight <overlay_weight>
```

#### Parameters

| Parameter | Description | Default Value |
| --------- | ----------- | ------------- |
| `dicom_file_path` | Path to the DICOM file | - |
| `--input_size` | Input size of the images | `256 256` |
| `--load_path` | Path to load the trained model | `./weights` |
| `--overlay_weight` | Weight of the CAM overlay on the image | `0.8` |

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more information.
