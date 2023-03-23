# Deep-learning model associating lateral cervical radiographic features with Cormack-Lehane grade 3 or 4 glottic view

## Overview

This repository contains two Python scripts for a deep-learning model:

- `train.py`: A script for training the model on a dataset.
- `test.py`: A script for testing the trained model on a DICOM file and generating a class activation map (CAM) with an overlay on the input image.

## Dependencies

This project requires the following dependencies to be installed:

- Python 3.7+
- OpenCV
- Numpy
- Tensorflow 1.14
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


##

<br>
<br>

## [PAPER] Deep-learning model associating lateral cervical radiographic features with Cormack–Lehane grade 3 or 4 glottic view

#### Authors
H.-Y.Cho, K.Lee. H-J.Kong, H.L-Yang. C.W. Jung, H.-P.Park, J.Y.Hwang, and H.-C.Lee

#### Summary
Unanticipated difﬁcult laryngoscopy is associated with serious airway-related complications. We aimed to develop and test a convolutional neural network-based deep-learning model that uses lateral cervical spine radiographs to predict Cormack–Lehane grade 3 or 4 direct laryngoscopy views of the glottis. We analysed the radiographs of 5939 thyroid surgery patients at our hospital, 253 (4%) of whom had grade 3 or 4 glottic views. We used 10 randomly sampled datasets to train a model. We compared the new model with six similar models (VGG, ResNet, Xception, ResNext, DenseNet and SENet). The Brier score (95%CI) of the new model, 0.023 (0.021–0.025), was lower (`better´) than the other models: VGG, 0.034 (0.034–0.035); ResNet, 0.033 (0.033–0.035); Xception, 0.032 (0.031–0.033); ResNext, 0.033 (0.032–0.033); DenseNet, 0.030 (0.029–0.032); SENet, 0.031 (0.029–0.032), all p < 0.001. We calculated mean (95%CI) of the new model for: R2, 0.428 (0.388–0.468); mean squared error, 0.023 (0.021–0.025); mean absolute error, 0.048 (0.046–0.049); balanced accuracy, 0.713 (0.684–0.742); and area under the receiver operating characteristic curve, 0.965 (0.962–0.969). Radiographic features around the hyoid bone, pharynx and cervical spine were associated with grade 3 and 4 glottic views.



#### Main Architecture

<p align="center">
    <img src="assets/Fig2.tif" width="80%"\>
</p>

Figure. Convolutional neural network-based deep-learning model with a convolutional layer, pooling layer, self-attention layer, and final fully connected layer to predict difficult laryngoscopy.


## Citation

If you find our work useful in your research, please consider citing our paper:

```
@article{cho2023deep,
  title={Deep-learning model associating lateral cervical radiographic features with Cormack--Lehane grade 3 or 4 glottic view},
  author={Cho, H-Y and Lee, Kyungsu and Kong, H-J and Yang, H-L and Jung, C-W and Park, H-P and Hwang, Jae Youn and Lee, H-C},
  journal={Anaesthesia},
  volume={78},
  number={1},
  pages={64--72},
  year={2023},
  publisher={Wiley Online Library}
}
```
