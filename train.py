import argparse

import cv2

from util.directory import Directory
from util.model import Model
from util.model.callback.custom_callback import ImageCallBack

parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--path', type=str, default='./dataset', help='Path to the dataset')
parser.add_argument('--input_size', type=int, nargs=2, default=[256, 256], help='Input size of the images')
parser.add_argument('--batch_size', type=int, default=15, help='Batch size for training')
parser.add_argument('--total_epoch', type=int, default=100, help='Total number of epochs')
parser.add_argument('--save_path', type=str, default='./weights', help='Path to save the trained model')
parser.add_argument('--save_interval', type=int, default=1, help='Interval between model checkpoints')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
parser.add_argument('--load_path', type=str, default=None, help='Path to load the weights of a pre-trained model')


if __name__ == '__main__':

    args = parser.parse_args()

    path = args.path
    input_size = tuple(args.input_size)
    batch_size = args.batch_size
    total_epoch = args.total_epoch
    save_path = args.save_path
    save_interval = args.save_interval
    learning_rate = args.learning_rate
    load_path = args.load_path

    train_imgs, train_labels, valid_imgs, valid_labels = Directory(root=path)(shuffle=False, image_callback=lambda img: cv2.cvtColor(cv2.resize(img, dsize=input_size), cv2.COLOR_GRAY2BGR) / 255.)

    model = Model(input_shape=(*input_size, 3), batch_size=batch_size)
    model.compile(
        # load_path=load_path,
        learning_rate=learning_rate
    )

    model.train(
        total_epoch=total_epoch,
        trainset=[train_imgs, train_labels],
        validationset=[valid_imgs, valid_labels],
        save_path=save_path,
        save_interval=save_interval,
        callbacks=[ImageCallBack()],
    )
