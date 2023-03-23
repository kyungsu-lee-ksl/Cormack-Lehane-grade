import os
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from util.preprocessing import read_dicom


class Directory:
    _image_path = 'dicom_files'
    _label_path = 'labels'
    _label_name = 'label.csv'

    def __init__(self, root, max_images=None):
        dicom_files = [name for name in os.listdir(f"{root}/{self._image_path}") if os.path.splitext(name)[-1].lower() == '.dcm']
        label_table = pd.read_csv(f"{root}/{self._label_path}/{self._label_name}")
        label_table['Patients'] = label_table['Patients'].astype(str)

        if max_images is not None:
            np.random.shuffle(dicom_files)
            dicom_files = dicom_files[:max_images]

        self.imgs, self.labels, self.patient = list(), list(), list()

        for name in tqdm(dicom_files, desc='Reading Files...'):
            _img = read_dicom(root=f"{root}/{self._image_path}", filename=name)

            if np.sum(_img) == 0 or self.get_class(name, label_table) < 0: continue

            self.labels.append(self.get_class(name, label_table))
            self.patient.append(self.get_patient(name))
            self.imgs.append(_img)

    def get_class(self, file_name: str, label_table: pd.DataFrame) -> int:
        try:
            return label_table[label_table['Patients'] == os.path.splitext(os.path.basename(file_name))[0]].to_numpy()[0][1]
        except Exception as ex:
            # print(ex)
            return -1

    def get_patient(self, file_name: str):
        return os.path.splitext(os.path.basename(file_name))[0]

    def __call__(self, ratio_train_val: float = 0.8, shuffle=True, image_callback=None, *args, **kwargs):

        ratio_train_val = min(1.0, max(0.0, ratio_train_val))

        if image_callback is not None:
            self.imgs = [image_callback(img) for img in self.imgs]
        self.imgs = np.asarray(self.imgs)
        self.labels = np.asarray(self.labels)

        if shuffle:
            indexes = np.asarray([i for i in range(len(self.labels))])
            np.random.shuffle(indexes)

            self.imgs = self.imgs[indexes]
            self.labels = self.labels[indexes]

        train_imgs, train_labels = tuple(map(lambda x: x[:int(len(self.labels) * ratio_train_val)], [self.imgs, self.labels]))
        valid_imgs, valid_labels = tuple(map(lambda x: x[int(len(self.labels) * ratio_train_val):], [self.imgs, self.labels]))

        return train_imgs, train_labels, valid_imgs, valid_labels

    @classmethod
    def lambda_for_train(cls, img):
        return np.expand_dims(cv2.resize(img, dsize=(256, 256)), axis=-1) / 255.
