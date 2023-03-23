import argparse
import cv2

from util.model import Model

parser = argparse.ArgumentParser(description='Predict a label and CAM for a DICOM file')
parser.add_argument('dicom_file_path', type=str, help='Path to the DICOM file')
parser.add_argument('--input_size', type=int, nargs=2, default=[256, 256], help='Input size of the images')
parser.add_argument('--load_path', type=str, default='./weights', help='Path to load the trained model')
parser.add_argument('--overlay_weight', type=float, default=0.8, help='Weight of the CAM overlay on the image')

args = parser.parse_args()

if __name__ == '__main__':
    dicom_file_path = args.dicom_file_path
    input_size = tuple(args.input_size)
    load_path = args.load_path
    overlay_weight = args.overlay_weight

    image = cv2.imread(dicom_file_path, cv2.IMREAD_COLOR)

    model = Model(input_shape=(*input_size, 3), batch_size=1)
    model.compile(
        load_path=load_path,
    )

    predicted_label, predicted_cam = model.predict(
        image=image
    )

    overlay_cam = cv2.addWeighted(image, 1.0, predicted_cam, overlay_weight, 0.0)
