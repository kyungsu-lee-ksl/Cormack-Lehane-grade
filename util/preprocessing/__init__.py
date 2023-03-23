import numpy as np
import pydicom
import os


def read_dicom(root: str, filename: str):
    ds = pydicom.read_file(os.path.join(root, filename))
    pixel_array = ds.pixel_array  # dicom image

    if ('RescaleSlope' in ds) and ('RescaleIntercept' in ds):
        pixel_array = (pixel_array * ds.RescaleSlope) + ds.RescaleIntercept

    if 'WindowCenter' in ds:
        if type(ds.WindowCenter) == pydicom.multival.MultiValue:
            window_center = float(ds.WindowCenter[0])
            window_width = float(ds.WindowWidth[0])
            lwin = window_center - (window_width / 2.0)
            rwin = window_center + (window_width / 2.0)
        else:
            window_center = float(ds.WindowCenter)
            window_width = float(ds.WindowWidth)
            lwin = window_center - (window_width / 2.0)
            rwin = window_center + (window_width / 2.0)
    else:
        lwin = np.min(pixel_array)
        rwin = np.max(pixel_array)

    pixel_array[np.where(pixel_array < lwin)] = lwin
    pixel_array[np.where(pixel_array > rwin)] = rwin
    pixel_array = pixel_array - lwin

    if ds.PhotometricInterpretation == 'MONOCHROME1':
        pixel_array[np.where(pixel_array < lwin)] = lwin
        pixel_array[np.where(pixel_array > rwin)] = rwin
        pixel_array = pixel_array - lwin
        pixel_array = 1.0 - pixel_array

    else:
        pixel_array[np.where(pixel_array < lwin)] = lwin
        pixel_array[np.where(pixel_array > rwin)] = rwin
        pixel_array = pixel_array - lwin

    # normalization
    pixel_array = ((pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array)) * 255).astype(np.uint8)

    return pixel_array

