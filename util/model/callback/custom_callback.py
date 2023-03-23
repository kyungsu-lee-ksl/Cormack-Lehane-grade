import cv2
import platform
import numpy as np

from util.model import CallBack


class ImageCallBack(CallBack):

    def __init__(self, overlay_weight: float = 0.8, num_display_imgs: int = 10):
        super().__init__()
        self.__overlay_weight__ = overlay_weight
        self.__num_display_imgs__ = num_display_imgs

    def __on_batch_end__(self, current_epoch: int, current_batch: int, feed_dict: dict, log: list, tensors: list):
        imgs, gnds = tuple(map(lambda x: feed_dict[x], list(feed_dict.keys())))
        imgs = (imgs * 255).astype(np.uint8)
        sess, cam = tensors[0], tensors[-1]

        _cams = [cv2.applyColorMap((c * 255).astype(np.uint8), cv2.COLORMAP_JET) for c in sess.run(cam, feed_dict=feed_dict)]

        merged = np.vstack([np.hstack([imgs[i], _cams[i], cv2.addWeighted(imgs[i], 1.0, _cams[i], self.__overlay_weight__, 0.0)]) for i in range(min(len(imgs), self.__num_display_imgs__))])

        if platform.system() == 'Windows':
            cv2.imshow('result', merged)
            cv2.waitKey(1)
