"""Inference-only subset; unused upstream utilities have been removed.
"""

import numpy as np
import cv2

def crop_image_and_adjust_intrinsics(K, input_h, input_w, aspect_ratio):
    """
    计算满足宽高比的裁剪坐标，并调整内参矩阵。
    
    :param K: 3x3 内参矩阵 (numpy array)
    :param input_h: 输入图像高度
    :param input_w: 输入图像宽度
    :param aspect_ratio: 目标宽高比 (w/h)
    :return: (crop_x1, crop_y1, crop_x2, crop_y2), new_K
    """
    input_ratio = input_w / input_h

    if input_ratio > aspect_ratio:
        # 需要裁剪宽度
        new_w = int(input_h * aspect_ratio)
        new_h = input_h
        crop_x1 = (input_w - new_w) // 2
        crop_x2 = crop_x1 + new_w
        crop_y1 = 0
        crop_y2 = input_h
    else:
        # 需要裁剪高度
        new_h = int(input_w / aspect_ratio)
        new_w = input_w
        crop_y1 = (input_h - new_h) // 2
        crop_y2 = crop_y1 + new_h
        crop_x1 = 0
        crop_x2 = input_w

    # 计算新的内参矩阵
    new_K = K.copy()
    new_K[0, 2] -= crop_x1  # 调整 cx
    new_K[1, 2] -= crop_y1  # 调整 cy

    return (crop_x1, crop_y1, crop_x2, crop_y2), new_K


def crop_and_resize(K, image, input_h, input_w, target_h, target_w):
    """
    裁剪图像并调整内参矩阵，以满足目标宽高比。
    
    :param K: 3x3 内参矩阵 (numpy array)
    :param image: 输入图像 (numpy array, shape [3, H, W])
    :param input_h: 输入图像高度
    :param input_w: 输入图像宽度
    :param target_h: 目标图像高度
    :param target_w: 目标图像宽度
    :return: 裁剪后的图像 (numpy array, shape [3, target_h, target_w])
    """
    aspect_ratio = target_w / target_h
    (crop_x1, crop_y1, crop_x2, crop_y2), new_K = crop_image_and_adjust_intrinsics(K, input_h, input_w, aspect_ratio)

    cropped_image = image[:, crop_y1:crop_y2, crop_x1:crop_x2].astype(np.uint8).transpose(1, 2, 0)
    resized_image = cv2.resize(cropped_image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    resized_image = resized_image.transpose(2, 0, 1).astype(np.float32)

    intrinsic_w_ratio = target_w / (crop_x2 - crop_x1)
    intrinsic_h_ratio = target_h / (crop_y2 - crop_y1)
    new_K[0, 0] *= intrinsic_w_ratio
    new_K[1, 1] *= intrinsic_h_ratio
    new_K[0, 2] *= intrinsic_w_ratio
    new_K[1, 2] *= intrinsic_h_ratio
    
    return resized_image, new_K
