import math
import random
from copy import deepcopy
from typing import Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics.data.augment import BaseMixTransform, Compose, Albumentations, RandomFlip, RandomPerspective, LetterBox

from ultralytics.data.utils import polygons2masks, polygons2masks_overlap
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.checks import check_version
from ultralytics.utils.instance import Instances
from ultralytics.utils.metrics import bbox_ioa
from ultralytics.utils.ops import segment2box, xyxyxyxy2xywhr
from ultralytics.utils.torch_utils import TORCHVISION_0_10, TORCHVISION_0_11, TORCHVISION_0_13

DEFAULT_MEAN = (0.0, 0.0, 0.0)
DEFAULT_STD = (1.0, 1.0, 1.0)
DEFAULT_CROP_FRACTION = 1.0

class Format:
    """
    A class for formatting image annotations for object detection, instance segmentation, and pose estimation tasks.

    This class standardizes image and instance annotations to be used by the `collate_fn` in PyTorch DataLoader.

    Attributes:
        bbox_format (str): Format for bounding boxes. Options are 'xywh' or 'xyxy'.
        normalize (bool): Whether to normalize bounding boxes.
        return_mask (bool): Whether to return instance masks for segmentation.
        return_keypoint (bool): Whether to return keypoints for pose estimation.
        return_obb (bool): Whether to return oriented bounding boxes.
        mask_ratio (int): Downsample ratio for masks.
        mask_overlap (bool): Whether to overlap masks.
        batch_idx (bool): Whether to keep batch indexes.
        bgr (float): The probability to return BGR images.

    Methods:
        __call__: Formats labels dictionary with image, classes, bounding boxes, and optionally masks and keypoints.
        _format_img: Converts image from Numpy array to PyTorch tensor.
        _format_segments: Converts polygon points to bitmap masks.

    Examples:
        >>> formatter = Format(bbox_format='xywh', normalize=True, return_mask=True)
        >>> formatted_labels = formatter(labels)
        >>> img = formatted_labels['img']
        >>> bboxes = formatted_labels['bboxes']
        >>> masks = formatted_labels['masks']
    """

    def __init__(
        self,
        bbox_format="xywh",
        normalize=True,
        return_mask=False,
        return_keypoint=False,
        return_obb=False,
        mask_ratio=4,
        mask_overlap=True,
        batch_idx=True,
        bgr=0.0,
    ):
        """
        Initializes the Format class with given parameters for image and instance annotation formatting.

        This class standardizes image and instance annotations for object detection, instance segmentation, and pose
        estimation tasks, preparing them for use in PyTorch DataLoader's `collate_fn`.

        Args:
            bbox_format (str): Format for bounding boxes. Options are 'xywh', 'xyxy', etc.
            normalize (bool): Whether to normalize bounding boxes to [0,1].
            return_mask (bool): If True, returns instance masks for segmentation tasks.
            return_keypoint (bool): If True, returns keypoints for pose estimation tasks.
            return_obb (bool): If True, returns oriented bounding boxes.
            mask_ratio (int): Downsample ratio for masks.
            mask_overlap (bool): If True, allows mask overlap.
            batch_idx (bool): If True, keeps batch indexes.
            bgr (float): Probability of returning BGR images instead of RGB.

        Attributes:
            bbox_format (str): Format for bounding boxes.
            normalize (bool): Whether bounding boxes are normalized.
            return_mask (bool): Whether to return instance masks.
            return_keypoint (bool): Whether to return keypoints.
            return_obb (bool): Whether to return oriented bounding boxes.
            mask_ratio (int): Downsample ratio for masks.
            mask_overlap (bool): Whether masks can overlap.
            batch_idx (bool): Whether to keep batch indexes.
            bgr (float): The probability to return BGR images.

        Examples:
            >>> format = Format(bbox_format='xyxy', return_mask=True, return_keypoint=False)
            >>> print(format.bbox_format)
            xyxy
        """
        self.bbox_format = bbox_format
        self.normalize = normalize
        self.return_mask = return_mask  # set False when training detection only
        self.return_keypoint = return_keypoint
        self.return_obb = return_obb
        self.mask_ratio = mask_ratio
        self.mask_overlap = mask_overlap
        self.batch_idx = batch_idx  # keep the batch indexes
        self.bgr = bgr

    def __call__(self, labels):
        """
        Formats image annotations for object detection, instance segmentation, and pose estimation tasks.

        This method standardizes the image and instance annotations to be used by the `collate_fn` in PyTorch
        DataLoader. It processes the input labels dictionary, converting annotations to the specified format and
        applying normalization if required.

        Args:
            labels (Dict): A dictionary containing image and annotation data with the following keys:
                - 'img': The input image as a numpy array.
                - 'cls': Class labels for instances.
                - 'instances': An Instances object containing bounding boxes, segments, and keypoints.

        Returns:
            (Dict): A dictionary with formatted data, including:
                - 'img': Formatted image tensor.
                - 'cls': Class labels tensor.
                - 'bboxes': Bounding boxes tensor in the specified format.
                - 'masks': Instance masks tensor (if return_mask is True).
                - 'keypoints': Keypoints tensor (if return_keypoint is True).
                - 'batch_idx': Batch index tensor (if batch_idx is True).

        Examples:
            >>> formatter = Format(bbox_format='xywh', normalize=True, return_mask=True)
            >>> labels = {'img': np.random.rand(640, 640, 3), 'cls': np.array([0, 1]), 'instances': Instances(...)}
            >>> formatted_labels = formatter(labels)
            >>> print(formatted_labels.keys())
        """
        img = labels.pop("img")
        h, w = img.shape[:2]
        cls = labels.pop("cls")
        instances = labels.pop("instances")
        instances.convert_bbox(format=self.bbox_format)
        instances.denormalize(w, h)
        nl = len(instances)

        if self.return_mask:
            if nl:
                masks, instances, cls = self._format_segments(instances, cls, w, h)
                masks = torch.from_numpy(masks)
            else:
                masks = torch.zeros(
                    1 if self.mask_overlap else nl, img.shape[0] // self.mask_ratio, img.shape[1] // self.mask_ratio
                )
            labels["masks"] = masks
        labels["img"] = self._format_img(img)
        labels["cls"] = torch.from_numpy(cls) if nl else torch.zeros(nl)
        labels["bboxes"] = torch.from_numpy(instances.bboxes) if nl else torch.zeros((nl, 4))
        if self.return_keypoint:
            labels["keypoints"] = torch.from_numpy(instances.keypoints)
            if self.normalize:
                labels["keypoints"][..., 0] /= w
                labels["keypoints"][..., 1] /= h
        if self.return_obb:
            labels["bboxes"] = (
                xyxyxyxy2xywhr(torch.from_numpy(instances.segments)) if len(instances.segments) else torch.zeros((0, 5))
            )
        # NOTE: need to normalize obb in xywhr format for width-height consistency
        if self.normalize:
            labels["bboxes"][:, [0, 2]] /= w
            labels["bboxes"][:, [1, 3]] /= h
        # Then we can use collate_fn
        if self.batch_idx:
            labels["batch_idx"] = torch.zeros(nl)
        return labels

    def _format_img(self, img):
        """
        Formats an image for YOLO from a Numpy array to a PyTorch tensor.

        This function performs the following operations:
        1. Ensures the image has 3 dimensions (adds a channel dimension if needed).
        2. Transposes the image from HWC to CHW format.
        3. Optionally flips the color channels from RGB to BGR.
        4. Converts the image to a contiguous array.
        5. Converts the Numpy array to a PyTorch tensor.

        Args:
            img (np.ndarray): Input image as a Numpy array with shape (H, W, C) or (H, W).

        Returns:
            (torch.Tensor): Formatted image as a PyTorch tensor with shape (C, H, W).

        Examples:
            >>> import numpy as np
            >>> img = np.random.rand(100, 100, 3)
            >>> formatted_img = self._format_img(img)
            >>> print(formatted_img.shape)
            torch.Size([3, 100, 100])
        """
        # if len(img.shape) < 3:
        #     img = np.expand_dims(img, -1)
        # img = img.transpose(2, 0, 1)
        # img = np.ascontiguousarray(img[::-1] if random.uniform(0, 1) > self.bgr else img)
        # img = torch.from_numpy(img)
        # return img
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        if (img.shape[2] == 1):
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
        elif (img.shape[2] == 4):
            img3c = np.ascontiguousarray(img.transpose(2, 0, 1)[:3, :, :][::-1])
            img1c = img.transpose(2, 0, 1)[-1:, :, :]
            img = np.concatenate((img3c, img1c), axis=0)
            # img = torch.from_numpy(img)
            # ----------------------------   3 _format_img
        elif (img.shape[2] == 6):
            img3c = np.ascontiguousarray(img.transpose(2, 0, 1)[:3, :, :][::-1])
            img3c2 =  np.ascontiguousarray(img.transpose(2, 0, 1)[3:, :, :][::-1])
            img = np.concatenate((img3c, img3c2), axis=0)
        else:
            img = np.ascontiguousarray(img.transpose(2, 0, 1)[::-1])
        img = torch.from_numpy(img)
        return img

    def _format_segments(self, instances, cls, w, h):
        """
        Converts polygon segments to bitmap masks.

        Args:
            instances (Instances): Object containing segment information.
            cls (numpy.ndarray): Class labels for each instance.
            w (int): Width of the image.
            h (int): Height of the image.

        Returns:
            (tuple): Tuple containing:
                masks (numpy.ndarray): Bitmap masks with shape (N, H, W) or (1, H, W) if mask_overlap is True.
                instances (Instances): Updated instances object with sorted segments if mask_overlap is True.
                cls (numpy.ndarray): Updated class labels, sorted if mask_overlap is True.

        Notes:
            - If self.mask_overlap is True, masks are overlapped and sorted by area.
            - If self.mask_overlap is False, each mask is represented separately.
            - Masks are downsampled according to self.mask_ratio.
        """
        segments = instances.segments
        if self.mask_overlap:
            masks, sorted_idx = polygons2masks_overlap((h, w), segments, downsample_ratio=self.mask_ratio)
            masks = masks[None]  # (640, 640) -> (1, 640, 640)
            instances = instances[sorted_idx]
            cls = cls[sorted_idx]
        else:
            masks = polygons2masks((h, w), segments, color=1, downsample_ratio=self.mask_ratio)

        return masks, instances, cls

class Mosaic(BaseMixTransform):
    """
    Mosaic augmentation.

    This class performs mosaic augmentation by combining multiple (4 or 9) images into a single mosaic image.
    The augmentation is applied to a dataset with a given probability.

    Attributes:
        dataset: The dataset on which the mosaic augmentation is applied.
        imgsz (int, optional): Image size (height and width) after mosaic pipeline of a single image. Default to 640.
        p (float, optional): Probability of applying the mosaic augmentation. Must be in the range 0-1. Default to 1.0.
        n (int, optional): The grid size, either 4 (for 2x2) or 9 (for 3x3).
    """

    def __init__(self, dataset, imgsz=640, p=1.0, n=4,dtype=np.uint8):
        """Initializes the object with a dataset, image size, probability, and border."""
        assert 0 <= p <= 1.0, f'The probability should be in range [0, 1], but got {p}.'
        assert n in (4, 9), 'grid must be equal to 4 or 9.'
        super().__init__(dataset=dataset, p=p)
        self.dataset = dataset
        self.imgsz = imgsz
        self.border = (-imgsz // 2, -imgsz // 2)  # width, height
        self.n = n
        self.dtype=dtype
        self.gray_value=114 if dtype==np.uint8 else 114*256

    def get_indexes(self, buffer=True):
        """Return a list of random indexes from the dataset."""
        if buffer:  # select images from buffer
            return random.choices(list(self.dataset.buffer), k=self.n - 1)
        else:  # select any images
            return [random.randint(0, len(self.dataset) - 1) for _ in range(self.n - 1)]

    def _mix_transform(self, labels):
        """Apply mixup transformation to the input image and labels."""
        assert labels.get('rect_shape', None) is None, 'rect and mosaic are mutually exclusive.'
        assert len(labels.get('mix_labels', [])), 'There are no other images for mosaic augment.'
        return self._mosaic4(labels) if self.n == 4 else self._mosaic9(labels)

    def _mosaic4(self, labels):
        """Create a 2x2 image mosaic."""
        mosaic_labels = []
        s = self.imgsz
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.border)  # mosaic center x, y
        for i in range(4):
            labels_patch = labels if i == 0 else labels['mix_labels'][i - 1]
            # Load image
            img = labels_patch['img']
            h, w = labels_patch.pop('resized_shape')
            # print(img.shape)
            # Place img in img4
            if i == 0:  # top left
                if len(img.shape)==2:  # single channel
                    img4 = np.full((s * 2, s * 2, 1), self.gray_value, dtype=self.dtype)  # base image with 4 tiles
                else:
                    img4 = np.full((s * 2, s * 2, img.shape[2]), self.gray_value, dtype=self.dtype)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            if len(img.shape)==2:  # single channel
                img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b,np.newaxis]  # img4[ymin:ymax, xmin:xmax]
            else:
                img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            labels_patch = self._update_labels(labels_patch, padw, padh)
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)
        final_labels['img'] = img4
        return final_labels

    def _mosaic9(self, labels):
        """Create a 3x3 image mosaic."""
        mosaic_labels = []
        s = self.imgsz
        hp, wp = -1, -1  # height, width previous
        for i in range(9):
            labels_patch = labels if i == 0 else labels['mix_labels'][i - 1]
            # Load image
            img = labels_patch['img']
            h, w = labels_patch.pop('resized_shape')

            # Place img in img9
            if i == 0:  # center
                img9 = np.full((s * 3, s * 3, img.shape[2]),self.gray_value, dtype=self.dtype)  # base image with 4 tiles
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
            elif i == 1:  # top
                c = s, s - h, s + w, s
            elif i == 2:  # top right
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:  # right
                c = s + w0, s, s + w0 + w, s + h
            elif i == 4:  # bottom right
                c = s + w0, s + hp, s + w0 + w, s + hp + h
            elif i == 5:  # bottom
                c = s + w0 - w, s + h0, s + w0, s + h0 + h
            elif i == 6:  # bottom left
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:  # left
                c = s - w, s + h0 - h, s, s + h0
            elif i == 8:  # top left
                c = s - w, s + h0 - hp - h, s, s + h0 - hp

            padw, padh = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coords

            # Image
            img9[y1:y2, x1:x2] = img[y1 - padh:, x1 - padw:]  # img9[ymin:ymax, xmin:xmax]
            hp, wp = h, w  # height, width previous for next iteration

            # Labels assuming imgsz*2 mosaic size
            labels_patch = self._update_labels(labels_patch, padw + self.border[0], padh + self.border[1])
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)

        final_labels['img'] = img9[-self.border[0]:self.border[0], -self.border[1]:self.border[1]]
        return final_labels

    @staticmethod
    def _update_labels(labels, padw, padh):
        """Update labels."""
        nh, nw = labels['img'].shape[:2]
        labels['instances'].convert_bbox(format='xyxy')
        labels['instances'].denormalize(nw, nh)
        labels['instances'].add_padding(padw, padh)
        return labels

    def _cat_labels(self, mosaic_labels):
        """Return labels with mosaic border instances clipped."""
        if len(mosaic_labels) == 0:
            return {}
        cls = []
        instances = []
        imgsz = self.imgsz * 2  # mosaic imgsz
        for labels in mosaic_labels:
            cls.append(labels['cls'])
            instances.append(labels['instances'])
        final_labels = {
            'im_file': mosaic_labels[0]['im_file'],
            'ori_shape': mosaic_labels[0]['ori_shape'],
            'resized_shape': (imgsz, imgsz),
            'cls': np.concatenate(cls, 0),
            'instances': Instances.concatenate(instances, axis=0),
            'mosaic_border': self.border}  # final_labels
        final_labels['instances'].clip(imgsz, imgsz)
        good = final_labels['instances'].remove_zero_area_boxes()
        final_labels['cls'] = final_labels['cls'][good]
        return final_labels

class MixUp(BaseMixTransform):

    def __init__(self, dataset, pre_transform=None, p=0.0,dtype=np.uint8) -> None:
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)
        self.dtype=dtype
    def get_indexes(self):
        """Get a random index from the dataset."""
        return random.randint(0, len(self.dataset) - 1)

    def _mix_transform(self, labels):
        """Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf."""
        r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
        labels2 = labels['mix_labels'][0]
        labels['img'] = (labels['img'] * r + labels2['img'] * (1 - r)).astype(self.dtype)
        labels['instances'] = Instances.concatenate([labels['instances'], labels2['instances']], axis=0)
        labels['cls'] = np.concatenate([labels['cls'], labels2['cls']], 0)
        return labels

class RandomHSV:

    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5,dtype=np.uint8 ) -> None:
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain
        self.dtype=dtype

    def __call__(self, labels):
        """Applies random horizontal or vertical flip to an image with a given probability."""
        img = labels['img']
        if img.shape[-1] != 3:  # only apply to RGB images
            return labels
        if self.hgain or self.sgain or self.vgain:
            r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # random gains
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            dtype = self.dtype  # uint8
            # if self.dtype == np.uint8:
            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
            # else:
            #     x = np.arange(0, 65536, dtype=r.dtype)
            #     lut_hue = ((x * r[0]) % 180).astype(dtype)
            #     lut_sat = np.clip(x * r[1], 0, 65535).astype(dtype)
            #     lut_val = np.clip(x * r[2], 0, 65535).astype(dtype)
            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
        return labels

class RandomBrightness:

    def __init__(self, gain=0.5, dtype=np.uint8,p=0.2,use_transform=True):
        self.gain = gain
        self.dtype = dtype
        self.use_transform=use_transform
        self.p=p
    def __call__(self, labels):
        img = labels['img']
        # print(self.dtype)
        # print(img.dtype)
        if self.gain:
            if self.dtype == np.uint8:
                r = np.random.uniform(-1, 1) * self.gain + 1  # random gain
                x = np.arange(0, 256, dtype=self.dtype)
                lut = np.clip(x * r, 0, 255).astype(self.dtype)
                img = cv2.LUT(img, lut)
            else:
                if self.use_transform==True and  random.random() < self.p:
                    r = np.random.uniform(-1, 1) * self.gain + 1  # random gain
                    x = np.arange(0, 256, dtype=np.uint8)
                    lut = np.clip(x * r, 0, 255).astype(np.uint8)  # 将查找表转换为 np.uint8
                    img = cv2.LUT((img * 255).astype(np.uint8), lut)  # 将图像转换为 np.uint8 并应用查找表
                    img = img.astype(self.dtype) / np.max(img)  # 将图像转换回 np.float32
            labels['img'] = img
        return labels

class CopyPaste:

    def __init__(self, p=0.5,dtype=np.uint8) -> None:
        self.p = p
        self.dtype = dtype

    def __call__(self, labels):
        """Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)."""
        im = labels['img']
        cls = labels['cls']
        h, w = im.shape[:2]
        instances = labels.pop('instances')
        instances.convert_bbox(format='xyxy')
        instances.denormalize(w, h)
        if self.p and len(instances.segments):
            n = len(instances)
            _, w, _ = im.shape  # height, width, channels
            im_new = np.zeros(im.shape, self.dtype)

            # Calculate ioa first then select indexes randomly
            ins_flip = deepcopy(instances)
            ins_flip.fliplr(w)

            ioa = bbox_ioa(ins_flip.bboxes, instances.bboxes)  # intersection over area, (N, M)
            indexes = np.nonzero((ioa < 0.30).all(1))[0]  # (N, )
            n = len(indexes)
            for j in random.sample(list(indexes), k=round(self.p * n)):
                cls = np.concatenate((cls, cls[[j]]), axis=0)
                instances = Instances.concatenate((instances, ins_flip[[j]]), axis=0)
                cv2.drawContours(im_new, instances.segments[[j]].astype(np.int32), -1, (1, 1, 1), cv2.FILLED)

            result = cv2.flip(im, 1)  # augment segments (flip left-right)
            i = cv2.flip(im_new, 1).astype(bool)
            im[i] = result[i]  # cv2.imwrite('debug.jpg', im)  # debug

        labels['img'] = im
        labels['cls'] = cls
        labels['instances'] = instances
        return labels

try:
    import albumentations as A
except:
    pass

def make_odd(num):
    num = math.ceil(num)
    if num % 2 == 0:
        num += 1
    return num


class RandomHSV4C:

    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5, brightness_range=(0.5, 1.5)) -> None:
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain
        self.brightness_range = brightness_range

    def __call__(self, labels):
        """Applies image HSV augmentation"""
        img = labels['img']
        # print("img.shape=",img.shape)
        gray = img[:, :, 3]  # Extract the grayscale channel from the 4th channel
        bgr = img[:, :, :3]  # Extract the BGR channels
        if self.hgain or self.sgain or self.vgain:
            r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # random gains for H, S, V
            hue, sat, val = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))  # split the 3 channels: H, S, V
            dtype = bgr.dtype  # uint8

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            bgr_transformed = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)  # Transform BGR channels with HSV values

            # Adjust brightness of the grayscale channel
            # brightness_factor = np.random.uniform(self.brightness_range[0], self.brightness_range[1])
            # adjusted_gray = np.clip(gray * brightness_factor, 0, 255).astype(gray.dtype)

            r = np.random.uniform(-1, 1) * self.vgain + 1  # random gain
            dtype = gray.dtype  # uint8
            x = np.arange(0, 256, dtype=dtype)
            lut = np.clip(x * r, 0, 255).astype(dtype)
            gray = cv2.LUT(gray, lut)

            img[:, :, :3] = bgr_transformed  # Update the BGR channels with the transformed values
            img[:, :, 3] = gray  # Update the grayscale channel with adjusted brightness
            labels['img']=img
        return labels


class RandomHSV6C:

    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5, brightness_range=(0.5, 1.5)) -> None:
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain
        self.brightness_range = brightness_range

    def __call__(self, labels):
        """Applies image HSV augmentation"""
        img = labels['img']
        gray = img[:, :, 3:]  # Extract the grayscale channel from the 4th channel
        bgr = img[:, :, :3]  # Extract the BGR channels
        if self.hgain or self.sgain or self.vgain:
            r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # random gains for H, S, V
            hue, sat, val = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))  # split the 3 channels: H, S, V
            dtype = bgr.dtype  # uint8

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            bgr_transformed = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)  # Transform BGR channels with HSV values

            # Adjust brightness of the grayscale channel
            # brightness_factor = np.random.uniform(self.brightness_range[0], self.brightness_range[1])
            # adjusted_gray = np.clip(gray * brightness_factor, 0, 255).astype(gray.dtype)

            hue, sat, val = cv2.split(cv2.cvtColor(gray, cv2.COLOR_BGR2HSV))  # split the 3 channels: H, S, V
            dtype = gray.dtype  # uint8

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            im_hsv2 = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            bgr_transformed2 = cv2.cvtColor(im_hsv2, cv2.COLOR_HSV2BGR)  # Transform BGR channels with HSV values

            img[:, :, :3] = bgr_transformed  # Update the BGR channels with the transformed values
            img[:, :, 3:] = bgr_transformed2  # Update the grayscale channel with adjusted brightness
            labels['img']=img
        return labels


class Albumentations4C:
    """Albumentations transformations. Optional, uninstall package to disable.
    Applies Blur, Median Blur, convert to grayscale, Contrast Limited Adaptive Histogram Equalization,
    random change of brightness and contrast, RandomGamma and lowering of image quality by compression."""

    def __init__(self, p=1.0):
        """Initialize the transform object for YOLO bbox formatted params."""
        self.p = p
        self.transform = None
        prefix = colorstr('albumentations: ')
        try:
            import albumentations as A

            check_version(A.__version__, '1.0.3', hard=True)  # version requirement

            T = [
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                # A.ToGray(p=0.01),
                # A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_lower=75, p=0.0)]  # transforms
            self.transform = A.Compose(T, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

            LOGGER.info(prefix + ', '.join(f'{x}'.replace('always_apply=False, ', '') for x in T if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            LOGGER.info(f'{prefix}{e}')

    def __call__(self, labels):
        """Generates object detections and returns a dictionary with detection results."""
        im = labels['img']
        cls = labels['cls']
        if len(cls):
            labels['instances'].convert_bbox('xywh')
            labels['instances'].normalize(*im.shape[:2][::-1])
            bboxes = labels['instances'].bboxes
            # TODO: add supports of segments and keypoints
            if self.transform and random.random() < self.p:
                new = self.transform(image=im, bboxes=bboxes, class_labels=cls)  # transformed
                if len(new['class_labels']) > 0:  # skip update if no bbox in new im
                    labels['img'] = new['image']
                    labels['cls'] = np.array(new['class_labels'])
                    bboxes = np.array(new['bboxes'], dtype=np.float32)
            labels['instances'].update(bboxes=bboxes)
        return labels

def v8_transforms(dataset, imgsz, hyp,stretch=False):
    """Convert images to a size suitable for YOLOv8 training."""
    dtype=np.uint8 if hyp.use_simotm!="Gray16bit" else np.float32
    pre_transform = Compose([
        Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic,dtype=dtype),
        CopyPaste(p=hyp.copy_paste,dtype=dtype ),
        RandomPerspective(
            degrees=hyp.degrees,
            translate=hyp.translate,
            scale=hyp.scale,
            shear=hyp.shear,
            perspective=hyp.perspective,
            # pre_transform=LetterBox(new_shape=(imgsz, imgsz)),
            pre_transform=None if stretch else LetterBox(new_shape=(imgsz, imgsz)),
        )])
    flip_idx = dataset.data.get('flip_idx', None)  # for keypoints augmentation
    if dataset.use_keypoints:
        kpt_shape = dataset.data.get('kpt_shape', None)
        if flip_idx is None and hyp.fliplr > 0.0:
            hyp.fliplr = 0.0
            LOGGER.warning("WARNING ⚠️ No 'flip_idx' array defined in data.yaml, setting augmentation 'fliplr=0.0'")
        elif flip_idx and (len(flip_idx) != kpt_shape[0]):
            raise ValueError(f'data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}')
    alb=Albumentations(p=1.0) if dtype == np.uint8 else Albumentations(p=0)
    random_hsv= RandomBrightness(gain=hyp.hsv_v, dtype=dtype, p=hyp.brightness) if hyp.channels == 1 else RandomHSV(
            hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v, dtype=dtype)

    if  hyp.channels == 4:
        alb=Albumentations4C(p=1.0)
        random_hsv = RandomHSV4C(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v)
    if  hyp.channels == 6:
        alb=Albumentations4C(p=1.0)
        random_hsv = RandomHSV6C(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v)
    return Compose([
        pre_transform,
        MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup, dtype=dtype),
        alb,
        random_hsv,
        RandomFlip(direction='vertical', p=hyp.flipud),
        RandomFlip(direction='horizontal', p=hyp.fliplr, flip_idx=flip_idx)])  # transforms