import argparse

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import torch

torch.set_grad_enabled(False)
import numpy as np
from detectron2.config import get_cfg
import detectron2.data.transforms as T
import detectron2.data.detection_utils as utils
from tools.train_net import Trainer

import seaborn as sns
import torchvision.ops as ops
from torchvision.ops import box_area
import random

from copy import copy
from scene.scene_change_detector import SceneChangeDetector


def get_args():
    parser = argparse.ArgumentParser("Demo of few-shot prediction with prototypes")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to the config file for the model",
        default='devit/configs/open-vocabulary/lvis/vitl.yaml'
    )
    parser.add_argument(
        "-r",
        "--rpn_config",
        type=str,
        help="Path to the RPN config file",
        default='devit/configs/RPN/mask_rcnn_R_50_FPN_1x.yaml'
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Path to model file",
        default='weights/trained/open-vocabulary/lvis/vitl_0069999.pth'
    )
    parser.add_argument(
        "-i",
        "--input_video",
        type=str,
        help="Path the input video",
        required=True
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path the output dir",
        default='output'
    )
    parser.add_argument(
        "--category",
        type=str,
        help="Path the category space file",
        required=True
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default='cuda',
        help="The device to use for inference"
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.8,
        help = "The detection threshold"
    )
    parser.add_argument(
        "--enable_filtering",
        action='store_true',
        help="Whether to enable detections filtering by size",
    )
    parser.add_argument(
        "--enable_drawing",
        action='store_true',
        help="Whether to enable detections drawing",
    )

    args = parser.parse_args()
    return args


def filter_boxes(instances, threshold=0.0):
    indexes = instances.scores >= threshold

    if indexes.sum() == 0:
        return [], [], []

    boxes = instances.pred_boxes.tensor[indexes, :]
    pred_classes = instances.pred_classes[indexes]
    return boxes, pred_classes, instances.scores[indexes]


def assign_colors(pred_classes, label_names, seed=1):
    all_classes = torch.unique(pred_classes).tolist()
    all_classes = list(set([label_names[ci] for ci in all_classes]))
    colors = list(sns.color_palette("hls", len(all_classes)).as_hex())
    random.seed(seed)
    random.shuffle(colors)
    class2color = {}
    for cname, hx in zip(all_classes, colors):
        class2color[cname] = hx
    colors = [class2color[label_names[cid]] for cid in pred_classes.tolist()]
    return colors


def list_replace(lst, old=1, new=10):
    """replace list elements (inplace)"""
    i = -1
    lst = copy(lst)
    try:
        while True:
            i = lst.index(old, i + 1)
            lst[i] = new
    except ValueError:
        pass
    return lst


def object_to_yolo(box, label, img_width, img_height):
    x, y, x1, y1 = box

    w = x1 - x
    h = y1 - y

    x = x / img_width
    w = w / img_width
    y = y / img_height
    h = h / img_height

    cx = x - (w / 2)
    cy = y - (h / 2)

    return f"{label} {cx} {cy} {w} {h}"

def write_yolo_object(boxes, labels, label_names, img_width, img_height, dst_path):
    yolo_annotation = []

    for box, label in zip(boxes, labels):
        current_label = label_names[label]

        if current_label.isdigit():
            converted_label = int(current_label)
        else:
            converted_label = label

        yolo_annotation.append(object_to_yolo(box, converted_label, img_width, img_height))

    with open(dst_path, 'w') as f:
        f.writelines(yolo_annotation)


def filter_objects_by_size(boxes, labels, exclude_filter_objects_ids, max_width, max_height):
    if len(boxes) == 0:
        return [], []

    filtered_objects = tuple(zip(*((box, label) for box, label in zip(boxes, labels) if
                          (True if label in exclude_filter_objects_ids else
                          (box[2] - box[0] < max_width and
                           box[3] - box[1] < max_height))
    )))

    if len(filtered_objects):
        return filtered_objects

    return [], []


def main(
        config_file,
        rpn_config_file,
        model_path,
        video_path,
        output_dir,
        category_space,
        device,
        threshold,
        enable_filtering=True,
        enable_detections_drawing=True,
        topk=1,
        output_pth=False):
    exclude_filter_objects_ids = []
    max_object_width_rel = 0.3
    max_object_height_rel = 0.5

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_video_dir_path = output_pth

    if os.path.basename(output_dir) != video_name:
        output_video_dir_path = os.path.join(output_dir, video_name)

        if not os.path.isdir(output_video_dir_path):
            os.makedirs(output_video_dir_path, exist_ok=True)

    detection_drawing_dir = os.path.join(output_video_dir_path, 'drawing')

    if enable_detections_drawing:
        if not os.path.isdir(detection_drawing_dir):
            os.makedirs(detection_drawing_dir, exist_ok=True)

    config = get_cfg()
    config.merge_from_file(config_file)
    config.DE.OFFLINE_RPN_CONFIG = rpn_config_file
    config.DE.TOPK = topk
    config.MODEL.MASK_ON = True

    config.freeze()

    augs = utils.build_augmentation(config, False)
    augmentations = T.AugmentationList(augs)

    # building models
    model = Trainer.build_model(config).to(device)
    # loaded = torch.load(model_path, map_location=device)
    # model.load_state_dict(loaded['model'])
    model.load_state_dict(torch.load(model_path, map_location=device)['model'])
    model.eval()
    model = model.to(device)

    if category_space is not None:
        category_space = torch.load(category_space)
        model.label_names = category_space['label_names']
        model.test_class_weight = category_space['prototypes'].to(device)

    scene_change_detector = SceneChangeDetector(10, 0.1, (480, 270))

    class_colors = []

    for i in range(len(model.label_names)):
        color = list(np.random.choice(range(256), size=3))
        class_colors.append((int(color[0]), int(color[1]), int(color[2])))

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_counter = 0

    while True:
        ret, image = cap.read()

        if not ret:
            break

        if scene_change_detector.process(image):
            image = cv2.resize(image, (1280, 720))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            dataset_dict = {}
            image_width = image.shape[1]
            image_height = image.shape[0]
            dataset_dict["height"], dataset_dict["width"] = image_height, image_width

            aug_input = T.AugInput(image)

            augmentations(aug_input)
            dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(aug_input.image.transpose(2, 0, 1))).to(device)

            batched_inputs = [dataset_dict]
            output = model(batched_inputs)[0]
            output['label_names'] = model.label_names

            instances = output['instances']

            if len(instances) == 0:
                continue

            boxes, pred_classes, scores = filter_boxes(instances, threshold=threshold)

            if len(boxes) == 0:
                continue

            mask = box_area(boxes) >= 400
            boxes = boxes[mask]
            pred_classes = pred_classes[mask]
            scores = scores[mask]
            mask = ops.nms(boxes, scores, 0.3)
            boxes = boxes[mask]
            pred_classes = pred_classes[mask]
            areas = box_area(boxes)
            indexes = list(range(len(pred_classes)))
            for c in torch.unique(pred_classes).tolist():
                box_id_indexes = (pred_classes == c).nonzero().flatten().tolist()
                for i in range(len(box_id_indexes)):
                    for j in range(i + 1, len(box_id_indexes)):
                        bid1 = box_id_indexes[i]
                        bid2 = box_id_indexes[j]
                        arr1 = boxes[bid1].cpu().numpy()
                        arr2 = boxes[bid2].cpu().numpy()
                        a1 = np.prod(arr1[2:] - arr1[:2])
                        a2 = np.prod(arr2[2:] - arr2[:2])
                        top_left = np.maximum(arr1[:2], arr2[:2])  # [[x, y]]
                        bottom_right = np.minimum(arr1[2:], arr2[2:])  # [[x, y]]
                        wh = bottom_right - top_left
                        ia = wh[0].clip(0) * wh[1].clip(0)
                        if ia >= 0.9 * min(a1,
                                           a2):  # same class overlapping case, and larger one is much larger than small
                            if a1 >= a2:
                                if bid2 in indexes:
                                    indexes.remove(bid2)
                            else:
                                if bid1 in indexes:
                                    indexes.remove(bid1)

            boxes = boxes[indexes].to(torch.int64).tolist()
            pred_classes = pred_classes[indexes].to(torch.int64).tolist()

            if enable_filtering:
                boxes, pred_classes = filter_objects_by_size(boxes, pred_classes, exclude_filter_objects_ids,
                                                             image_width * max_object_width_rel,
                                                             image_height * max_object_height_rel)

            if len(boxes) == 0:
                continue

            sample_name = os.path.join(output_video_dir_path, video_name + '_' + str(frame_counter))
            img_path = sample_name + '.jpg'
            txt_path = sample_name + '.txt'

            write_yolo_object(boxes, pred_classes, category_space['label_names'], image_width, image_height, txt_path)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_path, image)

            if enable_detections_drawing:
                for box, label in zip(boxes, pred_classes):
                    color = class_colors[label]

                    converted_label = category_space['label_names'][label]

                    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
                    cv2.putText(image, converted_label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                cv2.imshow('img', image)
                drawen_sample_path = os.path.join(detection_drawing_dir, video_name + '_' + str(frame_counter) + '.jpg')
                cv2.imwrite(drawen_sample_path, image)
                cv2.waitKey(1)

        frame_counter += 1

        if frame_counter % 100 == 0:
            print(f'Processing frame {frame_counter} / {total_frames}')


if __name__ == "__main__":
    options = get_args()
    main(
        config_file=options.config,
        rpn_config_file=options.rpn_config,
        model_path=options.model,
        video_path=options.input_video,
        output_dir=options.output,
        category_space=options.category,
        device=options.device,
        threshold=options.threshold,
        enable_filtering=options.enable_filtering,
        enable_detections_drawing=options.enable_drawing
    )
