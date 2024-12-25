import argparse
import os
from typing import List, Tuple

import numpy as np
import cv2

import torch
from segment_anything import SamPredictor, sam_model_registry


def get_args():
    parser = argparse.ArgumentParser("Demo of few-shot prediction with prototypes")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path to the input yolo root dir"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to the output generated masks dir"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default='vit_h',
        choices=['vit_h', 'vit_l', 'vit_b'],
        help="The SAM model type"
    )

    args = parser.parse_args()
    return args


def extract_predictor_mask(masks):
    return masks[0].astype(np.uint8) * 255


def convert_box(yolo_object: str, img):
    class_id, cx, cy, w, h = map(float, yolo_object.split())

    image_height, image_width = img.shape[:2]
    x_min = int((cx - w / 2) * image_width)
    y_min = int((cy - h / 2) * image_height)
    width = int(w * image_width)
    height = int(h * image_height)

    return class_id, x_min, y_min, width, height


def extract_yolo_objects(yolo_annotation_path: str, img) -> List[Tuple[int, List[int]]]:
    objects: List[Tuple[int, List[int]]] = []

    with open(yolo_annotation_path, 'r') as file:
        for line in file:
            class_id, x_min, y_min, width, height = convert_box(line.strip(), img)
            objects.append((int(class_id), [x_min, y_min, width, height]))

    return objects


def process_sample(img_name, predictor, dst_root, batch_inference = True):
    # Determine the annotation file path
    annotation_file_path = os.path.splitext(img_name)[0] + ".txt"

    # Check if the annotation file exists
    if not os.path.exists(annotation_file_path):
        return

    # Read the image using OpenCV
    img = cv2.imread(img_name)
    if img is None:
        return

    # Extract objects from annotation file
    objects = extract_yolo_objects(annotation_file_path, img)

    input_boxes = []
    class_ids = []

    for obj in objects:
        class_id, [x_min, y_min, width, height] = obj
        box = [x_min, y_min, x_min + width, y_min + height]
        input_boxes.append(box)
        class_ids.append(class_id)

    if not input_boxes:
        return

    predictor.set_image(img)

    # Accumulate masks for each class
    masks_of_classes = dict()

    if batch_inference:
        # Convert the input_boxes list to a Torch tensor
        input_boxes_tensor = torch.tensor(input_boxes, device=predictor.device)

        # Transform boxes using predictor's transformation method
        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes_tensor, img.shape[:2])

        # Perform batch prediction
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        for i, class_id in enumerate(class_ids):
            extracted_mask = extract_predictor_mask(masks[i].cpu().numpy())
            if class_id not in masks_of_classes:
                masks_of_classes[class_id] = extracted_mask
            else:
                masks_of_classes[class_id] += extracted_mask
    else:
        for i, class_id in enumerate(class_ids):
            x_min, y_min, width, height = input_boxes[i]
            box = np.array([x_min, y_min, x_min + width, y_min + height])
            masks, _, _ = predictor.predict(point_coords=None, point_labels=None, box=box[None, :],
                                            multimask_output=False)

            extracted_mask = extract_predictor_mask(masks)

            if class_id not in masks_of_classes:
                masks_of_classes[class_id] = extracted_mask
            else:
                masks_of_classes[class_id] += extracted_mask

    for class_id, mask in masks_of_classes.items():
        if cv2.countNonZero(mask) == 0:
            print('sample', os.path.basename(img_name), 'class', class_id, 'has no mask')
            continue

        dst_dir = os.path.join(dst_root, str(class_id))

        if not os.path.isdir(dst_dir):
            os.mkdir(dst_dir)

        cv2.imwrite(os.path.join(dst_dir, os.path.basename(img_name)), img)
        cv2.imwrite(os.path.join(dst_dir, os.path.splitext(os.path.basename(img_name))[0] + '.mask.png'), mask)


def process(yolo_root, predictor, dst_root, batch_inference = True):
    image_paths = [os.path.join(root, file)
                   for root, _, files in os.walk(yolo_root)
                   for file in files if file.endswith(('.png', '.jpg', '.jpeg'))]

    total_count = len(image_paths)

    for i, image_path in enumerate(image_paths):
        process_sample(image_path, predictor, dst_root, batch_inference)

        if i % 10 == 0:
            print(f"Processed {i + 1} / {total_count} images")


def main(yolo_root, dst_root, model_type):
    checkpoint_paths = {"vit_h": "checkpoints/sam_vit_h_4b8939.pth",
                        "vit_l": "checkpoints/sam_vit_l_0b3195.pth",
                        "vit_b": "checkpoints/sam_vit_b_01ec64.pth"}

    checkpoint_path = checkpoint_paths[model_type]
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to('cuda')
    print('SAM using device:', sam.device)
    predictor = SamPredictor(sam)

    process(yolo_root, predictor, dst_root, True)

    return

if __name__ == "__main__":
    options = get_args()
    main(
        yolo_root=options.input,
        dst_root=options.output,
        model_type=options.model
    )
