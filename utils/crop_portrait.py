"""
Crop upper boddy in every video frame, square bounding box is averaged among all frames and fixed.
"""

import os
import cv2
import argparse
from tqdm import tqdm
import face_recognition

import util


def calc_bbox(image_list, batch_size=5):
    """Batch infer of face location, batch_size should be factor of total frame number."""
    top_sum = right_sum = bottom_sum = left_sum = 0

    for i in tqdm(range(len(image_list) // batch_size)):
        image_batch = []

        for j in range(i * batch_size, (i + 1) * batch_size):
            image = face_recognition.load_image_file(image_list[j])
            image_batch.append(image)

        face_locations = face_recognition.batch_face_locations(image_batch, number_of_times_to_upsample=0, batch_size=batch_size)

        for face_location in face_locations:
            top, right, bottom, left = face_location[0]  # assuming only one face detected in the frame
            top_sum += top
            right_sum += right
            bottom_sum += bottom
            left_sum += left

    return (top_sum // len(image_list), right_sum // len(image_list), bottom_sum // len(image_list), left_sum // len(image_list))


def crop_image(data_dir, dest_size, crop_level, vertical_adjust):
    image_list = util.get_file_list(os.path.join(data_dir, 'full'))
    top, right, bottom, left = calc_bbox(image_list)

    height = bottom - top
    width = right - left

    crop_size = int(height * crop_level)

    horizontal_delta = (crop_size - width) // 2
    left -= horizontal_delta
    right += horizontal_delta

    top = int(top * vertical_adjust)
    bottom = top + crop_size

    for i in tqdm(range(len(image_list))):
        image =cv2.imread(image_list[i])
        image = image[top:bottom, left:right]

        image = cv2.resize(image, (dest_size, dest_size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(args.data_dir, 'crop', os.path.basename(image_list[i])), image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--dest_size', type=int, default=256)
    parser.add_argument('--crop_level', type=float, default=2.0, help='Adjust crop image size.')
    parser.add_argument('--vertical_adjust', type=float, default=0.3, help='Adjust vertical location of portrait in image.')
    args = parser.parse_args()

    crop_image(args.data_dir, dest_size=args.dest_size, crop_level=args.crop_level, vertical_adjust=args.vertical_adjust)
