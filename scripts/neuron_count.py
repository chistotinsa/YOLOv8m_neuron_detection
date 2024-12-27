import cv2
import os
import numpy as np
import pandas as pd
import random
from typing import List, Tuple
from inference_model import yolo_inference
from del_outliers import del_outliers
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, '..', 'models', 'YOLOv8m_brain_cell_v3_maP50_0.742.pt')

PIXEL_STEP = 700


# 870pxl = 300µm on a 2048х1504pxl photo with microscopic magnification 20x.
# We set pixel_step = 700pxl for tech reasons.
# If you have another photo`s resolution: find out how many pixels are in 300µm
# on your photo, subtract ~20% to get your PIXEL_STEP value.

def neuron_count(intrend_annotations: List[List[float]], image_path: str, pixel_step: int) -> Tuple[str, int]:
    """
    Counts neurons in a random section of the hippocampus 300µm long.
    :param intrend_annotations: List of lists with coordinates of bboxes
           with detected neurons in xyxy format without outliers
    :param image_path: path to the single image in folder with images
    :param pixel_step: 300µm hippocampus area translated to pixels
    """
    image = cv2.imread(image_path)

    # Check if there are any annotations to process
    if not intrend_annotations:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        print(f'No annotations available for pic {image_name}')
        return image_name, 0

    # Extracting coordinates of bbox centers
    x_values = [(rect[0] + rect[2]) / 2 for rect in intrend_annotations]
    y_values = [(rect[1] + rect[3]) / 2 for rect in intrend_annotations]

    # Check if there are enough points to fit a polynomial
    if len(x_values) < 2:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        print(f'Pic name: {image_name}')
        print('Not enough points for polynomial fitting.')
        return image_name, len(intrend_annotations)

    p = np.polyfit(x_values, y_values, 4)

    # Making a trendline
    x_trend_pixels = np.linspace(min(x_values), max(x_values), 100)
    y_trend_pixels = np.polyval(p, x_trend_pixels)

    # Setting min distance from a random point on the trendline to the pic's edge
    min_distance_from_edge = pixel_step / 2

    max_attempts = 100
    attempts = 0

    while attempts < max_attempts:
        random_index = random.randint(0, len(x_trend_pixels) - 1)
        random_point_x = x_trend_pixels[random_index]
        random_point_y = y_trend_pixels[random_index]

        # Checking if a point is at a sufficient distance from the edges of the pic
        if (min_distance_from_edge <= random_point_x <= image.shape[1] - min_distance_from_edge and
                min_distance_from_edge <= random_point_y <= image.shape[0] - min_distance_from_edge):
            break

        attempts += 1

    if attempts == max_attempts:
        print(f'Warning: Could not find a valid point after {max_attempts} attempts. Using fallback values.')

        # Setting default point - the trend`s center
        random_point_x = x_trend_pixels[50]
        random_point_y = y_trend_pixels[50]

    half_patch_size = pixel_step // 2

    # Calculating the boundaries of a cropped area
    left_x = int(random_point_x - half_patch_size)
    bottom_y = int(random_point_y - half_patch_size)
    right_x = int(random_point_x + half_patch_size)
    top_y = int(random_point_y + half_patch_size)

    # Calculating the number of annotations in a croped area
    num_annotations_in_patch = 0
    for annotation in intrend_annotations:
        x1, y1, x2, y2 = annotation

        if x1 >= left_x and y1 <= top_y and x2 <= right_x and y2 >= bottom_y:
            num_annotations_in_patch += 1

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    print(f'Pic name: {image_name}')
    print(f'The number of alive neurons in 300µm random area: {num_annotations_in_patch}')

    return image_name, num_annotations_in_patch


def run(photo_path: str,
        results: str,
        save: bool = False,
        model_path: str = model_path) -> None:

    valid_extensions = ('.bmp', '.jpg', '.jpeg', '.png')

    image_files = [f for f in os.listdir(photo_path) if f.lower().endswith(valid_extensions)]
    files_count = len(image_files)

    preds = yolo_inference(photo_path, model_path, files_count, save)
    df = pd.DataFrame(columns=['image_name', 'neurons_in_300µm'])

    counter = 0
    for pred, photo in zip(preds, image_files):
        intrend_annotations = del_outliers(pred)
        image_path = os.path.join(photo_path, photo)
        image_name, num_annotations_in_patch = neuron_count(intrend_annotations=intrend_annotations,
                                                            image_path=image_path,
                                                            pixel_step=PIXEL_STEP)
        df.loc[counter, 'image_name'] = image_name
        df.loc[counter, 'neurons_in_300µm'] = num_annotations_in_patch
        counter += 1

    os.makedirs(os.path.dirname(results), exist_ok=True)

    if not results.endswith('.xlsx'):
        results += '.xlsx'

    df.to_excel(results, index=False)


def main():
    parser = argparse.ArgumentParser(
        description=r"Нейросеть, считающая количество живых нейронов в области гиппокампа = 300мкм")

    parser.add_argument("--save", type=bool, required=False, default=True,
                        help=r"Сохранять или нет результаты работы модели в формате фото с размеченными нейронами, по умолчанию --save True")
    parser.add_argument("--photo_path", type=str, required=True,
                        help=r"Укажите полный путь папки с фото для обработки, например --photo_path 'C:\Users\user\Desktop\folder'")
    parser.add_argument("--results", type=str, required=True,
                        help=r"Укажите полный путь для сохранения excel с подсчетами, например --results 'C:\Users\user\Desktop\results.xlsx'")

    args = parser.parse_args()

    run(args.photo_path, args.results, args.save)


if __name__ == '__main__':
    main()


#TODO: поработать над размером кропа, сейчас это квадрат 980х980 на фото 2048х1504, это очень много
#TODO: + есть лимиты от края - это половина от 980, т.е. очень мало места остается для выбора центра кропа.
