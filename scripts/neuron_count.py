import cv2
import os
import numpy as np
import pandas as pd
import random
from typing import List, Tuple
from inference_model import yolo_inference  # Убедитесь, что этот модуль доступен
from del_outliers import del_outliers  # Убедитесь, что этот модуль доступен
import argparse
import matplotlib.pyplot as plt  # Добавлен для отображения изображений

PIXEL_STEP = 950


# 870pxl = 300µm on a 2048х1504pxl photo with microscopic magnification 20x.
# We set pixel_step = 950pxl for tech reasons (to add a frame in which
# neurons that fall more than half into the 300 µm area will be detected).
# If you have another photo`s resolution: find out how many pixels are in 300µm
# on your photo, add 9.2% to get your PIXEL_STEP value.

def neuron_count(intrend_annotations: List[List[float]], image_path: str, pixel_step: int) -> Tuple[str, int]:
    """
    Сounts neurons in a random section of the hippocampus 300µm long.
    :param intrend_annotations: List of lists with coordinates of bboxes
           with detected neurons in xyxy format without outliers
    :param image_path: path to the single image in folder with images
    :param pixel_step: 300µm hippocampus area translated to pixels
    """

    image = cv2.imread(image_path)

    # Extracting coordinates of bbox centers
    x_values = [(rect[0] + rect[2]) / 2 for rect in intrend_annotations]
    y_values = [(rect[1] + rect[3]) / 2 for rect in intrend_annotations]

    p = np.polyfit(x_values, y_values, 4)

    # Making a trendline
    x_trend_pixels = np.linspace(min(x_values), max(x_values), 100)
    y_trend_pixels = np.polyval(p, x_trend_pixels)

    # Setting min distance from a random point on the trendline to the pic's edge
    min_distance_from_edge = pixel_step / 2

    while True:
        # If the random point doesn't fit the condition, set another random point
        random_index = random.randint(0, len(x_trend_pixels) - 1)
        random_point_x = x_trend_pixels[random_index]
        random_point_y = y_trend_pixels[random_index]

        # Checking if a point is at a sufficient distance from the edges of the pic
        if min_distance_from_edge <= random_point_x <= image.shape[1] - min_distance_from_edge and \
                min_distance_from_edge <= random_point_y <= image.shape[0] - min_distance_from_edge:
            break

    half_patch_size = pixel_step // 2  # Set aside half the patch size in 4 directions from a random point on the trendline

    # Calculating the boundaries of a cropped area
    top_left_x = int(random_point_x - half_patch_size)
    top_left_y = int(random_point_y - half_patch_size)
    bottom_right_x = int(random_point_x + half_patch_size)
    bottom_right_y = int(random_point_y + half_patch_size)

    # plt.imshow(patch, interpolation='nearest')
    # plt.show()

    # Calculating the number of bboxes in a cropped area
    num_annotations_in_patch = 0
    for annotation in intrend_annotations:
        x1, y1, x2, y2 = annotation

        if x1 >= top_left_x and y1 >= top_left_y and x2 <= bottom_right_x and y2 <= bottom_right_y:
            num_annotations_in_patch += 1

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    print(f'Pic name: {image_name}')
    print(f'The number of alive neurons in 300µm random area: {num_annotations_in_patch}')

    return image_name, num_annotations_in_patch


def run(photo_path: str,
        results: str,
        save: bool = True,
        model_path: str = '../models/YOLOv8m_brain_cell_v3_maP50_0.742.pt') -> None:

    files_count = len(os.listdir(photo_path))
    preds = yolo_inference(photo_path, model_path, files_count, save)
    df = pd.DataFrame(columns=['image_name', 'neurons_in_300µm'])

    counter = 0
    for pred, photo in zip(preds, os.listdir(photo_path)):
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
