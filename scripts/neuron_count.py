import argparse
import cv2
import os
import numpy as np
import pandas as pd
import random
from typing import List, Tuple
from inference_model import yolo_inference
from del_outliers import del_outliers

model_path = 'models/YOLOv8m_brain_cell_v3_maP50_0.742.pt'


def neuron_count(intrend_annotations: List[List[float]], image_path: str, pixel_step: int) -> Tuple[str, int]:
    """
    Сounts neurons in a random section of the hippocampus 300µm long.
    :param intrend_annotations: List of lists with coordinates of bboxes
           with detected neurons in xyxy foramt without liers
    :param image_path: path to the single image in folder with images
    :param pixel_step: 300µm hippocampus area translated to pixels
    """

    # image = cv2.imread(f'{photo_path}{image_path}')
    image = cv2.imread(image_path)

    # Extracting coordinates of bbox centers
    x_values = [(rect[0] + rect[2]) / 2 for rect in intrend_annotations]
    y_values = [(rect[1] + rect[3]) / 2 for rect in intrend_annotations]

    p = np.polyfit(x_values, y_values, 4)

    # TODO: a suspicion that everything can crush because of np.linspace by x

    # Making a trendline
    x_trend_pixels = np.linspace(min(x_values), max(x_values), 100)
    y_trend_pixels = np.polyval(p, x_trend_pixels)

    # setting min distance from a random point on the trendline to the pic`s edge
    min_distance_from_edge = pixel_step / 2

    while True:
        # if the random point doesn`t fit the condition, set another random point
        random_index = random.randint(0, len(x_trend_pixels) - 1)
        random_point_x = x_trend_pixels[random_index]
        random_point_y = y_trend_pixels[random_index]

        # Checking if a point is at a sufficient distance from the edges of the pic
        if min_distance_from_edge <= random_point_x <= image.shape[0] - min_distance_from_edge and \
           min_distance_from_edge <= random_point_y <= image.shape[1] - min_distance_from_edge:

            break

    half_patch_size = pixel_step // 2  # set aside half the patch size in 4 directions from a random point on the trendline

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
    print(f'pic name: {image_name}')
    print(f"the number of alive neurons in 300µm random area: {num_annotations_in_patch}")

    return image_name, num_annotations_in_patch


def main(photo_path, result_excel_path, pixel_step):
    files_count = len(os.listdir(photo_path))
    preds = yolo_inference(photo_path, model_path, files_count=files_count)
    df = pd.DataFrame(columns=['image_name', 'neurons_in_300µm'])

    counter = 0
    for pred, photo in zip(preds, os.listdir(photo_path)):
        intrend_annotations = del_outliers(pred)
        image_name, num_annotations_in_patch = neuron_count(intrend_annotations=intrend_annotations,
                                                            image_path=photo,
                                                            pixel_step=pixel_step)
        df.loc[counter, 'image_name'] = image_name
        df.loc[counter, 'neurons_in_300µm'] = num_annotations_in_patch
        counter += 1

    df.to_excel(result_excel_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neuron counting in hippocampus area.')

    parser.add_argument('--photos', type=str, help='Path to the folder with photos. '
                                                   r'Example: C:\Users\user\Desktop\neurons', required=True)

    parser.add_argument('--results', type=str, help='Path where the Excel with results file will be created. '
                                                    r'Example: C:\Users\user\Desktop\results.xlsx', required=True)

    parser.add_argument('--pixel_step', type=int, help='Pixel equivalent of 300µm for select random hippocampus area. '
                                                       'Default is 950, which is standart for 4096x3008pxl photos',
                        required=False, default=950)
    # 870pxl = 300µm. Set 950pxl for tech reasons (to add a frame in which
    # neurons that fall more than half into the 300 µm area will be detected)
    # Note that 300µm on 640x640pxl photo = 161 pixel_step (~ 176 if padding is used, which is preferable)

    args = parser.parse_args()

    main(photo_path=args.photos, result_excel_path=args.results, pixel_step=args.pixel_step)
