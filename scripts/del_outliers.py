from typing import List
import numpy as np


def del_outliers(list_annotations: List[List[float]]) -> List[List[float]]:
    """
    Pick only those bboxes with neurons, that lie on the trend of hippocampus.
    :param list_annotations: List of lists with coordinates of bboxes with detected neurons in xyxy foramt
    """
    # Получить координаты X и Y для всех аннотаций
    x_coordinates = [annotation[0] for annotation in list_annotations]
    y_coordinates = [annotation[1] for annotation in list_annotations]

    #  Вычислить среднее значение и стандартное отклонение для координат X и Y
    mean_x = np.mean(x_coordinates)
    std_x = np.std(x_coordinates)
    mean_y = np.mean(y_coordinates)
    std_y = np.std(y_coordinates)

    # Вычислить Z-оценки для координат X и Y для каждой аннотации
    z_scores_x = [(x - mean_x) / std_x for x in x_coordinates]
    z_scores_y = [(y - mean_y) / std_y for y in y_coordinates]

    # Определить порог Z для исключения выбросов
    threshold_z = 1.5

    # Исключить аннотации с выбросами
    intrend_annotations = []
    for i in range(len(list_annotations)):
        if abs(z_scores_x[i]) <= threshold_z and abs(z_scores_y[i]) <= threshold_z:
            intrend_annotations.append(list_annotations[i])

    return intrend_annotations
