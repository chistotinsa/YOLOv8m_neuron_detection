from typing import List
import numpy as np


def del_outliers(list_annotations: List[List[float]]) -> List[List[float]]:
    """
    Pick only those bboxes with neurons, that lie on the trend of hippocampus.
    :param list_annotations: List of lists with coordinates of bboxes with detected neurons in xyxy foramt
    """
    # get X and Y coordinates for all annotations
    x_coordinates = [annotation[0] for annotation in list_annotations]
    y_coordinates = [annotation[1] for annotation in list_annotations]

    # Compute std and mean for X and Y coordinates
    mean_x = np.mean(x_coordinates)
    std_x = np.std(x_coordinates)
    mean_y = np.mean(y_coordinates)
    std_y = np.std(y_coordinates)

    # Compute Z-values for X and Y for every annotation
    z_scores_x = [(x - mean_x) / std_x for x in x_coordinates]
    z_scores_y = [(y - mean_y) / std_y for y in y_coordinates]

    # define Z threshold for outliers delete
    threshold_z = 1.5

    # delete annotations with outliers
    intrend_annotations = []
    for i in range(len(list_annotations)):
        if abs(z_scores_x[i]) <= threshold_z and abs(z_scores_y[i]) <= threshold_z:
            intrend_annotations.append(list_annotations[i])

    return intrend_annotations
