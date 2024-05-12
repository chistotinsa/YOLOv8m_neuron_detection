from typing import List
from ultralyticsplus import YOLO


def yolo_inference(pics: str, model_path: str, files_count: int) -> List[List[List[float]]]:
    """Detect alive neurons on photos, returns their coordinates in xyxy format
    set in lists and saves photos with bounding boxes."""
    model = YOLO(model_path)
    results = model.predict(pics, save=False, hide_labels=True)

    return [results[num].boxes.xyxy.tolist() for num in range(files_count)]


# if __name__ == '__main__':
#     yolo_inference(pics=photo_folder_path, model_path=model_path)
