import cv2


def _parse_yolo_annotation(annotation_file, image_path):
    """
    Функция используется только для парсинга аннотаций в формате txt
    для приведения к формате аннотаций [x1,y1,x2,y2], как и в предсказаниях
    На вход изображение и его аннотация
    """
    annotations = []
    with open(annotation_file, 'r') as f:
        for line in f:
            data = line.strip().split()
            class_id = {0: 'neuron'}
            image = cv2.imread(image_path)
            image_height, image_width, _ = image.shape
            x_center, y_center, width, height = map(float, data[1:])
            # Перевод относительных координат в абсолютные пиксельные координаты
            x_center_px = int(x_center * image_width)
            y_center_px = int(y_center * image_height)
            width_px = int(width * image_width)
            height_px = int(height * image_height)

    # Вычисление координат bbox
            x1 = int(x_center_px - width_px / 2)
            y1 = int(y_center_px - height_px / 2)
            x2 = int(x_center_px + width_px / 2)
            y2 = int(y_center_px + height_px / 2)
            annotations.append((x1, y1, x2, y2))

    return annotations
