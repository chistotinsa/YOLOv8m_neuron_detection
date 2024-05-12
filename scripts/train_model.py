from ultralyticsplus import YOLO, render_result


custom_dataset_path = ''
finetuned_model_name = 'yolov8m_neuron_detec_high_rez_17_04'

model = YOLO('keremberke/yolov8m-blood-cell-detection')
model.train(data=custom_dataset_path,
            name=finetuned_model_name,
            epochs=40,
            batch=8)