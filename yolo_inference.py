from ultralytics import YOLO
from roboflow import Roboflow

model = YOLO('yolov5x.pt')

resutls = model.predict('input_files/08fd33_4.mp4', save=True)
print(resutls[0])
print('=============================')
for box in resutls[0].boxes:
    print(box)

