import torch
from ultralytics import YOLO
from .utils import Result

# Загрузка модели YOLO
model = YOLO('CoinCounter/models/model.onnx', task='detect',verbose=False)

def predict(path='data', conf=0.5, iou=0.5):
    """
    Выполняет предсказание с помощью модели YOLO.

    :param path: Путь к изображению или директории с изображениями для предсказания.
    :param conf: Порог уверенности для предсказаний (0.0 - 1.0). Чем выше значение, тем меньше ложных срабатываний.
    :param iou: Порог перекрытия для фильтрации предсказанных рамок (0.0 - 1.0). Чем выше значение, тем более строгий отбор.
    :return: Результаты предсказания в формате объекта Result.
    """
    # Выполнение предсказания с заданными параметрами
    results = model.predict(
        path, 
        conf=conf,   # Параметр уверенности
        iou=iou,     # Параметр IoU
        verbose=False
    )
    return Result(results)
