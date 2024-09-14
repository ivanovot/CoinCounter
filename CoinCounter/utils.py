import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import matplotlib.colors as mcolors


class_map = {
    0: 'Dime',
    1: 'Nickel',
    2: 'Penny',
    3: 'Quarter',
    4: 'fifty',
    5: 'five',
    6: 'hundred',
    7: 'one',
    8: 'ten',
    9: 'twenty'
}

value_map = {
    'Penny': 0.01,
    'Nickel': 0.05,
    'Dime': 0.10,
    'Quarter': 0.25,
    'one': 1.00,
    'five': 5.00,
    'ten': 10.00,
    'twenty': 20.00,
    'fifty': 50.00,
    'hundred': 100.00
}

nomimals = ['Dime', 'Nickel', 'Penny', 'Quarter', 'fifty', 'five', 'hundred', 'one', 'ten', 'twenty']

class Result:
    def __init__(self,results):
        self.results = results
        self.nomimals = nomimals

        self.objects = [[int(box.cls) for box in result.boxes] for result in self.results]
        
        self.df = pd.DataFrame([{class_map[obj]: lst.count(obj) for obj in class_map} for lst in self.objects])
        self.df['total'] = self.df.apply(lambda row: sum(row[obj] * value_map[obj] for obj in value_map), axis=1)

    def __len__(self):
        return len(self.objects)

    def total(self,id=None):
        if id is None:
            return round(self.df.total.sum(),2)
        
        elif type(id) == int:
            return round(self.df.iloc(axis=0)[id].total,2)
        
        else:
            print('pls use int')

    def show(self, num_cols=1, scale=1):
        """
        Функция для создания коллажа с размеченными объектами на фотографиях.
        :param num_cols: Количество столбцов в коллаже
        :param scale: Масштаб для отображения изображений
        """
        # Определяем количество изображений
        num_images = len(self.results)
        
        # Рассчитываем количество строк
        num_rows = (num_images + num_cols - 1) // num_cols  # Округление вверх
        
        # Создаем фигуру и оси для отображения изображений
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5 * scale, num_rows * 5 * scale))
        
        # Преобразуем оси в одномерный массив для удобства
        if num_rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Определяем цвета для объектов
        object_colors = list(mcolors.TABLEAU_COLORS.values())
        
        for i, result in enumerate(self.results):
            if i >= len(axes):
                break

            # Получаем исходное изображение и результат детекции
            img = result.orig_img
            boxes = result.boxes

            # Преобразуем изображение в формат для отображения
            img_pil = Image.fromarray(img)
            
            # Выводим изображение на текущую ось
            axes[i].imshow(img_pil)
            axes[i].axis('off')

            # Добавляем номер изображения
            axes[i].text(0.5, -0.05, i+1, size=12, ha='center', va='top', transform=axes[i].transAxes)

            # Если есть разметка, добавляем её
            if boxes is not None:
                # Переносим данные на CPU и преобразуем в numpy
                xyxy = boxes.xyxy.cpu().numpy()  # Координаты [x1, y1, x2, y2]
                conf = boxes.conf.cpu().numpy()   # Уверенность
                cls = boxes.cls.cpu().numpy()     # Классы

                for j in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[j]
                    class_id = int(cls[j])
                    class_name = result.names[class_id]
                    confidence = conf[j]
                    
                    # Присваиваем уникальный цвет для класса
                    color = object_colors[class_id % len(object_colors)]
                    
                    # Рисуем прямоугольники на изображении
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=color, linewidth=2)
                    axes[i].add_patch(rect)
                    # Добавляем имя класса и уверенность
                    axes[i].text(x1, y1, f'{class_name} {confidence:.2f}', color=color, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

        # Убираем пустые оси, если изображений меньше, чем ячеек в сетке
        for j in range(num_images, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()
