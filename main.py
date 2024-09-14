import tkinter as tk
from tkinter import filedialog, messagebox
from CoinCounter.model import predict
import os
import glob

class CoinCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CoinCounter")
        self.root.geometry("300x150") 

        # Кнопка для выбора файла
        self.file_button = tk.Button(root, text="Выбрать файл", command=self.select_file)
        self.file_button.pack(pady=10)

        # Кнопка для выбора папки
        self.folder_button = tk.Button(root, text="Выбрать папку", command=self.select_folder)
        self.folder_button.pack(pady=10)

    def select_file(self):
        file_path = filedialog.askopenfilename(
            title="Выберите файл",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            self.process_image(file_path)

    def select_folder(self):
        folder_path = filedialog.askdirectory(title="Выберите папку")
        if folder_path:
            if self.check_images_in_folder(folder_path):
                self.process_image(folder_path)
            else:
                messagebox.showinfo("Информация", "В папке нет изображений")

    def check_images_in_folder(self, folder_path):
        image_files = glob.glob(os.path.join(folder_path, "*.jpg")) + \
                      glob.glob(os.path.join(folder_path, "*.jpeg")) + \
                      glob.glob(os.path.join(folder_path, "*.png"))
        return len(image_files) > 0

    def process_image(self, path):
        # Показать окно с сообщением о процессе прогноза
        self.processing_window = tk.Toplevel(self.root)
        self.processing_window.title("Процесс")
        self.processing_window.geometry("350x100")  # Увеличиваем размер окна
        tk.Label(self.processing_window, text="Идет процесс прогноза... пожалуйста, подождите", wraplength=300).pack(pady=30)
        self.root.update_idletasks()  # Обновить интерфейс перед выполнением длительной задачи

        self.root.after(100, self.run_prediction, path)  # Запуск функции предсказания через 100 мс

    def run_prediction(self, path):
        try:
            # Запуск функции predict
            results = predict(
                path=path,
                conf=0.3,
                iou=0.3
            )
            total_amount = results.total()
            self.processing_window.destroy()  # Закрыть окно с процессом
            messagebox.showinfo("Результат", f"Общая сумма: {total_amount:.2f}$")
        except Exception as e:
            self.processing_window.destroy()  # Закрыть окно с процессом
            messagebox.showerror("Ошибка", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = CoinCounterApp(root)
    root.mainloop()
