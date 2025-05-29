import customtkinter as ctk
from tkinter import filedialog, messagebox
from Backend.model_inference import detect_objects, detect_video, detect_folder
import os
import subprocess

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("YOLO Обработка")
        self.geometry("500x300")
        self.resizable(False, False)

        self.file_path = None

        self.label = ctk.CTkLabel(self, text="Выберите изображение или видео", font=("Arial", 16))
        self.label.pack(pady=20)

        self.select_button = ctk.CTkButton(self, text="📂 Выбрать файл", command=self.select_file)
        self.select_button.pack(pady=10)

        self.process_button = ctk.CTkButton(self, text="🚀 Обработать", command=self.process_file, state="disabled")
        self.process_button.pack(pady=10)

        self.open_folder_button = ctk.CTkButton(self, text="📁 Открыть папку с результатами", command=self.open_output_folder)
        self.folder_button = ctk.CTkButton(self, text="📁 Выбрать папку для обработки", command=self.select_folder)
        self.folder_button.pack(pady=10)
        self.open_folder_button.pack(pady=10)

    def select_file(self):
        path = filedialog.askopenfilename(title="Выберите файл",
                                          filetypes=[("Изображения и Видео", "*.jpg *.png *.jpeg *.mp4 *.avi *.mov")])
        if path:
            self.file_path = path
            self.label.configure(text=f"Выбран: {os.path.basename(path)}")
            self.process_button.configure(state="normal")

    def process_file(self):
        if not self.file_path:
            return

        ext = os.path.splitext(self.file_path)[1].lower()
        try:
            if ext in [".jpg", ".jpeg", ".png"]:
                detect_objects(self.file_path, "output")
            elif ext in [".mp4", ".avi", ".mov"]:
                detect_video(self.file_path, "output")
            else:
                messagebox.showerror("Ошибка", "Неподдерживаемый формат.")
                return
            messagebox.showinfo("Успех", "Файл успешно обработан и сохранён в папку 'output'.")
        except Exception as e:
            messagebox.showerror("Ошибка обработки", str(e))

    def open_output_folder(self):
        path = os.path.abspath("output")
        os.makedirs(path, exist_ok=True)
        if os.name == 'nt':  # Windows
            os.startfile(path)
        else:  # macOS, Linux
            subprocess.run(["open" if os.name == "posix" else "xdg-open", path])

    def select_folder(self):
        folder_path = filedialog.askdirectory(title="Выберите папку с файлами")
        if folder_path:
            try:
                detect_folder(folder_path, "output")
                messagebox.showinfo("Успех", "Папка обработана! Результаты в 'output'.")
            except Exception as e:
                messagebox.showerror("Ошибка обработки папки", str(e))

if __name__ == "__main__":
    app = App()
    app.mainloop()
