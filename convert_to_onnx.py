import os
import torch
import torch.nn as nn

# Импортируем инструменты для вызова системного файлового диалога
import tkinter as tk
from tkinter import filedialog

# Скрываем главное неиспользуемое окно tkinter
root = tk.Tk()
root.withdraw()

# 1. Точно такой же каркас твоей сети
class TicTacToeNet(nn.Module):
    def __init__(self):
        super(TicTacToeNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 9)
        )
        
    def forward(self, x):
        return self.model(x)

def run_export():
    print("Ожидание выбора файла проекта ИИ...")
    
    # Открываем диалоговое окно выбора исходного файла .liai
    input_liai_path = filedialog.askopenfilename(
        initialdir=os.getcwd(),
        title="Выберите файл проекта Limehcik AI (.liai) для конвертации",
        filetypes=[("Limehcik AI Model", "*.liai")]
    )
    
    # Если пользователь передумал и закрыл окно проводника
    if not input_liai_path:
        print("Конвертация отменена пользователем.")
        return

    # Автоматически генерируем имя выходного .onnx файла в той же папке
    output_onnx_path = os.path.splitext(input_liai_path)[0] + ".onnx"

    # Инициализируем сеть
    net = TicTacToeNet()
    
    try:
        print(f"Загрузка и чтение контейнера: {os.path.basename(input_liai_path)}...")
        # Загружаем сохраненный словарь (.liai) принудительно на CPU
        checkpoint = torch.load(input_liai_path, map_location="cpu")
        
        # Проверяем структуру файла: наш монолит .liai или голые веса (на случай непредвиденного)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            # Извлекаем веса из нашего формата
            state_dict = checkpoint["model_state"]
            
            # Читаем метаданные и статистику с учетом нового формата v1.1
            metadata = checkpoint.get("metadata", {})
            stats = checkpoint.get("stats", {})
            config = checkpoint.get("config", {})
            
            # Подтягиваем метаданные: приоритет из config (новые файлы), либо старый metadata (обратная совместимость)
            author = config.get("meta_author", metadata.get("author", "Не указан"))
            model_version = config.get("meta_model_version", metadata.get("model_version", "Не указана"))
            app_version = metadata.get("app_version", "Ниже 1.2.0")
            
            print("\n--- Информация о проекте .liai ---")
            print(f" Создан в программе версии: v{app_version}")
            print(f" Версия модели ИИ:         {model_version}")
            print(f" Автор сборки:              {author}")
            print(f" Всего эпох самообучения:   {stats.get('total_background_epochs', 0)}")
            print(f" Сыграно матчей в панели:   {stats.get('games_played', 0)}")
            print(f" Статистика матчей: ИИ [{stats.get('ai_wins', 0)}] | Человек [{stats.get('human_wins', 0)}] | Ничьи [{stats.get('draws', 0)}]")
            print("-----------------------------------\n")
        else:
            # Предупреждение на случай, если подсунули старый .pth, переименованный в .liai
            print("⚠️ Внимание: Файл не содержит метаструктуры .liai, пробуем загрузить как голые веса...")
            state_dict = checkpoint
        
        # Загружаем веса в модель
        net.load_state_dict(state_dict)
        net.eval()
        
        # Создаем "пустышку" входа для компилятора (батч размера 1, 9 ячеек поля)
        dummy_input = torch.randn(1, 9)
        
        print(f"Начало экспорта матрицы весов -> {os.path.basename(output_onnx_path)}...")
        
        # Экспортируем структуру и веса в ONNX
        torch.onnx.export(
            net, 
            dummy_input, 
            output_onnx_path, 
            export_params=True,
            opset_version=11,
            input_names=['input'],
            output_names=['output']
        )
        
        print("\n==========================================")
        print(" УСПЕХ! Модель успешно скомпилирована в ONNX.")
        print(f" -> Исходный проект: {input_liai_path}")
        print(f" -> Готовый ONNX:     {output_onnx_path}")
        print("==========================================")
        print("Теперь этот .onnx файл можно закидывать в public папку твоего Firebase!")
        
    except Exception as e:
        print("\n❌ Произошла ошибка при конвертации проекта!")
        print(f"Детали ошибки: {e}")

if __name__ == "__main__":
    run_export()