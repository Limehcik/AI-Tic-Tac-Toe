import os
import torch
import torch.nn as nn

# Импортируем инструменты для вызова системного файлового диалога
import tkinter as tk
from tkinter import filedialog

# Скрываем главное неиспользуемое окно tkinter, чтобы оно не мешало
root = tk.Tk()
root.withdraw()

# 1. Объявляем точно такой же каркас сети, какой был в основной программе
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

def run_psychology_test():
    print("Ожидание выбора файла проекта ИИ...")
    
    # Открываем диалоговое окно по умолчанию на формат .liai
    selected_file = filedialog.askopenfilename(
        initialdir=os.getcwd(),
        title="Выберите файл проекта Limehcik AI (.liai) для глубокого анализа",
        filetypes=[("Limehcik AI Model", "*.liai")]
    )
    
    if not selected_file:
        print("Тестирование отменено пользователем.")
        return

    net = TicTacToeNet()

    try:
        # Загружаем сохраненный контейнер .liai на CPU
        checkpoint = torch.load(selected_file, map_location="cpu")
        
        # Переменная для хранения чистых весов нейросети
        state_dict = None
        
        print("\n" + "="*50)
        print(f" ГЛУБОКИЙ АНАЛИЗ ПРОЕКТА: {os.path.basename(selected_file)}")
        print("="*50)
        
        # Разбираем структуру .liai
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
            
            # Читаем метаданные, конфиг и статистику
            metadata = checkpoint.get("metadata", {})
            cfg = checkpoint.get("config", {})
            stats = checkpoint.get("stats", {})
            
            print("\n📋 ПАСПОРТ И МЕТАДАННЫЕ МОДЕЛИ:")
            print(f"  Формат файла:      {metadata.get('format', 'Limehcik AI Engine')}")
            print(f"  Версия структуры:  {metadata.get('version', '1.0')}")
            print(f"  Автор проекта:     {metadata.get('author', 'Limehcik')}")
            
            print("\n📈 СТАТИСТИКА И ЖИЗНЕННЫЙ ПУТЬ:")
            print(f"  Фоновых эпох обучения:   {stats.get('total_background_epochs', 0)}")
            print(f"  Сыграно матчей в панели: {stats.get('games_played', 0)}")
            ai_w = stats.get('ai_wins', 0)
            hu_w = stats.get('human_wins', 0)
            drw = stats.get('draws', 0)
            total_g = stats.get('games_played', 0)
            winrate = (hu_w / total_g * 100) if total_g > 0 else 0.0
            print(f"  Результаты матчей:       ИИ [{ai_w}] | Человек [{hu_w}] (Винрейт человека: {winrate:.1f}%) | Ничьи [{drw}]")
            
            print("\n🧠 ДНК ХАРАКТЕРА (Текущие настройки наград):")
            print(f"  Награда за победу:     {cfg.get('reward_win', 'Нет данных')}")
            print(f"  Награда за ничью:      {cfg.get('reward_draw', 'Нет данных')}")
            print(f"  Штраф за проигрыш:     {cfg.get('reward_loss', 'Нет данных')}")
            print(f"  Штраф за зевки линий:  {cfg.get('reward_zevok', 'Нет данных')}")
            print(f"  Температура (Риск):    {cfg.get('temperature', 'Нет данных')}")
            print(f"  LR (Пластичность):     {cfg.get('learning_rate', 'Нет данных')}")
            print(f"  Обучение на людях:     {'ВКЛЮЧЕНО' if cfg.get('train_on_human') else 'ВЫКЛЮЧЕНО'}")
        else:
            print("\n⚠️ Внимание: Файл не содержит метаструктуры .liai.")
            print("Попытка прочесть файл напрямую как старые сырые веса (.pth)...")
            state_dict = checkpoint
            
        # Загружаем веса в каркас модели
        net.load_state_dict(state_dict)
        net.eval()
        
        print("\n" + "-"*35)
        print("📐 АНАЛИЗ МАТРИЦЫ ВЕСОВ (Слои сети)")
        print("-"*35)
        for name, param in net.named_parameters():
            if "weight" in name:
                print(f"Слой {name}:")
                print(f"  Среднее значение: {param.data.mean().item():.5f}")
                print(f"  Максимальный вес: {param.data.max().item():.5f}")
                print(f"  Минимальный вес:  {param.data.min().item():.5f}")
                print("  " + "." * 25)
                
        print("\n" + "="*50)
        print(" 🧪 ПСИХОЛОГИЧЕСКИЕ ТЕСТЫ НА СИТУАЦИИ НА ПОЛЕ")
        print("="*50)
        
        # Для симуляции: Человек = 1.0, ИИ = -1.0 (как в твоей игровой логике main.py)
        
        # Тест 1: Пустая доска (ИИ ходит первым)
        empty_board = [0.0] * 9
        with torch.no_grad():
            q_empty = net(torch.FloatTensor(empty_board)).numpy()
            
        print("\n[ТЕСТ 1] Приоритеты ходов на ПУСТОЙ доске (Куда ИИ хочет пойти в начале):")
        for i in range(3):
            print(f"  [{q_empty[i*3]:.2f}] [{q_empty[i*3+1]:.2f}] [{q_empty[i*3+2]:.2f}]")
            
        # Тест 2: Человек занял центр
        # [0, 0, 0]
        # [0, 1, 0]   (1.0 - ход человека)
        # [0, 0, 0]
        center_board = [0.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 0.0]
        with torch.no_grad():
            q_center = net(torch.FloatTensor(center_board)).numpy()
            
        print("\n[ТЕСТ 2] Реакция ИИ, если Человек занял ЦЕНТР поля:")
        for i in range(3):
            print(f"  [{q_center[i*3]:.2f}] [{q_center[i*3+1]:.2f}] [{q_center[i*3+2]:.2f}]")
            
        # Тест 3: Проверка на ЗЕВОК (Человек построил 2 в ряд на верхней линии и готов победить)
        # [1, 1, 0]   (Два крестика человека, ячейка 2 пуста — ИИ ОБЯЗАН закрыть её!)
        # [0, 0, 0]
        # [0, 0, 0]
        threat_board = [1.0, 1.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0]
        with torch.no_grad():
            q_threat = net(torch.FloatTensor(threat_board)).numpy()
            
        print("\n[ТЕСТ 3] Критическая угроза! Человек построил [X][X][ ]. Видит ли ИИ блок (ячейка 2):")
        for i in range(3):
            print(f"  [{q_threat[i*3]:.2f}] [{q_threat[i*3+1]:.2f}] [{q_threat[i*3+2]:.2f}]")
        best_block_move = q_threat.argmax()
        print(f"  👉 ИИ выберет ячейку №{best_block_move} с оценкой {q_threat[best_block_move]:.2f} "
              f"{'— ОТЛИЧНО, БЛОК ВИДИТ!' if best_block_move == 2 else '— ❌ СЗЕВНУЛ! ИИ проигнорировал атаку.'}")

        # Тест 4: Шанс на победу (ИИ сам построил 2 в ряд на нижней линии и может выиграть)
        # [0, 0, 0]
        # [0, 0, 0]
        # [-1, -1, 0] (-1.0 - это нолики ИИ. Ячейка 8 принесет ему победу)
        win_board = [0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  -1.0, -1.0, 0.0]
        with torch.no_grad():
            q_win = net(torch.FloatTensor(win_board)).numpy()
            
        print("\n[ТЕСТ 4] Шанс победить! У ИИ на нижней строке [O][O][ ]. Пойдет ли он на победу (ячейка 8):")
        for i in range(3):
            print(f"  [{q_win[i*3]:.2f}] [{q_win[i*3+1]:.2f}] [{q_win[i*3+2]:.2f}]")
        best_win_move = q_win.argmax()
        print(f"  👉 ИИ выберет ячейку №{best_win_move} с оценкой {q_win[best_win_move]:.2f} "
              f"{'— ИДЕАЛЬНО, ИИ забирает победу!' if best_win_move == 8 else '— ❌ ТУПИТ! ИИ упустил летальный ход.'}")

        print("\n" + "="*50 + "\n")
                
    except Exception as e:
        print(f"\n❌ Ошибка при глубоком анализе проекта!")
        print(f"Детали: {e}")

if __name__ == "__main__":
    run_psychology_test()