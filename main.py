import sys
import random
import os
import threading
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pygame

# Импортируем инструменты для вызова системного файлового диалога
import tkinter as tk
from tkinter import filedialog

APP_VERSION = "1.2.0"

# Скрываем главное неиспользуемое окно tkinter
root = tk.Tk()
root.withdraw()

# Инициализация Pygame
pygame.init()
pygame.font.init()

# Настройки окна 
WIDTH, HEIGHT = 950, 600  
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(f"Tic-Tac-Toe AI [Training Panel] v{APP_VERSION}")
CLOCK = pygame.time.Clock()

# Цвета
BG_COLOR = (240, 244, 248)
LINE_COLOR = (180, 190, 201)
X_COLOR = (231, 76, 60)      
O_COLOR = (41, 128, 185)     
PANEL_BG = (44, 62, 80)      
TEXT_COLOR = (236, 240, 241)
WHITE = (255, 255, 255)

# Цвета элементов интерфейса
BTN_COLOR = (31, 46, 61)
BTN_HOVER = (51, 66, 80)
PROGRESS_COLOR = (46, 204, 113)
INPUT_BG = (27, 38, 49)        

SLIDER_BG = (31, 46, 61)       
SLIDER_HANDLE = (26, 188, 156) 
SLIDER_LINE = (52, 152, 219)   

INPUT_ACTIVE_COLOR = (46, 204, 113)
TOGGLE_ON_COLOR = (46, 204, 113)
TOGGLE_OFF_COLOR = (231, 76, 60)
TAB_ACTIVE_COLOR = (26, 188, 156)
TAB_INACTIVE_COLOR = (52, 73, 94)

# Шрифты
FONT = pygame.font.SysFont("Arial", 20)
FONT_BOLD = pygame.font.SysFont("Arial", 20, bold=True)
FONT_SMALL = pygame.font.SysFont("Arial", 14)
FONT_BIG = pygame.font.SysFont("Arial", 28, bold=True)

# ==========================================
# Настройки конфигурации ИИ
# ==========================================
cfg = {
    "reward_win": 1.0,
    "reward_draw": 0.1,
    "reward_loss": -1.0,
    "reward_zevok": -2.0,  
    "learning_rate": 0.001,
    "temperature": 0.7,
    "train_on_human": False,
    "use_time_decay": False,       
    "win_speed_factor": 0.15,      
    "loss_survival_factor": 0.15,
    
    # Новые поля метаданных проекта
    "meta_author": "Limehcik",
    "meta_model_version": "1.0"
}

CURRENT_MODEL_PATH = "best_tic_tac_toe_ai.liai"
active_tab = "info"  
total_background_epochs = 0  

# ==========================================
# 1. НЕЙРОСЕТЬ И ПОТОК ОБУЧЕНИЯ
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

net = TicTacToeNet().to(device)
optimizer = optim.Adam(net.parameters(), lr=cfg["learning_rate"])
criterion = nn.MSELoss()

training_progress = 0
training_total = 0
is_training = False
lock = threading.Lock()

stats = {
    "games_played": 0,
    "ai_wins": 0,
    "human_wins": 0,
    "draws": 0,
    "last_loss": 0.0,
    "ai_status": "Ready",
    "device_name": f"Core: {str(device).upper()}"
}

def train_step(state, action, target_val):
    optimizer.zero_grad()
    state_tensor = torch.FloatTensor(state).to(device)
    current_q = net(state_tensor)
    
    target_q = current_q.clone().detach()
    target_q[action] = target_val
    
    loss = criterion(current_q, target_q)
    loss.backward()
    optimizer.step()
    stats["last_loss"] = float(loss.item())

def get_tempered_move(q_vals, available_moves, temp):
    masked_q = q_vals.clone()
    for i in range(9):
        if i not in available_moves:
            masked_q[i] = -9999.0
            
    temp = max(0.01, temp)
    scaled_q = masked_q / temp
    probabilities = F.softmax(scaled_q, dim=0).cpu().numpy()
    probabilities /= probabilities.sum()
    return random.choices(range(9), weights=probabilities, k=1)[0]

def bg_training_worker(episodes, current_cfg):
    global training_progress, is_training, total_background_epochs
    is_training = True
    stats["ai_status"] = "Training..."
    
    with lock:
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_cfg["learning_rate"]
            
    for ep in range(episodes):
        board_sim = [0] * 9
        current_player = 1
        history = []
        
        while True:
            available_moves = [i for i, x in enumerate(board_sim) if x == 0]
            if not available_moves: break
            
            with lock:
                net.eval()
                state_tensor = torch.FloatTensor(board_sim).to(device)
                with torch.no_grad():
                    q_vals = net(state_tensor)
            
            move = get_tempered_move(q_vals, available_moves, current_cfg["temperature"])
            history.append((list(board_sim), move, current_player))
            board_sim[move] = current_player
            
            win_states = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
            winner = None
            for win in win_states:
                if board_sim[win[0]] == board_sim[win[1]] == board_sim[win[2]] != 0:
                    winner = board_sim[win[0]]
            if winner is None and 0 not in board_sim: winner = 0
            
            if winner is not None:
                with lock:
                    net.train()
                    for state, act, plr in history:
                        if winner == 0:
                            reward = current_cfg["reward_draw"]
                        else:
                            if winner == plr:
                                reward = current_cfg["reward_win"]
                                if current_cfg["use_time_decay"]:
                                    # Считаем, сколько ходов сделал именно этот победитель
                                    moves_count = sum(1 for _, _, p in history if p == plr)
                                    moves_count = max(1, moves_count)
                                    speed_factor = 1.5 - (moves_count * current_cfg["win_speed_factor"])
                                    reward *= max(0.5, min(speed_factor, 1.5))
                            else:
                                reward = current_cfg["reward_loss"]
                                if current_cfg["use_time_decay"]:
                                    # Считаем ходы проигравшего
                                    moves_count = sum(1 for _, _, p in history if p == plr)
                                    moves_count = max(1, moves_count)
                                    survival_factor = 0.4 + (moves_count * current_cfg["loss_survival_factor"])
                                    # Быстрая смерть (мало ходов) -> штраф умножается на (2.0 - survival_factor) -> штраф сильнее
                                    reward *= (2.0 - max(0.5, min(survival_factor, 1.5)))
                                    
                        train_step(state, act, reward)
                break
            current_player = -current_player
            
        training_progress = ep + 1
        total_background_epochs += 1

    is_training = False
    stats["ai_status"] = "Ready"

def start_async_training(episodes=500):
    global training_progress, training_total, is_training
    if is_training: return 
    
    training_progress = 0
    training_total = episodes
    current_cfg = cfg.copy()
    threading.Thread(target=bg_training_worker, args=(episodes, current_cfg), daemon=True).start()

# ==========================================
# 2. КОМПОНЕНТЫ ИНТЕРФЕЙСА (UI)
# ==========================================
class Button:
    def __init__(self, x, y, w, h, text, callback):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.callback = callback

    def draw(self, screen):
        mouse_pos = pygame.mouse.get_pos()
        color = BTN_HOVER if self.rect.collidepoint(mouse_pos) else BTN_COLOR
        pygame.draw.rect(screen, color, self.rect, border_radius=5)
        
        txt_surf = FONT_SMALL.render(self.text, True, WHITE)
        txt_rect = txt_surf.get_rect(center=self.rect.center)
        screen.blit(txt_surf, txt_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.callback()

class TabButton:
    def __init__(self, x, y, w, h, text, tab_id):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.tab_id = tab_id

    def draw(self, screen):
        color = TAB_ACTIVE_COLOR if active_tab == self.tab_id else TAB_INACTIVE_COLOR
        pygame.draw.rect(screen, color, self.rect, border_top_left_radius=6, border_top_right_radius=6)
        
        txt_surf = FONT_BOLD.render(self.text, True, WHITE)
        txt_rect = txt_surf.get_rect(center=self.rect.center)
        screen.blit(txt_surf, txt_rect)

    def handle_event(self, event):
        global active_tab
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                active_tab = self.tab_id

class ToggleButton:
    def __init__(self, x, y, w, h, label, cfg_key):
        self.rect = pygame.Rect(x, y, w, h)
        self.label = label
        self.cfg_key = cfg_key

    def draw(self, screen):
        state_color = TOGGLE_ON_COLOR if cfg[self.cfg_key] else TOGGLE_OFF_COLOR
        state_text = "ВКЛЮЧЕНО" if cfg[self.cfg_key] else "ВЫКЛЮЧЕНО"
        
        lbl_surf = FONT_SMALL.render(self.label, True, TEXT_COLOR)
        screen.blit(lbl_surf, (self.rect.x, self.rect.y - 18))
        
        pygame.draw.rect(screen, state_color, self.rect, border_radius=5)
        txt_surf = FONT_BOLD.render(state_text, True, WHITE)
        txt_rect = txt_surf.get_rect(center=self.rect.center)
        screen.blit(txt_surf, txt_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                cfg[self.cfg_key] = not cfg[self.cfg_key]

class SliderWithInput:
    def __init__(self, x, y, w, min_val, max_val, current_val, label):
        self.slider_rect = pygame.Rect(x, y, w, 12)  
        self.min_val = min_val
        self.max_val = max_val
        self.val = current_val
        self.label = label
        self.grabbed = False
        
        self.input_rect = pygame.Rect(x + w + 15, y - 8, 80, 28) 
        self.is_active = False
        self.update_text_from_val()
        self.update_handle_pos()

    def update_handle_pos(self):
        clamped_val = max(self.min_val, min(self.val, self.max_val))
        ratio = (clamped_val - self.min_val) / (self.max_val - self.min_val)
        self.handle_x = self.slider_rect.x + int(ratio * self.slider_rect.width)

    def update_text_from_val(self):
        if self.max_val <= 0.02:  
            self.text_input = f"{self.val:.4f}"
        else:
            self.text_input = f"{self.val:.2f}"

    def draw(self, screen):
        lbl_surf = FONT_SMALL.render(self.label, True, TEXT_COLOR)
        screen.blit(lbl_surf, (self.slider_rect.x, self.slider_rect.y - 18))
        
        pygame.draw.rect(screen, SLIDER_BG, self.slider_rect, border_radius=4)
        fill_w = self.handle_x - self.slider_rect.x
        if fill_w > 0:
            pygame.draw.rect(screen, SLIDER_LINE, (self.slider_rect.x, self.slider_rect.y, fill_w, self.slider_rect.height), border_radius=4)
            
        pygame.draw.circle(screen, SLIDER_HANDLE, (self.handle_x, self.slider_rect.y + 6), 10)
        
        box_border_color = INPUT_ACTIVE_COLOR if self.is_active else BTN_COLOR
        pygame.draw.rect(screen, INPUT_BG, self.input_rect, border_radius=5)
        pygame.draw.rect(screen, box_border_color, self.input_rect, width=2, border_radius=5)
        
        txt_surf = FONT_SMALL.render(self.text_input, True, WHITE)
        txt_rect = txt_surf.get_rect(center=self.input_rect.center)
        screen.blit(txt_surf, txt_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.input_rect.collidepoint(event.pos):
                self.is_active = True
            else:
                if self.is_active: self.submit_text_value()
                self.is_active = False
                
            check_slider_rect = pygame.Rect(self.slider_rect.x, self.slider_rect.y - 8, self.slider_rect.width, 28)
            if check_slider_rect.collidepoint(event.pos):
                self.grabbed = True
                self.update_val_from_mouse(event.pos[0])
                
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.grabbed = False
        elif event.type == pygame.MOUSEMOTION and self.grabbed:
            self.update_val_from_mouse(event.pos[0])
        elif event.type == pygame.KEYDOWN and self.is_active:
            if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                self.submit_text_value()
                self.is_active = False
            elif event.key == pygame.K_BACKSPACE:
                self.text_input = self.text_input[:-1]
            else:
                if event.unicode in "0123456789.-": self.text_input += event.unicode

    def update_val_from_mouse(self, mouse_x):
        mouse_x = max(self.slider_rect.x, min(mouse_x, self.slider_rect.x + self.slider_rect.width))
        ratio = (mouse_x - self.slider_rect.x) / self.slider_rect.width
        self.val = self.min_val + ratio * (self.max_val - self.min_val)
        self.update_handle_pos()
        self.update_text_from_val()

    def submit_text_value(self):
        try:
            parsed_val = float(self.text_input)
            self.val = max(self.min_val, min(parsed_val, self.max_val))
        except ValueError: pass
        self.update_text_from_val()
        self.update_handle_pos()

class TextInputBox:
    def __init__(self, x, y, w, h, label, cfg_key):
        self.rect = pygame.Rect(x, y, w, h)
        self.label = label
        self.cfg_key = cfg_key
        self.is_active = False
        self.text = str(cfg[cfg_key])

    def draw(self, screen):
        # Отрисовка названия поля
        lbl_surf = FONT_SMALL.render(self.label, True, TEXT_COLOR)
        screen.blit(lbl_surf, (self.rect.x, self.rect.y - 18))
        
        # Подсветка активного поля
        box_border_color = INPUT_ACTIVE_COLOR if self.is_active else BTN_COLOR
        pygame.draw.rect(screen, INPUT_BG, self.rect, border_radius=5)
        pygame.draw.rect(screen, box_border_color, self.rect, width=2, border_radius=5)
        
        # Отрисовка текста внутри
        txt_surf = FONT_SMALL.render(self.text, True, WHITE)
        txt_rect = txt_surf.get_rect(left=self.rect.x + 8, centery=self.rect.centery)
        screen.blit(txt_surf, txt_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.is_active = True
            else:
                if self.is_active:
                    cfg[self.cfg_key] = self.text  # Сохраняем в конфиг при потере фокуса
                self.is_active = False
                
        elif event.type == pygame.KEYDOWN and self.is_active:
            if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                cfg[self.cfg_key] = self.text
                self.is_active = False
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            else:
                # Ограничим длину, чтобы текст не вылезал за границы инпута
                if len(self.text) < 20 and event.unicode.isprintable():
                    self.text += event.unicode

    def update_from_cfg(self):
        self.text = str(cfg[self.cfg_key])

# ==========================================
# ИНТЕГРАЦИЯ ПРОВОДНИКА И СИНХРОНИЗАЦИИ (ФОРМАТ .LIAI)
# ==========================================
def sync_sliders_with_cfg():
    sliders[0].val = cfg["reward_win"]
    sliders[1].val = cfg["reward_draw"]
    sliders[2].val = cfg["reward_loss"]
    sliders[3].val = cfg["reward_zevok"]
    sliders[4].val = cfg["temperature"]
    sliders[5].val = cfg["learning_rate"]
    sliders[6].val = cfg["win_speed_factor"]
    sliders[7].val = cfg["loss_survival_factor"]
    input_author.update_from_cfg()
    input_version.update_from_cfg()
    for s in sliders:
        s.update_handle_pos()
        s.update_text_from_val()

def save_model_dialog():
    global CURRENT_MODEL_PATH
    file_path = filedialog.asksaveasfilename(
        initialdir=os.getcwd(),
        title="Сохранить проект ИИ",
        defaultextension=".liai",
        filetypes=[("Limehcik AI Model", "*.liai"), ("All Files", "*.*")]
    )
    if not file_path: return
        
    CURRENT_MODEL_PATH = file_path
    
    # Принудительно забираем актуальные строки из полей (если пользователь забыл нажать Enter)
    cfg["meta_author"] = input_author.text
    cfg["meta_model_version"] = input_version.text
    
    keys = ["reward_win", "reward_draw", "reward_loss", "reward_zevok", "temperature", "learning_rate", "win_speed_factor", "loss_survival_factor"]
    for idx, key in enumerate(keys):
        cfg[key] = sliders[idx].val
    
    save_data = {
        "metadata": {
            "format_version": "1.1",
            "app_version": APP_VERSION,  # Ведём версию программы в проекте
            "author": cfg["meta_author"],
            "model_version": cfg["meta_model_version"]
        },
        "config": cfg,
        "stats": {
            "games_played": stats["games_played"],
            "ai_wins": stats["ai_wins"],
            "human_wins": stats["human_wins"],
            "draws": stats["draws"],
            "total_background_epochs": total_background_epochs
        },
        "model_state": net.state_dict()
    }
    
    try:
        with lock:
            torch.save(save_data, CURRENT_MODEL_PATH)
        stats["ai_status"] = "Проект сохранен!"
        update_window_title()
    except Exception as e:
        stats["ai_status"] = "Ошибка сохранения!"
        print(f"Ошибка при сохранении: {e}")

def load_model_dialog():
    global CURRENT_MODEL_PATH, optimizer, total_background_epochs
    file_path = filedialog.askopenfilename(
        initialdir=os.getcwd(),
        title="Выберите файл проекта ИИ",
        filetypes=[("Limehcik AI Model", "*.liai"), ("PyTorch Weights", "*.pth"), ("All Files", "*.*")]
    )
    if not file_path: return
        
    CURRENT_MODEL_PATH = file_path
    
    try:
        with lock:
            checkpoint = torch.load(CURRENT_MODEL_PATH, map_location=device)
            
            if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                net.load_state_dict(checkpoint["model_state"])
                
                if "config" in checkpoint:
                    for key in cfg:
                        if key in checkpoint["config"]:
                            cfg[key] = checkpoint["config"][key]
                
                # Дополнительная проверка старых версий .liai, где метаданных в конфиге еще не было
                if "metadata" in checkpoint:
                    meta = checkpoint["metadata"]
                    cfg["meta_author"] = meta.get("author", "Unknown")
                    cfg["meta_model_version"] = meta.get("model_version", "1.0")
                    # Здесь при желании можно выводить в консоль meta.get("app_version") для отладки
                            
                if "stats" in checkpoint:
                    s = checkpoint["stats"]
                    stats["games_played"] = s.get("games_played", stats["games_played"])
                    stats["ai_wins"] = s.get("ai_wins", stats["ai_wins"])
                    stats["human_wins"] = s.get("human_wins", stats["human_wins"])
                    stats["draws"] = s.get("draws", stats["draws"])
                    total_background_epochs = s.get("total_background_epochs", total_background_epochs)
            else:
                net.load_state_dict(checkpoint)
                # Дефолты для чистых .pth файлов
                cfg["meta_author"] = "Imported Weights"
                cfg["meta_model_version"] = "1.0"
                                
            net.eval()
            for param_group in optimizer.param_groups:
                param_group['lr'] = cfg["learning_rate"]
                
        stats["ai_status"] = "Проект загружен!"
        update_window_title()
        sync_sliders_with_cfg()
    except Exception as e:
        stats["ai_status"] = "Ошибка чтения!"
        print(f"Ошибка при загрузке: {e}")

def trigger_train(): 
    start_async_training(500)

def reset_weights():
    global net, optimizer, total_background_epochs
    if is_training: return
    with lock:
        net = TicTacToeNet().to(device)
        optimizer = optim.Adam(net.parameters(), lr=cfg["learning_rate"])
    stats["ai_status"] = "Матрица сброшена!"
    stats["last_loss"] = 0.0
    total_background_epochs = 0
    stats["games_played"] = 0
    stats["ai_wins"] = 0
    stats["human_wins"] = 0
    stats["draws"] = 0

def get_weights_summary():
    try:
        with lock:
            w0 = net.model[0].weight.data
            w2 = net.model[2].weight.data
            w4 = net.model[4].weight.data
        return {
            "l0": (float(w0.mean()), float(w0.max()), float(w0.min())),
            "l2": (float(w2.mean()), float(w2.max()), float(w2.min())),
            "l4": (float(w4.mean()), float(w4.max()), float(w4.min()))
        }
    except:
        return None

# Инициализация интерфейса 
tab_buttons = [
    TabButton(470, 115, 145, 35, "ИНФО", "info"),
    TabButton(625, 115, 145, 35, "ОБУЧЕНИЕ", "train"),
    TabButton(780, 115, 145, 35, "УПРАВЛЕНИЕ", "manage")
]

sliders = [
    SliderWithInput(490, 175, 280, 0.0, 3.0, cfg["reward_win"], "Награда: ПОБЕДА"),
    SliderWithInput(490, 219, 280, -2.0, 2.0, cfg["reward_draw"], "Награда: НИЧЬЯ"),
    SliderWithInput(490, 263, 280, -3.0, 0.0, cfg["reward_loss"], "Штраф: ПРОИГРЫШ"),
    SliderWithInput(490, 307, 280, -5.0, 0.0, cfg["reward_zevok"], "Штраф: ЗЕВОК (Игнор линий)"),
    SliderWithInput(490, 351, 280, 0.01, 3.0, cfg["temperature"], "Температура ИИ (Случайность)"),
    SliderWithInput(490, 395, 280, 0.0001, 0.02, cfg["learning_rate"], "Скорость обучения (LR)"),
    SliderWithInput(490, 439, 280, 0.0, 0.5, cfg["win_speed_factor"], "Множитель скорости победы"),
    SliderWithInput(490, 483, 280, 0.0, 0.5, cfg["loss_survival_factor"], "Множитель времени выживания")
]

toggle_train = ToggleButton(490, 532, 130, 24, "Обучение в игре", "train_on_human")
toggle_decay = ToggleButton(635, 532, 130, 24, "Динамика времени", "use_time_decay")
btn_train_bg = Button(775, 520, 145, 30, "Обучить 500 игр", trigger_train)

# Кнопки изменены под новый формат
# Новые поля для редактирования метаданных проекта
input_author = TextInputBox(490, 200, 200, 30, "Автор проекта:", "meta_author")
input_version = TextInputBox(710, 200, 200, 30, "Версия модели:", "meta_model_version")

# Сдвигаем кнопки импорта/экспорта чуть ниже под инпуты
btn_save = Button(490, 255, 420, 40, "Экспортировать проект (.liai)", save_model_dialog)
btn_load = Button(490, 310, 420, 40, "Импортировать проект (.liai / .pth)", load_model_dialog)

btn_reset = Button(490, 430, 420, 45, "СБРОСИТЬ ВСЮ НЕЙРОСЕТЬ", reset_weights)

# ==========================================
# 3. ИГРОВАЯ ЛОГИКА
# ==========================================
board = [0] * 9
game_history = []
human_player = 1  
current_turn = 1  
game_over = False
winner_msg = ""
raw_q_values = [0.0] * 9  

def check_winner():
    win_states = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
    for win in win_states:
        if board[win[0]] == board[win[1]] == board[win[2]] != 0: return board[win[0]]
    if 0 not in board: return 0
    return None

def reset_game():
    global board, game_history, current_turn, game_over, winner_msg, human_player
    board = [0] * 9
    game_history = []
    game_over = False
    winner_msg = ""
    human_player = 1 if random.random() > 0.5 else -1
    current_turn = 1 

def draw_grid():
    for i in range(1, 3):
        pygame.draw.line(SCREEN, LINE_COLOR, (i * 150, 75), (i * 150, 525), 5)
        pygame.draw.line(SCREEN, LINE_COLOR, (0, i * 150 + 75), (450, i * 150 + 75), 5)

def draw_shapes():
    for i, val in enumerate(board):
        row = i // 3
        col = i % 3
        center_x = col * 150 + 75
        center_y = row * 150 + 150
        
        if val == 1: 
            pygame.draw.line(SCREEN, X_COLOR, (center_x - 40, center_y - 40), (center_x + 40, center_y + 40), 8)
            pygame.draw.line(SCREEN, X_COLOR, (center_x + 40, center_y - 40), (center_x - 40, center_y + 40), 8)
        elif val == -1: 
            pygame.draw.circle(SCREEN, O_COLOR, (center_x, center_y), 45, 8)

        if not game_over and current_turn == -human_player:
            q_text = FONT_SMALL.render(f"Q: {raw_q_values[i]:.2f}", True, (120, 130, 140))
            SCREEN.blit(q_text, (col * 150 + 12, row * 150 + 85))

def draw_sidebar():
    pygame.draw.rect(SCREEN, PANEL_BG, (450, 0, 500, HEIGHT))
    
    title = FONT_BIG.render("AI SYSTEM PANEL", True, WHITE)
    SCREEN.blit(title, (470, 15))
    
    file_txt = FONT_SMALL.render(f"Файл: {os.path.basename(CURRENT_MODEL_PATH)}  |  {stats['device_name']}", True, (149, 165, 166))
    SCREEN.blit(file_txt, (470, 48))
    
    if is_training and training_total > 0:
        progress_width = int((training_progress / training_total) * 415)
        pygame.draw.rect(SCREEN, (34, 49, 63), (470, 75, 415, 16), border_radius=4)
        pygame.draw.rect(SCREEN, PROGRESS_COLOR, (470, 75, progress_width, 16), border_radius=4)
        prog_txt = FONT_SMALL.render(f"{training_progress}/{training_total}", True, WHITE)
        SCREEN.blit(prog_txt, (895, 74))
    else:
        status_txt = FONT.render(f"Статус: {stats['ai_status']}  |  Loss: {stats['last_loss']:.6f}", True, TEXT_COLOR)
        SCREEN.blit(status_txt, (470, 74))
    
    for tab_btn in tab_buttons: tab_btn.draw(SCREEN)
    
    pygame.draw.rect(SCREEN, (52, 73, 94), (470, 150, 460, 415), border_radius=6)
    
    if active_tab == "info":
        stats_title = FONT_BOLD.render("АНАЛИТИКА ТЕКУЩИХ МАТЧЕЙ", True, WHITE)
        SCREEN.blit(stats_title, (490, 175))
        
        winrate = (stats["human_wins"] / stats["games_played"] * 100) if stats["games_played"] > 0 else 0.0
        
        SCREEN.blit(FONT.render(f"Сыграно игр: {stats['games_played']}", True, TEXT_COLOR), (490, 215))
        SCREEN.blit(FONT.render(f"Эпох самообучения: {total_background_epochs}", True, TEXT_COLOR), (490, 245))
        SCREEN.blit(FONT.render(f"Побед человека: {stats['human_wins']} (Винрейт: {winrate:.1f}%)", True, TEXT_COLOR), (490, 275))
        SCREEN.blit(FONT.render(f"Побед ИИ: {stats['ai_wins']}  |  Ничьих: {stats['draws']}", True, TEXT_COLOR), (490, 305))
        
        w_summary = get_weights_summary()
        if w_summary:
            pygame.draw.line(SCREEN, LINE_COLOR, (490, 350), (910, 350), 1)
            weight_title = FONT_BOLD.render("АНАЛИЗ ВЕСОВ", True, WHITE)
            SCREEN.blit(weight_title, (490, 365))
            
            l0_str = f"L1 (9->64): Ср {w_summary['l0'][0]:.3f} | Max {w_summary['l0'][1]:.2f} | Min {w_summary['l0'][2]:.2f}"
            l2_str = f"L2 (64->32): Ср {w_summary['l2'][0]:.3f} | Max {w_summary['l2'][1]:.2f} | Min {w_summary['l2'][2]:.2f}"
            l4_str = f"L3 (32->9): Ср {w_summary['l4'][0]:.3f} | Max {w_summary['l4'][1]:.2f} | Min {w_summary['l4'][2]:.2f}"
            
            SCREEN.blit(FONT.render(l0_str, True, TAB_ACTIVE_COLOR), (490, 400))
            SCREEN.blit(FONT.render(l2_str, True, TAB_ACTIVE_COLOR), (490, 440))
            SCREEN.blit(FONT.render(l4_str, True, TAB_ACTIVE_COLOR), (490, 480))
            
    elif active_tab == "train":
        for s in sliders: s.draw(SCREEN)
        toggle_train.draw(SCREEN)
        toggle_decay.draw(SCREEN)  # ВОТ ТУТ ОНА ТЕПЕРЬ ОТРИСОВЫВАЕТСЯ!
        btn_train_bg.draw(SCREEN)
        
    elif active_tab == "manage":
        input_author.draw(SCREEN)
        input_version.draw(SCREEN)
        btn_save.draw(SCREEN)
        btn_load.draw(SCREEN)
        
        pygame.draw.line(SCREEN, LINE_COLOR, (490, 380), (910, 380), 1)
        danger_title = FONT_BOLD.render("СБРОС ДАННЫХ", True, X_COLOR)
        SCREEN.blit(danger_title, (490, 395))
        btn_reset.draw(SCREEN)

    pygame.draw.rect(SCREEN, PANEL_BG, (0, 0, 450, 75))
    if game_over:
        msg = FONT_BOLD.render(winner_msg, True, PROGRESS_COLOR if "ПОЗДРАВЛЯЕМ" in winner_msg or "ТВОЯ" in winner_msg else X_COLOR if "ИИ" in winner_msg else WHITE)
        txt_rect = msg.get_rect(center=(225, 37))
        SCREEN.blit(msg, txt_rect)
    else:
        turn_str = "ВАШ ХОД (КРЕСТИКИ 'X')" if current_turn == human_player else "ОБСЧИТЫВАЮ ХОД..."
        msg = FONT_BOLD.render(turn_str, True, X_COLOR if current_turn == human_player else O_COLOR)
        txt_rect = msg.get_rect(center=(225, 37))
        SCREEN.blit(msg, txt_rect)

def update_window_title():
    base_title = f"Tic-Tac-Toe AI [Training Panel] v{APP_VERSION}"
    file_name = os.path.basename(CURRENT_MODEL_PATH)
    pygame.display.set_caption(f"{base_title} — {file_name}")

reset_game()

# ==========================================
# 4. СИСТЕМНЫЙ ЦИКЛ
# ==========================================
while True:
    SCREEN.fill(BG_COLOR)
    
    if active_tab == "train":
        keys = ["reward_win", "reward_draw", "reward_loss", "reward_zevok", "temperature", "learning_rate", "win_speed_factor", "loss_survival_factor"]
        for idx, key in enumerate(keys):
            if not sliders[idx].is_active:
                cfg[key] = sliders[idx].val
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
            
        for tab_btn in tab_buttons: tab_btn.handle_event(event)
        
        if active_tab == "train":
            for s in sliders: s.handle_event(event)
            toggle_train.handle_event(event)
            toggle_decay.handle_event(event)  # ТЕПЕРЬ ОНА ОБРАБАТЫВАЕТ КЛИКИ!
            btn_train_bg.handle_event(event)
        elif active_tab == "manage":
            input_author.handle_event(event)
            input_version.handle_event(event)
            btn_save.handle_event(event)
            btn_load.handle_event(event)
            btn_reset.handle_event(event)
            
        if event.type == pygame.MOUSEBUTTONDOWN and not game_over and current_turn == human_player:
            x, y = pygame.mouse.get_pos()
            if x < 450 and 75 <= y < 525: 
                col = x // 150
                row = (y - 75) // 150
                index = row * 3 + col
                
                if index < 9 and board[index] == 0:
                    board[index] = human_player
                    current_turn = -human_player 
                    
        if event.type == pygame.KEYDOWN:
            is_any_input_active = any(s.is_active for s in sliders)
            if event.key == pygame.K_SPACE and game_over and not is_any_input_active:
                reset_game()

    if not game_over:
        winner = check_winner()
        if winner is not None:
            game_over = True
            stats["games_played"] += 1
            
            # Количество ходов ИИ в этой партии с человеком
            ai_moves_count = len(game_history)
            ai_moves_count = max(1, ai_moves_count)
            
            if winner == human_player:
                winner_msg = "ТВОЯ ПОБЕДА!"
                stats["human_wins"] += 1
                if cfg["train_on_human"]:
                    with lock:
                        net.train()
                        win_states = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
                        
                        # Расчет штрафа проигрыша с динамикой времени
                        final_loss_reward = cfg["reward_loss"]
                        if cfg["use_time_decay"]:
                            survival_factor = 0.4 + (ai_moves_count * cfg["loss_survival_factor"])
                            final_loss_reward *= (2.0 - max(0.5, min(survival_factor, 1.5)))
                        
                        for state, action in game_history:
                            was_zevok = False
                            ai_could_win = False
                            for line in win_states:
                                ai_count = sum(1 for idx in line if state[idx] == -human_player)
                                empty_count = sum(1 for idx in line if state[idx] == 0)
                                if ai_count == 2 and empty_count == 1:
                                    ai_could_win = True
                                    break
                            
                            if not ai_could_win:
                                for line in win_states:
                                    human_count = sum(1 for idx in line if state[idx] == human_player)
                                    empty_count = sum(1 for idx in line if state[idx] == 0)
                                    if human_count == 2 and empty_count == 1:
                                        critical_cell = [idx for idx in line if state[idx] == 0][0]
                                        if action != critical_cell:
                                            was_zevok = True
                                            break
                            
                            if was_zevok:
                                train_step(state, action, cfg["reward_zevok"])
                            else:
                                train_step(state, action, final_loss_reward)
                                
            elif winner == -human_player:
                winner_msg = "ПОБЕДА ИИ!"
                stats["ai_wins"] += 1
                if cfg["train_on_human"]:
                    with lock:
                        net.train()
                        
                        # Расчет награды победы с динамикой времени
                        final_win_reward = cfg["reward_win"]
                        if cfg["use_time_decay"]:
                            speed_factor = 1.5 - (ai_moves_count * cfg["win_speed_factor"])
                            final_win_reward *= max(0.5, min(speed_factor, 1.5))
                            
                        for s, a in game_history: 
                            train_step(s, a, final_win_reward)
            else:
                winner_msg = "НИЧЬЯ!"
                stats["draws"] += 1
                if cfg["train_on_human"]:
                    with lock:
                        net.train()
                        for s, a in game_history: train_step(s, a, cfg["reward_draw"])

    if not game_over and current_turn == -human_player:
        pygame.time.wait(180)
        available_moves = [i for i, x in enumerate(board) if x == 0]
        
        with lock:
            net.eval()
            state_tensor = torch.FloatTensor(board).to(device)
            with torch.no_grad():
                q_values = net(state_tensor)
                raw_q_values = q_values.cpu().numpy().tolist() 
            
        ai_move = get_tempered_move(q_values, available_moves, cfg["temperature"])
        
        game_history.append((list(board), ai_move))
        board[ai_move] = -human_player
        current_turn = human_player

    draw_grid()
    draw_shapes()
    draw_sidebar()
    
    pygame.display.flip()
    CLOCK.tick(30)