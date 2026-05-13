import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

class ElasticPatienceProfiler(Callback):
    def __init__(self, orchestrator, fold_name, max_epochs, bonus_ratio=0.1, min_delta=0.001,
                 hard_prune_epoch=20,     # Эпоха включения гильотины
                 hard_prune_z=4.0,        # <-- ТЕПЕРЬ ЭТО ТОЖЕ Z-SCORE (например, 4 сигмы)
                 max_z_score=2.0):      # Множитель разброса для Z-Score
        super().__init__()
        self.orchestrator = orchestrator
        self.fold_name = fold_name
        self.max_epochs = max_epochs
        self.epoch_times = []
        self.pruned = False
        
        self.micro_patience = max(1, int(0.1 * max_epochs))
        self.macro_patience = max(3.0, float(0.3 * max_epochs))
        self.macro_bonus = bonus_ratio * self.micro_patience 
        self.min_delta = min_delta 
        
        self.micro_wait = 0
        self.local_best_loss = np.inf # <-- Глобальный якорь лучшего лосса в текущем ране
        self.run_start_time = 0
        self.epoch_start_time = 0
        self.avg_epoch_time = 0.0
        self.overhead_time = 0.0
        self.total_ttc = 0.0
        
        # Настройки жесткости отсева
        self.hard_prune_epoch = hard_prune_epoch
        self.hard_prune_z = hard_prune_z
        self.max_z_score = max_z_score
        
        # ========================================================
        # 🧠 ЭЛИТНАЯ МАТЕМАТИКА ДЛЯ Z-SCORE (Поиск Асимптоты)
        # ========================================================
        valid_losses = []
        try:
            import sqlite3
            # Берем путь к базе напрямую из оркестратора
            db_path = self.orchestrator.db_path 
            
            if os.path.exists(db_path):
                # Открываем короткое соединение только для чтения истории
                with sqlite3.connect(db_path, timeout=10.0) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT val_loss FROM runs WHERE fold = ? AND val_loss IS NOT NULL", (self.fold_name,))
                    valid_losses = [row[0] for row in cursor.fetchall()]
        except Exception as e:
            print(f"⚠️ Не удалось прочитать базу для Elite Pruning: {e}")

        if len(valid_losses) > 5:
            # Берем ТОЛЬКО ТОП-15% лучших результатов (формируем "элиту")
            k = max(3, int(len(valid_losses) * 0.15))
            elite_losses = np.sort(valid_losses)[:k]
            
            self.mu = np.mean(elite_losses) # Наша асимптота (цель)
            self.sigma = np.std(elite_losses) + 1e-6 # Жесткий коридор разброса
        else:
            # Fallback для самых первых ранов
            self.mu = 1.15 
            self.sigma = 0.1
            
        print(f"📐 [Elite Pruning] Элитная цель (Асимптота): {self.mu:.4f} | Допустимый разброс: ±{self.sigma:.4f}")
        
    def on_train_begin(self, logs=None):
        self.run_start_time = time.time()
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        epoch_duration = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_duration)
        
        current_loss = logs.get('val_loss')
        if current_loss is None: return
        
        # Обновляем лучший лосс ДО проверки гильотины, чтобы она знала о рекордах
        if current_loss < self.local_best_loss - self.min_delta:
            self.local_best_loss = current_loss
            self.micro_wait = 0
            self.macro_patience = min(float(self.max_epochs), self.macro_patience + self.macro_bonus)
        else:
            self.micro_wait += 1

        # 🔪 ПРАВИЛО 1: Жесткая Гильотина (динамическая, через дисперсию!)
        guillotine_limit = self.mu + (self.hard_prune_z * self.sigma)
        
        if epoch >= self.hard_prune_epoch and self.local_best_loss > guillotine_limit:
            print(f"\n🔪 [Hard Pruning] Эпоха {epoch}: Лучший Loss {self.local_best_loss:.4f} > лимита {guillotine_limit:.4f} ({self.hard_prune_z} сигм от элиты). Итерация убита.")
            self.model.stop_training = True
            self.pruned = True
            return

        if self.micro_wait >= self.micro_patience:
            # 🎯 ПРАВИЛО 2: Элитный Z-Score (сравниваем ЛУЧШИЙ лосс этого рана)
            z_score = (self.local_best_loss - self.mu) / self.sigma
            
            if z_score > self.max_z_score:
                print(f"\n🔪 [Elite Z-Score] Нет улучшений {self.micro_patience} эпох. ЛУЧШИЙ Loss {self.local_best_loss:.4f} вылетел из элитного коридора (Z={z_score:.2f} > {self.max_z_score}). Итерация убита.")
                self.model.stop_training = True
                self.pruned = True
                return
            else:
                self.micro_wait = 0 
                
        if (epoch + 1) >= int(self.macro_patience):
            print(f"\n⏳ [Early Stopping] Обучение остановлено. Эластичный лимит: {int(self.macro_patience)} эпох.")
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        self.total_ttc = time.time() - self.run_start_time
        clean_epochs = self.epoch_times[1:] if len(self.epoch_times) > 1 else self.epoch_times
        self.avg_epoch_time = float(np.mean(clean_epochs)) if clean_epochs else 0.0
        pure_compute_time = sum(self.epoch_times)
        self.overhead_time = max(0.0, self.total_ttc - pure_compute_time)


class SmartBacktrackCallback(Callback):
    def __init__(self, best_weights_path, monitor_loss='val_loss', factor=0.5, patience=4, min_lr=1e-6, max_rollbacks=3):
        super().__init__()
        self.monitor_loss = monitor_loss
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.max_rollbacks = max_rollbacks
        self.best_weights_path = str(best_weights_path)
        
        self.wait = 0
        self.rollback_count = 0
        self.best_loss = np.inf # <-- Глобально лучший лосс для откатов

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.monitor_loss)
        if current_loss is None: return

        if current_loss < self.best_loss - 1e-4:
            self.best_loss = current_loss
            self.wait = 0
            self.rollback_count = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.rollback_count += 1
                
                if self.rollback_count >= self.max_rollbacks:
                    print(f"\n🛑 Лимит откатов LR исчерпан.")
                    self.model.stop_training = True
                    return

                if os.path.exists(self.best_weights_path):
                    # Откатываемся к ЛУЧШИМ весам, а не просто на шаг назад
                    self.model.load_weights(self.best_weights_path)

                old_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
                if old_lr > self.min_lr:
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    self.model.optimizer.learning_rate.assign(new_lr)
                    self.wait = 0
                    print(f"\n📉 [Backtrack] Откат к ЛУЧШИМ весам. Новый LR: {new_lr}")