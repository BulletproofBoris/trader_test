#!/bin/bash

# --- НАСТРОЙКИ ---
CMD='python run_walkforward.py --dataset_dir "data/processed/2000_2026_1d_60_10" --bonus_ratio 0.2 --runs 200 --epochs 100 --l2_reg "1e-5" --lr "1e-3" --start_fold "fold_2018" --append'

VRAM_PER_WORKER=3000

# Буфер под ОС и графический интерфейс (в МБ)
OS_BUFFER=1700 

# Пауза между запусками (в секундах). 
# Дает первому процессу время скомпилировать XLA и уйти в быстрые эпохи.
STAGGER_DELAY=10 
# -----------------

echo "🔍 Анализирую ресурсы GPU..."

# Получаем общий объем VRAM через nvidia-smi
TOTAL_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)

if [ -z "$TOTAL_VRAM" ]; then
    echo "❌ Ошибка: Не удалось получить данные от nvidia-smi."
    exit 1
fi

# Вычисляем доступную память и максимальное количество воркеров
AVAILABLE_VRAM=$((TOTAL_VRAM - OS_BUFFER))
MAX_WORKERS=$((AVAILABLE_VRAM / VRAM_PER_WORKER))

# Защита от безумия (не больше 8 воркеров, чтобы не убить CPU)
if [ "$MAX_WORKERS" -gt 8 ]; then
    MAX_WORKERS=8
fi

echo "📊 VRAM всего: ${TOTAL_VRAM} MB"
echo "📊 VRAM доступно (без ОС): ${AVAILABLE_VRAM} MB"
echo "🎯 Рассчитано воркеров: ${MAX_WORKERS} (по ${VRAM_PER_WORKER} MB каждый)"
echo "---------------------------------------------------"

for (( i=1; i<=MAX_WORKERS; i++ ))
do
    SESSION="worker_swarm_$i"
    
    # Проверяем, нет ли уже такой сессии
    tmux has-session -t "$SESSION" 2>/dev/null
    
    if [ $? != 0 ]; then
        echo "🚀 [Воркер $i/$MAX_WORKERS] Создаю сессию: $SESSION"
        tmux new-session -d -s "$SESSION"
        tmux send-keys -t "$SESSION" "$CMD > ${SESSION}_log.txt 2>&1" C-m
        
        # Если это не последний воркер, делаем умную паузу
        if [ "$i" -lt "$MAX_WORKERS" ]; then
            echo "⏳ Жду $STAGGER_DELAY сек, пока XLA-компилятор освободит CPU..."
            sleep $STAGGER_DELAY
        fi
    else
        echo "⚠️ Сессия $SESSION уже существует, пропускаю."
    fi
done

echo "==================================================="
echo "✅ Рой из $MAX_WORKERS процессов успешно запущен!"
echo "👉 Подключиться к первому: tmux attach -t worker_swarm_1"
echo "👉 Посмотреть список всех: tmux ls"