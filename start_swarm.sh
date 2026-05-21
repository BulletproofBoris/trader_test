#!/bin/bash

# Генерируем уникальный ID для этого пула воркеров
export SWARM_ID="s_$(date +'%d_%H%M%S')"

# Проверяем, переданы ли аргументы в скрипт
if [ $# -eq 0 ]; then
    echo "⚠️ Аргументы не переданы! Использую дефолтные настройки..."
    ARGS="--dataset_dir data/processed/2000_2026_1d_20_1 --bonus_ratio 0.2 --runs 200 --epochs 100 --l2_reg 1e-5 --lr 4e-3 --start_fold fold_2020 --append"
else
    # Если переданы, берем их все ($@)
    ARGS="$@"
fi

CMD="python run_walkforward.py $ARGS"

# --- НАСТРОЙКИ VRAM ---
VRAM_PER_WORKER=1700
OS_BUFFER=1500 
STAGGER_DELAY=10 
# -----------------

echo "🔍 Анализирую ресурсы GPU..."

TOTAL_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)

if [ -z "$TOTAL_VRAM" ]; then
    echo "❌ Ошибка: Не удалось получить данные от nvidia-smi."
    exit 1
fi

AVAILABLE_VRAM=$((TOTAL_VRAM - OS_BUFFER))
MAX_WORKERS=$((AVAILABLE_VRAM / VRAM_PER_WORKER))

if [ "$MAX_WORKERS" -gt 8 ]; then
    MAX_WORKERS=8
fi

echo "📊 VRAM всего: ${TOTAL_VRAM} MB"
echo "📊 VRAM доступно (без ОС): ${AVAILABLE_VRAM} MB"
echo "🎯 Рассчитано воркеров: ${MAX_WORKERS} (по ${VRAM_PER_WORKER} MB каждый)"
echo "---------------------------------------------------"
echo "🚀 Запуск с аргументами: $ARGS"
echo "🏷️ Идентификатор сессии (Swarm ID): $SWARM_ID"
echo "---------------------------------------------------"

for (( i=1; i<=MAX_WORKERS; i++ ))
do
    SESSION="worker_swarm_$i"
    
    tmux has-session -t "$SESSION" 2>/dev/null
    
    if [ $? != 0 ]; then
        echo "🚀 [Воркер $i/$MAX_WORKERS] Создаю сессию: $SESSION"
        tmux new-session -d -s "$SESSION"
        
        # ЧИСТАЯ СТРОКА С ЦИКЛОМ "КАМИКАДЗЕ" (Без склеек)
        tmux send-keys -t "$SESSION" "$CMD >> ${SESSION}_log.txt 2>&1" C-m
        
        if [ "$i" -lt "$MAX_WORKERS" ]; then
            echo "⏳ Жду $STAGGER_DELAY сек..."
            sleep $STAGGER_DELAY
        fi
    else
        echo "⚠️ Сессия $SESSION уже существует, пропускаю."
    fi
done

echo "==================================================="
echo "✅ Рой из $MAX_WORKERS процессов успешно запущен!"
echo "👉 Подключиться к первому: tmux attach -t worker_swarm_1"