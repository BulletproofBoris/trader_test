#!/bin/bash

# 1. Задаем команду (я обернул ее в переменную для красоты и удобства)
CMD='python run_walkforward.py --dataset_dir "data/processed/2000_2026_1d_20_1" --bonus_ratio 0.15 --min_delta 0.001 --runs 120 --batch_size 2048 --epochs 100 --l2_reg "1e-3" --lr "8e-2" --start_fold "fold_2010" --append'

# 2. Список твоих tmux-сессий
SESSIONS=(
    "train_20_1_worker_1"
    "train_20_1_worker_2"
    "train_20_1_worker_3"
    "train_20_1_worker_4"
)

echo "🚀 Запуск Swarm-пула..."

# 3. Перебираем каждую сессию
for SESSION in "${SESSIONS[@]}"
do
    # Проверяем, существует ли уже такая сессия tmux
    tmux has-session -t "$SESSION" 2>/dev/null

    if [ $? != 0 ]; then
        # Если сессии нет, создаем ее в фоновом режиме (-d)
        echo "Создаю новую сессию: $SESSION"
        tmux new-session -d -s "$SESSION"
    else
        echo "Сессия $SESSION уже открыта, использую ее."
    fi

    # Отправляем команду в сессию и имитируем нажатие Enter (C-m)
    tmux send-keys -t "$SESSION" "$CMD" C-m
    
    # Небольшая пауза, чтобы процессы не ломанулись в базу данных в одну и ту же миллисекунду
    sleep 2 
done

echo "✅ Все 4 воркера успешно отправлены в бой!"