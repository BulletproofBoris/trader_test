#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_tools/feature_selector_lstm_from_final_mon_v3.py
Patched: стабильный импорт pandas, фикс Pareto, float32, мелкие улучшения
"""
import os, sys, json, math, time, argparse, warnings
from pathlib import Path
from typing import List, Tuple, Optional

# Снизим болтливость TF
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, regularizers, optimizers, Model, callbacks

EXCLUDE_COLS = {"datetime","ticker","label"}

def sl_print(msg: str, end: str = '', flush: bool = True):
    sys.stdout.write('\r' + msg + end)
    if flush: sys.stdout.flush()

def sl_done():
    sys.stdout.write('\n'); sys.stdout.flush()

class SingleLineStatus(callbacks.Callback):
    def __init__(self, total_epochs: int, patience: int, desc: str = "", disable: bool = False):
        super().__init__()
        self.total_epochs = total_epochs
        self.patience = patience
        self.desc = desc
        self.disable = disable
        self.best_val = float('inf')
        self.wait = 0
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        if not self.disable:
            sl_print(self._fmt_line(0, logs or {}))

    def on_epoch_end(self, epoch, logs=None):
        if self.disable: 
            return
        logs = logs or {}
        vloss = logs.get('val_loss')
        if vloss is not None:
            if vloss < self.best_val - 1e-12:
                self.best_val = vloss
                self.wait = 0
            else:
                self.wait += 1
        sl_print(self._fmt_line(epoch+1, logs))

    def on_train_end(self, logs=None):
        if self.disable: 
            return
        sl_done()

    def _fmt_line(self, epoch, logs):
        elapsed = time.time() - self.start_time if self.start_time else 0.0
        loss = logs.get('loss')
        acc  = logs.get('accuracy')
        vls  = logs.get('val_loss')
        vacc = logs.get('val_accuracy')
        parts = [self.desc] if self.desc else []
        parts += [f"epoch {epoch}/{self.total_epochs}",
                  f"loss={loss:.4f}" if loss is not None else "loss=NA",
                  f"acc={acc:.4f}" if acc is not None else "acc=NA",
                  f"val_loss={vls:.4f}" if vls is not None else "val_loss=NA",
                  f"val_acc={vacc:.4f}" if vacc is not None else "val_acc=NA",
                  f"patience {self.wait}/{self.patience}",
                  f"{elapsed:.1f}s"]
        return " | ".join(parts)

def _read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Файл пустой: {path}")
    return df

def _feature_columns(df: pd.DataFrame, additional_exclude: Optional[List[str]]=None) -> List[str]:
    exclude = set(EXCLUDE_COLS)
    if additional_exclude:
        exclude.update([c for c in additional_exclude if c])
    feats = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if not feats:
        raise ValueError("Не найдены числовые фичи. Проверь входные CSV.")
    return feats

def _remap_labels_inplace(df: pd.DataFrame) -> None:
    uniq = pd.unique(df['label'])
    # Если у нас {-1,0,1} -> сдвигаем к {0,1,2}
    if set(pd.Series(uniq).dropna().tolist()) == {-1,0,1}:
        df['label'] = df['label'].map({-1:0, 0:1, 1:2}).astype(int)
    else:
        df['label'] = df['label'].astype(int)

def _to_sequences_singleline(df: pd.DataFrame, feature_cols: List[str], lookback: int, title: str) -> Tuple[np.ndarray, np.ndarray]:
    required = {'datetime','ticker','label'}
    if not required.issubset(df.columns):
        raise ValueError("Ожидались колонки: ['datetime','ticker','label']")
    X_list, y_list = [], []
    groups = list(df.sort_values(['ticker','datetime']).groupby('ticker', sort=False))
    total = len(groups)
    for i, (_, g) in enumerate(groups, start=1):
        g = g.reset_index(drop=True)
        if len(g) > lookback:
            values = g[feature_cols].to_numpy(dtype=np.float32, copy=False)
            labels = g['label'].to_numpy()
            for j in range(lookback, len(g)):
                seq = values[j-lookback:j, :]
                yv  = labels[j]
                if np.isnan(seq).any() or np.isinf(seq).any():
                    continue
                X_list.append(seq)
                y_list.append(yv)
        pct = int(i * 100 / total) if total > 0 else 100
        sl_print(f"{title}: {i}/{total} tickers ({pct}%) | built={len(X_list)}")
    sl_done()
    if not X_list:
        raise ValueError("После построения последовательностей не осталось примеров. Уменьши --lookback или проверь данные.")
    return np.stack(X_list, axis=0), np.asarray(y_list, dtype=int)

def build_lstm_model(timesteps, num_features, n_classes, hidden=64, dropout=0.2, l2_reg=1e-6, lr=1e-3):
    inp = layers.Input(shape=(timesteps, num_features), name="input")
    x = layers.LSTM(hidden, return_sequences=True, kernel_regularizer=regularizers.l2(l2_reg))(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(max(1, hidden//2), return_sequences=False, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(max(2, hidden//2), activation="relu", kernel_regularizer=regularizers.l2(l2_reg))(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = Model(inp, out)
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def train_eval_subset(X_train, y_train, X_val, y_val, feature_idx, n_classes, build_kwargs,
                      epochs, batch_size, patience, verbose=0, seed=42, single_line=True):
    tf.keras.utils.set_random_seed(seed)
    Xtr = X_train[:, :, feature_idx]
    Xva = X_val[:, :, feature_idx]
    model = build_lstm_model(Xtr.shape[1], Xtr.shape[2], n_classes, **build_kwargs)
    es = callbacks.EarlyStopping(monitor="val_loss", patience=patience, mode="min",
                                 restore_best_weights=True, verbose=0)
    cb = SingleLineStatus(epochs, patience, desc=f"[subset={Xtr.shape[2]}]", disable=not single_line)
    model.fit(Xtr, y_train, validation_data=(Xva, y_val), epochs=epochs,
              batch_size=batch_size, verbose=0, callbacks=[es, cb])
    eval_res = model.evaluate(Xva, y_val, verbose=0, return_dict=True)
    return float(eval_res["loss"]), float(eval_res["accuracy"])

def forward_selection(X_tr, y_tr, X_va, y_va, F, feature_names, n_classes, cfg):
    remaining = list(range(F))
    selected = []
    tried = []
    best_loss, best_acc = float("inf"), -1.0

    warm = cfg.warm_start_idx or []
    for w in warm:
        if w in remaining:
            selected.append(w); remaining.remove(w)

    step = 0
    while remaining and (cfg.max_features is None or len(selected) < cfg.max_features):
        step += 1
        cand = []
        improved = False
        for f in remaining:
            idx = selected + [f]
            vloss, vacc = train_eval_subset(X_tr, y_tr, X_va, y_va, idx, n_classes, cfg.model_kwargs,
                                            cfg.epochs, cfg.batch_size, cfg.patience, cfg.verbose, cfg.seed+step,
                                            single_line=True)
            cand.append((f, vloss, vacc))
            tried.append({"subset": idx.copy(), "names": [feature_names[i] for i in idx],
                          "val_loss": vloss, "val_accuracy": vacc})
        cand.sort(key=lambda t: (-t[2], t[1]))
        f_best, vloss, vacc = cand[0]
        sl_print(f"[SFS step {step}] best_acc={vacc:.4f} | best_loss={vloss:.4f} | subset_size={len(selected)+1}")
        sl_done()
        if (vacc > best_acc) or (math.isclose(vacc, best_acc) and vloss < best_loss):
            best_acc, best_loss = vacc, vloss
            selected.append(f_best); remaining.remove(f_best)
            improved = True
        if not improved:
            break
    return selected, tried

def random_search(X_tr, y_tr, X_va, y_va, F, feature_names, n_classes, cfg):
    rng = np.random.default_rng(cfg.seed)
    tried = []
    maxF = cfg.max_features or F
    sizes = list(range(1, min(F, maxF)+1))
    best_vacc, best_vloss = -1.0, float('inf')
    for it in range(cfg.iterations):
        size = int(rng.choice(sizes))
        idx = sorted(rng.choice(F, size=size, replace=False).tolist())
        vloss, vacc = train_eval_subset(X_tr, y_tr, X_va, y_va, idx, n_classes, cfg.model_kwargs,
                                        cfg.epochs, cfg.batch_size, cfg.patience, cfg.verbose, cfg.seed+1000+it,
                                        single_line=True)
        tried.append({"subset": idx.copy(), "names": [feature_names[i] for i in idx],
                      "val_loss": vloss, "val_accuracy": vacc})
        if (vacc > best_vacc) or (math.isclose(vacc, best_vacc) and vloss < best_vloss):
            best_vacc, best_vloss = vacc, vloss
        sl_print(f"[Random {it+1}/{cfg.iterations}] best_acc={best_vacc:.4f} | best_loss={best_vloss:.4f}")
    sl_done()
    return None, tried

def is_pareto_efficient(points: np.ndarray) -> np.ndarray:
    """
    Возвращает булев массив: True — точки на Парето-фронте.
    Минимизируем обе координаты.
    """
    pts = np.asarray(points, dtype=float)
    n = pts.shape[0]
    eff = np.ones(n, dtype=bool)
    for i in range(n):
        if not eff[i]:
            continue
        # i неэффективна, если существует j, доминирующая i: pts[j] <= pts[i] и строго < хотя бы по одной координате
        dominates_i = (np.all(pts <= pts[i], axis=1) & np.any(pts < pts[i], axis=1))
        dominates_i[i] = False
        if np.any(dominates_i):
            eff[i] = False
    return eff

def normalize(values):
    v = np.array(values, dtype=float)
    v_min, v_max = float(np.min(v)), float(np.max(v))
    if v_max == v_min:
        return np.ones_like(v) * 0.5
    return (v - v_min) / (v_max - v_min)

def choose_best_by_score(df, alpha):
    acc_norm = normalize(df["val_accuracy"].values)
    loss_norm = normalize(df["val_loss"].values)
    score = alpha*(1.0 - acc_norm) + (1.0 - alpha)*loss_norm
    best_idx = int(np.argmin(score))
    df = df.copy()
    df["acc_norm"] = acc_norm
    df["loss_norm"] = loss_norm
    df["score"] = score
    return df, best_idx

def main(args):
    data_dir = Path(args.data_dir)
    train_path = data_dir / "train_final.csv"
    test_path  = data_dir / "test_final.csv"

    train_df = _read_csv_required(train_path)
    if args.val_source == "test":
        val_df = _read_csv_required(test_path)

    for col in ["datetime","ticker","label"]:
        if col not in train_df.columns:
            raise ValueError(f"В {train_path} отсутствует колонка '{col}'")
    train_df['datetime'] = pd.to_datetime(train_df['datetime'])
    _remap_labels_inplace(train_df)

    feature_cols = _feature_columns(train_df, additional_exclude=args.exclude_cols)
    Fnames = feature_cols
    F = len(Fnames)
    print(f"Найдено фич: {F}")

    if args.val_source == "train_split":
        dates_sorted = np.sort(train_df['datetime'].unique())
        if args.val_split_idx_date:
            split_point = pd.to_datetime(args.val_split_idx_date)
        else:
            split_idx = int(len(dates_sorted) * (1.0 - args.val_fraction))
            split_idx = max(1, min(len(dates_sorted)-1, split_idx))
            split_point = pd.Timestamp(dates_sorted[split_idx])
        print(f"Хронологический split в пределах train: {split_point}")
        tr_df = train_df[train_df['datetime'] < split_point].copy()
        va_df = train_df[train_df['datetime'] >= split_point].copy()
    else:
        for col in ["datetime","ticker","label"]:
            if col not in val_df.columns:
                raise ValueError("В test_final.csv отсутствуют нужные колонки ['datetime','ticker','label']")
        val_df['datetime'] = pd.to_datetime(val_df['datetime'])
        _remap_labels_inplace(val_df)
        tr_df = train_df.copy()
        va_df = val_df.copy()
        print("Validation source: test_final.csv")

    if tr_df.empty or va_df.empty:
        raise ValueError("Пустой train или val. Проверь источники и фильтры.")

    X_tr, y_tr = _to_sequences_singleline(tr_df[['datetime','ticker','label']+Fnames], Fnames, args.lookback, "Build sequences (train)")
    X_va, y_va = _to_sequences_singleline(va_df[['datetime','ticker','label']+Fnames], Fnames, args.lookback, "Build sequences (val)")
    n_classes = int(len(np.unique(np.concatenate([y_tr, y_va]))))
    print(f"Datasets: N_train={len(y_tr)}, N_val={len(y_va)}, classes={n_classes} | shapes train {X_tr.shape}, val {X_va.shape}")

    warm_idx = None
    if args.warm_start_json:
        wp = Path(args.warm_start_json)
        if wp.exists():
            warm_names = json.loads(wp.read_text(encoding='utf-8'))
            name2idx = {n:i for i,n in enumerate(Fnames)}
            warm_idx = [name2idx[n] for n in warm_names if n in name2idx]
            print(f"Warm-start features: {len(warm_idx)}")

    class Cfg: pass
    cfg = Cfg()
    cfg.model_kwargs = dict(hidden=args.hidden, dropout=args.dropout, l2_reg=args.l2, lr=args.lr)
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.patience = args.patience
    cfg.seed = args.seed
    cfg.verbose = args.verbose
    cfg.max_features = args.max_features
    cfg.iterations = args.iterations
    cfg.warm_start_idx = warm_idx

    outdir = data_dir / "feature_selection" / args.model_name
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if args.strategy == "forward":
        selected, tried = forward_selection(X_tr, y_tr, X_va, y_va, F, Fnames, n_classes, cfg)
        if selected:
            vloss, vacc = train_eval_subset(X_tr, y_tr, X_va, y_va, selected, n_classes, cfg.model_kwargs,
                                            cfg.epochs, cfg.batch_size, cfg.patience, cfg.verbose, cfg.seed+999,
                                            single_line=True)
            tried.append({"subset": selected.copy(), "names": [Fnames[i] for i in selected],
                          "val_loss": vloss, "val_accuracy": vacc})
    else:
        selected, tried = random_search(X_tr, y_tr, X_va, y_va, F, Fnames, n_classes, cfg)

    rows = []
    for r in tried:
        rows.append({
            "subset_indices": json.dumps(r["subset"], ensure_ascii=False),
            "subset_size": len(r["subset"]),
            "subset_names": json.dumps(r.get("names"), ensure_ascii=False) if r.get("names") is not None else None,
            "val_loss": r["val_loss"],
            "val_accuracy": r["val_accuracy"],
        })

    df_res = pd.DataFrame(rows)
    df_res.to_csv(outdir / "results.csv", index=False)

    # Точки: (val_loss, -val_accuracy) — обе минимизируем
    pts = np.stack([df_res["val_loss"].values, -df_res["val_accuracy"].values], axis=1)
    mask = is_pareto_efficient(pts)
    pareto_df = df_res.loc[mask].copy().sort_values(["val_loss","val_accuracy"], ascending=[True, False])
    pareto_df.to_csv(outdir / "pareto.csv", index=False)

    df_scored, best_idx = choose_best_by_score(df_res, alpha=args.alpha)
    df_scored.to_csv(outdir / "results_scored.csv", index=False)

    best_row = df_scored.iloc[best_idx]
    best_subset = json.loads(best_row["subset_indices"])
    best_names  = [Fnames[i] for i in best_subset]

    out_json = data_dir / f"selected_features_{args.model_name}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(best_names, f, ensure_ascii=False, indent=2)

    np.save(outdir / "selected_indices.npy", np.array(best_subset, dtype=int))
    with open(outdir / "selected_names.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(best_names))

    print("=== Completed ===")
    print(f"Strategy: {args.strategy}")
    print(f"Tried subsets: {len(df_res)} | Pareto subsets: {pareto_df.shape[0]}")
    print(f"Saved canonical feature list to: {out_json}")
    print(f"Time: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True, help="Имя для selected_features_<model_name>.json")
    ap.add_argument("--data_dir", required=True, help="Папка с train_final.csv/test_final.csv")
    ap.add_argument("--strategy", choices=["forward","random"], default="forward")
    ap.add_argument("--max_features", type=int, default=None)
    ap.add_argument("--iterations", type=int, default=120)
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--val_source", choices=["train_split","test"], default="train_split")
    ap.add_argument("--val_fraction", type=float, default=0.2)
    ap.add_argument("--val_split_idx_date", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--l2", type=float, default=1e-6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--exclude_cols", type=lambda s: s.split(","), default=None)
    ap.add_argument("--warm_start_json", type=str, default=None)
    args = ap.parse_args()
    main(args)
