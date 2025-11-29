# run_inmemory.py — Cross-Project Ready (CPU friendly)
# Modes CV:  lopo | stratk | sgkf | none
# Calibration: TempScaling (fallback Isotonic)
# Seuils: PPR (quantile) OU F1/F0.5
# Augmentation bug (TRAIN only): bruit gaussien
# Scalers fit seulement sur TRAIN/FOLD (jamais VAL/TEST)
# Entraînement final sur TRAIN+VAL, éval sur TEST
#
# EXEMPLES
# 1) SGKF (recommandé si pas de GPU, proxy LOPO rapide)
# py run_inmemory.py --projects_dir .\out --train ant camel poi synapse --val log4j --test jedit ^
#   --cv sgkf --n_splits 3 --preset fast_cpu --use_focal --ppr_target 0.60 --calib temp --scaler quantile
#
# 2) LOPO (plus lent, éval canonique cross-project)
# py run_inmemory.py --projects_dir .\out --train ant camel poi --val log4j --test jedit ^
#   --cv lopo --preset balanced --use_focal --ppr_target 0.55 --calib temp --scaler quantile
#
# 3) Sans CV (fallback rapide)
# py run_inmemory.py --projects_dir .\out --train ant camel poi --val log4j --test jedit ^
#   --cv none --preset fast_cpu --use_focal --ppr_target 0.60 --calib temp --scaler standard

import argparse, json, warnings, itertools, sys
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import (
    precision_recall_curve, roc_curve, classification_report,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_score, recall_score, f1_score
)

import tensorflow as tf
from tensorflow.keras import layers as L, Model, regularizers, optimizers, callbacks
import matplotlib.pyplot as plt

# ======================== Constantes / Presets ========================

ID_COLS = ["project","version","class","bug"]
SHOW_FIGS = False  # activé par --show_figs

PRESETS = {
    "fast_cpu":       dict(d_model=96,  n_layers=2, n_heads=2, d_ff=192, dropout=0.30,
                           lr=2e-4, weight_decay=2e-4, label_smoothing=0.06,
                           focal_alpha=0.9, focal_gamma=1.0, batch_size=32, epochs=35),
    "balanced":       dict(d_model=128, n_layers=4, n_heads=4, d_ff=256, dropout=0.30,
                           lr=3e-4, weight_decay=1e-4, label_smoothing=0.04,
                           focal_alpha=1.0, focal_gamma=0.75, batch_size=32, epochs=70),
    "high_recall":    dict(d_model=128, n_layers=5, n_heads=4, d_ff=256, dropout=0.30,
                           lr=3e-4, weight_decay=1e-4, label_smoothing=0.03,
                           focal_alpha=0.6, focal_gamma=0.7, batch_size=28, epochs=80),
    "high_precision": dict(d_model=96,  n_layers=3, n_heads=2, d_ff=256, dropout=0.20,
                           lr=2e-4, weight_decay=5e-5, label_smoothing=0.08,
                           focal_alpha=0.35, focal_gamma=1.00, batch_size=48, epochs=60),
}

rng = np.random.default_rng(42)

# ======================== IO helpers ========================

def load_csv(p: Path) -> pd.DataFrame:
    for enc in ("utf-8","utf-8-sig","latin1"):
        for sep in (",",";","\t"):
            try:
                return pd.read_csv(p, encoding=enc, sep=sep)
            except Exception:
                pass
    return pd.read_csv(p)

def resolve_project_file(projects_dir: Path, name: str) -> Path | None:
    candidates = [
        projects_dir / f"data_{name}.csv",
        projects_dir / f"{name}_matrice_final.csv",
        projects_dir / f"{name}.csv",
    ]
    for c in candidates:
        if c.exists(): return c
    return None

def assemble(projects_dir: Path, names: list[str]) -> pd.DataFrame:
    dfs = []
    for n in names:
        f = resolve_project_file(projects_dir, n)
        if f is None:
            print(f"[WARN] introuvable: {n} (data_{n}.csv | {n}_matrice_final.csv | {n}.csv)")
            continue
        df = load_csv(f)
        dfs.append(df)
        print(f"[OK] {n}: {len(df)} lignes ({f.name})")
    if not dfs:
        raise SystemExit("[ERROR] aucun projet valide fourni.")
    return pd.concat(dfs, ignore_index=True)

def infer_columns(df: pd.DataFrame):
    rel = [c for c in df.columns if str(c).startswith("deg_")]
    met = [c for c in df.columns if c not in set(ID_COLS) | set(rel)]
    met = [c for c in met if pd.api.types.is_numeric_dtype(df[c])]
    return met, rel

def binarize_bug(s: pd.Series) -> np.ndarray:
    s = pd.to_numeric(s, errors="coerce").fillna(0)
    return (s.astype(float) > 0).astype(int).values

def split_X_y_ids(df: pd.DataFrame, metric_cols, relation_cols):
    y  = binarize_bug(df["bug"])
    Xm = df[metric_cols].astype(float).fillna(0.0).values if metric_cols else np.zeros((len(df),0))
    Xr = df[relation_cols].astype(float).fillna(0.0).values if relation_cols else np.zeros((len(df),0))
    ids = dict(
        project=df["project"].astype(str).values if "project" in df.columns else np.array([""]*len(df)),
        version=df["version"].astype(str).values if "version" in df.columns else np.array([""]*len(df)),
        clazz=df["class"].astype(str).values if "class" in df.columns else np.array([""]*len(df)),
    )
    return Xm, Xr, y, ids

# ======================== Modèle Transformer ========================

def to_onehot(y):
    y = y.astype(int)
    oh = np.zeros((len(y), 2), dtype=np.float32)
    oh[np.arange(len(y)), y] = 1.0
    return oh

def focal_loss(alpha=0.45, gamma=0.75):
    @tf.function
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        pt = tf.reduce_sum(y_true * y_pred, axis=-1)
        w = alpha * tf.pow(1.0 - pt, gamma)
        return tf.reduce_mean(-w * tf.math.log(pt))
    return loss

def transformer_block(x, d_model, n_heads, d_ff, dropout):
    h = L.LayerNormalization()(x)
    h = L.MultiHeadAttention(num_heads=n_heads, key_dim=max(1, d_model//n_heads))(h, h)
    h = L.Dropout(dropout)(h); x = L.Add()([x, h])
    h = L.LayerNormalization()(x)
    h = L.Dense(d_ff, activation="gelu")(h)
    h = L.Dropout(dropout)(h); h = L.Dense(d_model)(h)
    return L.Add()([x, h])

def build_transformer(n_rel_dims: int, n_met_dims: int,
                      d_model=128, n_layers=4, n_heads=4, d_ff=256, dropout=0.2, l2reg=1e-5):
    inputs, tokens = [], []
    if n_rel_dims > 0:
        inp_rel = L.Input(shape=(n_rel_dims,), name="relations")
        inputs.append(inp_rel)
        x_rel = L.Lambda(lambda t: tf.expand_dims(t, axis=-1))(inp_rel)
        x_rel = L.Conv1D(filters=d_model, kernel_size=1, padding="valid",
                         activation="linear", name="rel_proj_conv1x1")(x_rel)
        tokens.append(x_rel)
    if n_met_dims > 0:
        inp_met = L.Input(shape=(n_met_dims,), name="metrics")
        inputs.append(inp_met)
        met_proj  = L.Dense(d_model, activation="linear", name="met_proj")(inp_met)
        met_token = L.Lambda(lambda z: tf.expand_dims(z, axis=1), name="metrics_token")(met_proj)
        tokens.append(met_token)
    if not tokens: raise ValueError("Aucune feature en entrée.")
    x = tokens[0] if len(tokens)==1 else L.Concatenate(axis=1, name="concat_tokens")(tokens)
    for _ in range(n_layers):
        x = transformer_block(x, d_model, n_heads, d_ff, dropout)
    x = L.LayerNormalization(name="pre_head_norm")(x)
    x = L.GlobalAveragePooling1D(name="gap")(x)
    x = L.Dropout(dropout, name="head_dropout")(x)
    out = L.Dense(2, activation="softmax", kernel_regularizer=regularizers.l2(l2reg), name="head")(x)
    return Model(inputs=inputs, outputs=out)

def pack_input(Xr, Xm):
    if Xr.shape[1]>0 and Xm.shape[1]>0: return [Xr, Xm]
    if Xr.shape[1]>0: return Xr
    return Xm

# ======================== Calibration & Prior-shift ========================

def _to_logits_from_probs(p):
    eps = np.finfo(np.float32).eps
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p) - np.log(1.0 - p)

def _binary_nll(y_true, p):
    eps = 1e-12
    return -np.mean(y_true*np.log(np.clip(p,eps,1-eps)) + (1-y_true)*np.log(np.clip(1-p,eps,1-eps)))

def fit_temperature_scaling_grid(y_true, p_val):
    z = _to_logits_from_probs(p_val)
    def apply_T(T): return 1.0 / (1.0 + np.exp(-z / T))
    Ts = np.concatenate([np.linspace(0.05,0.5,10), np.linspace(0.5,2.0,16), np.linspace(2.0,10.0,9)])
    best_T, best_nll = 1.0, 1e9
    for T in Ts:
        nll = _binary_nll(y_true, apply_T(T))
        if nll < best_nll: best_nll, best_T = nll, T
    around = np.linspace(max(0.05, best_T*0.5), min(10.0, best_T*1.5), 15)
    for T in around:
        nll = _binary_nll(y_true, apply_T(T))
        if nll < best_nll: best_nll, best_T = nll, T
    return float(max(0.05, min(10.0, best_T)))

def apply_temperature_scaling(p, T):
    z = _to_logits_from_probs(p)
    out = 1.0 / (1.0 + np.exp(-z / T))
    return np.clip(out, 1e-6, 1-1e-6)

def fit_isotonic(y_true, p_val):
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    ir.fit(p_val, y_true)
    return ir

def apply_isotonic(ir, p):
    return np.clip(ir.predict(p), 1e-6, 1-1e-6)

def correct_prior_shift(p, pi_train, pi_test):
    def logit(x):
        x = np.clip(x, 1e-6, 1-1e-6)
        return np.log(x) - np.log(1-x)
    z = _to_logits_from_probs(p)
    adj = logit(pi_test) - logit(pi_train)
    z_adj = z + adj
    out = 1.0 / (1.0 + np.exp(-z / T))
    return np.clip(1.0 / (1.0 + np.exp(-z_adj)), 1e-6, 1-1e-6)

# ======================== Seuils ========================

def threshold_fix_ppr(p, ppr_target: float):
    ppr_target = float(np.clip(ppr_target, 0.01, 0.99))
    thr = float(np.quantile(p, 1.0 - ppr_target))
    ppr_emp = float(np.mean(p >= thr))
    return thr, ppr_emp

def choose_threshold_by(ps, rs, ts, mode="f1", target_p=0.6):
    if mode == "f1":
        f1s = (2*ps*rs)/(ps+rs+1e-12)
        idx = int(np.nanargmax(f1s))
        thr = float(ts[idx] if idx < len(ts) else 0.5)
        return thr, {"crit":"F1","score":float(f1s[idx])}
    elif mode == "f0.5":
        beta=0.5; b2=beta*beta
        fs = ((1+b2)*ps*rs)/(b2*ps + rs + 1e-12)
        idx = int(np.nanargmax(fs))
        thr = float(ts[idx] if idx < len(ts) else 0.5)
        return thr, {"crit":"F0.5","score":float(fs[idx])}
    else:
        thr = 0.5; hit = 0.0
        for i in range(min(len(ps)-1, len(ts))):
            if ps[i] >= target_p:
                thr = float(ts[i]); hit = float(ps[i]); break
        return thr, {"crit":f"P≥{target_p:.2f}","score":hit}

# ======================== Augmentation (bug only) ========================

def augment_minority_gaussian(Xm, Xr, y, std=0.02, factor=0.5, seed=42):
    if std <= 0 or factor <= 0: 
        return Xm, Xr, y
    rng = np.random.default_rng(seed)
    bug_idx = np.where(y == 1)[0]
    if len(bug_idx) == 0:
        return Xm, Xr, y
    n_new = max(1, int(len(bug_idx) * factor))
    sel = rng.choice(bug_idx, size=n_new, replace=(n_new > len(bug_idx)))
    Xm_new = Xm[sel].copy()
    Xr_new = Xr[sel].copy()
    if Xm_new.shape[1] > 0:
        Xm_new += rng.normal(0, std, size=Xm_new.shape)
    if Xr_new.shape[1] > 0:
        Xr_new += rng.normal(0, std, size=Xr_new.shape)
    y_new = np.ones(n_new, dtype=int)
    Xm_aug = np.vstack([Xm, Xm_new]) if Xm.shape[1] else Xm
    Xr_aug = np.vstack([Xr, Xr_new]) if Xr.shape[1] else Xr
    y_aug  = np.concatenate([y, y_new])
    return Xm_aug, Xr_aug, y_aug

# ======================== Plot helpers ========================

def _render_fig(out_dir: Path|None, name: str|None):
    plt.tight_layout()
    if SHOW_FIGS or out_dir is None:
        plt.show(); plt.close()
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / name, dpi=150); plt.close()

def plot_history(hist, out_dir: Path|None):
    for key, title, fname in [
        ("loss", "Training/Validation Loss", "train_val_loss.png"),
        ("auc_pr", "Training/Validation AUC-PR", "train_val_auc_pr.png"),
        ("auc_roc","Training/Validation AUC-ROC","train_val_auc_roc.png")
    ]:
        if key in hist.history:
            plt.figure()
            plt.plot(hist.history[key], label=key)
            if "val_"+key in hist.history:
                plt.plot(hist.history["val_"+key], label="val_"+key)
            plt.title(title); plt.xlabel("epoch"); plt.ylabel(key); plt.legend()
            _render_fig(out_dir, fname)

def plot_pr(y_true, y_prob, title, out_dir: Path|None, fname="pr.png"):
    p, r, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure(); plt.plot(r, p, label=f"AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(title); plt.legend()
    _render_fig(out_dir, fname)

def plot_roc(y_true, y_prob, title, out_dir: Path|None, fname="roc.png"):
    fpr, tpr, _ = roc_curve(y_true, y_prob); auc = roc_auc_score(y_true, y_prob)
    plt.figure(); plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1],[0,1], linestyle="--", label="chance")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(title); plt.legend()
    _render_fig(out_dir, fname)

def plot_cm(cm_mat, title, out_dir: Path|None, fname="cm.png", normalize=False):
    if normalize:
        cmn = cm_mat.astype(float) / (cm_mat.sum(axis=1, keepdims=True) + 1e-12)
        mat, fmt = cmn, ".2f"
    else:
        mat, fmt = cm_mat, "d"
    plt.figure(); plt.imshow(mat, interpolation="nearest"); plt.title(title); plt.colorbar()
    tick = np.arange(2); labels = ["0","1"]
    plt.xticks(tick, labels); plt.yticks(tick, labels)
    for i, j in itertools.product(range(mat.shape[0]), range(mat.shape[1])):
        plt.text(j, i, format(mat[i, j], fmt), ha="center", va="center")
    plt.ylabel("True"); plt.xlabel("Pred")
    _render_fig(out_dir, fname)

# ======================== Évaluation ========================

def classify_at_threshold(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    rp = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": float((y_pred == y_true).mean()),
        "precision_pos": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_pos": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_pos": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "auc_pr": float(average_precision_score(y_true, y_prob)),
        "report": rp,
        "cm": cm.tolist(),
    }
    return y_pred, cm, metrics

# ======================== Utils entraînement ========================

def make_scalers(kind):
    if kind == "quantile":
        sm = QuantileTransformer(output_distribution="normal", random_state=42)
        sr = QuantileTransformer(output_distribution="normal", random_state=42)
    else:
        sm = StandardScaler(); sr = StandardScaler()
    return sm, sr

def compile_model(nr, nm, hp, use_focal):
    model = build_transformer(nr, nm,
                              d_model=hp["d_model"], n_layers=hp["n_layers"], n_heads=hp["n_heads"],
                              d_ff=hp["d_ff"], dropout=hp["dropout"], l2reg=1e-5)
    loss_fn = (focal_loss(alpha=hp["focal_alpha"], gamma=hp["focal_gamma"])
               if use_focal else tf.keras.losses.CategoricalCrossentropy(label_smoothing=hp["label_smoothing"]))
    opt = optimizers.AdamW(learning_rate=hp["lr"], weight_decay=hp["weight_decay"])
    model.compile(optimizer=opt, loss=loss_fn, metrics=[
        tf.keras.metrics.AUC(name="auc_roc", curve="ROC"),
        tf.keras.metrics.AUC(name="auc_pr",  curve="PR"),
    ])
    return model

# ======================== CV: LOPO ========================

def lopo_cv_threshold_and_calib(df_trva: pd.DataFrame, METRIC_COLS, RELATION_COLS,
                                hp, args, ppr_target_default=0.55):
    projects = np.unique(df_trva["project"].astype(str).values)
    all_T, all_thr, folds_info = [], [], []
    val_freq = max(1, int(args.val_freq))
    for pj in projects:
        mask_val = (df_trva["project"].astype(str).values == pj)
        df_tr = df_trva.loc[~mask_val].reset_index(drop=True)
        df_va = df_trva.loc[mask_val].reset_index(drop=True)

        Xm_tr, Xr_tr, y_tr, _ = split_X_y_ids(df_tr, METRIC_COLS, RELATION_COLS)
        Xm_va, Xr_va, y_va, _ = split_X_y_ids(df_va, METRIC_COLS, RELATION_COLS)

        sm, sr = make_scalers(args.scaler)
        Xm_tr = sm.fit_transform(Xm_tr) if Xm_tr.shape[1] else Xm_tr
        Xr_tr = sr.fit_transform(Xr_tr) if Xr_tr.shape[1] else Xr_tr
        Xm_va = sm.transform(Xm_va)     if Xm_tr.shape[1] else Xm_va
        Xr_va = sr.transform(Xr_va)     if Xr_tr.shape[1] else Xr_va

        Xm_tr, Xr_tr, y_tr = augment_minority_gaussian(
            Xm_tr, Xr_tr, y_tr, std=args.bug_aug_std, factor=args.bug_aug_factor, seed=42
        )

        model = compile_model(Xr_tr.shape[1], Xm_tr.shape[1], hp, args.use_focal)
        cbs = [
            callbacks.ReduceLROnPlateau(monitor="val_auc_pr", mode="max", factor=0.5, patience=5, verbose=0),
            callbacks.EarlyStopping(monitor="val_auc_pr", mode="max", patience=8, restore_best_weights=True, verbose=0),
        ]
        y_tr_oh, y_va_oh = to_onehot(y_tr), to_onehot(y_va)
        model.fit(pack_input(Xr_tr, Xm_tr), y_tr_oh,
                  validation_data=(pack_input(Xr_va, Xm_va), y_va_oh),
                  epochs=hp["epochs"], batch_size=hp["batch_size"], callbacks=cbs, verbose=0,
                  validation_freq=val_freq,class_weight={0: 1.0, 1: 2.5},)

        p_val = model.predict(pack_input(Xr_va, Xm_va), batch_size=256, verbose=0)[:,1]

        # calibration
        T_used, iso_model = None, None
        if args.calib == "temp":
            T = fit_temperature_scaling_grid(y_va, p_val)
            if T > args.calib_T_max:
                iso_model = fit_isotonic(y_va, p_val); T_used = None
            else:
                T_used = T
        elif args.calib == "isotonic":
            iso_model = fit_isotonic(y_va, p_val)

        if T_used is not None: p_val_cal = apply_temperature_scaling(p_val, T_used)
        elif iso_model is not None: p_val_cal = apply_isotonic(iso_model, p_val)
        else: p_val_cal = np.clip(p_val, 1e-6, 1-1e-6)

        # seuil
        if args.ppr_target is not None:
            thr, ppr_emp = threshold_fix_ppr(p_val_cal, ppr_target_default)
            crit = f"PPR≈{ppr_target_default:.2f}"
        else:
            ps, rs, ts = precision_recall_curve(y_va, p_val_cal)
            thr, info = choose_threshold_by(ps, rs, ts, mode=args.decision_mode, target_p=args.precision_target)
            ppr_emp, crit = float(np.mean(p_val_cal >= thr)), info["crit"]

        all_thr.append(thr)
        if T_used is not None: all_T.append(T_used)
        folds_info.append({
            "project_val": pj,
            "n_val": int(len(y_va)),
            "pi_val": float(np.mean(y_va)),
            "calib": ("temp" if T_used is not None else ("isotonic" if iso_model is not None else "none")),
            "T": (None if T_used is None else float(T_used)),
            "thr": float(thr),
            "crit": crit,
            "ppr_emp": float(ppr_emp),
            "auc_pr_val": float(average_precision_score(y_va, p_val_cal)),
            "auc_roc_val": float(roc_auc_score(y_va, p_val_cal)),
        })
    T_median = (None if len(all_T)==0 else float(np.median(all_T)))
    thr_median = float(np.median(all_thr))
    return T_median, thr_median, folds_info

# ======================== CV: Stratified K-Fold (classique) ========================

def stratk_cv_threshold_and_calib(df_trva: pd.DataFrame, METRIC_COLS, RELATION_COLS,
                                  hp, args):
    from sklearn.model_selection import StratifiedKFold
    Xm_all, Xr_all, y_all, _ = split_X_y_ids(df_trva, METRIC_COLS, RELATION_COLS)
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    all_T, all_thr, folds_info = [], [], []
    for k, (idx_tr, idx_va) in enumerate(skf.split(Xm_all, y_all), start=1):
        Xm_tr, Xr_tr, y_tr = Xm_all[idx_tr], Xr_all[idx_tr], y_all[idx_tr]
        Xm_va, Xr_va, y_va = Xm_all[idx_va], Xr_all[idx_va], y_all[idx_va]
        sm, sr = make_scalers(args.scaler)
        Xm_tr = sm.fit_transform(Xm_tr) if Xm_tr.shape[1] else Xm_tr
        Xr_tr = sr.fit_transform(Xr_tr) if Xr_tr.shape[1] else Xr_tr
        Xm_va = sm.transform(Xm_va)     if Xm_tr.shape[1] else Xm_va
        Xr_va = sr.transform(Xr_va)     if Xr_tr.shape[1] else Xr_va
        Xm_tr, Xr_tr, y_tr = augment_minority_gaussian(Xm_tr, Xr_tr, y_tr, std=args.bug_aug_std, factor=args.bug_aug_factor, seed=42+k)
        model = compile_model(Xr_tr.shape[1], Xm_tr.shape[1], hp, args.use_focal)
        cbs = [
            callbacks.ReduceLROnPlateau(monitor="val_auc_pr", mode="max", factor=0.5, patience=5, verbose=0),
            callbacks.EarlyStopping(monitor="val_auc_pr", mode="max", patience=8, restore_best_weights=True, verbose=0),
        ]
        y_tr_oh, y_va_oh = to_onehot(y_tr), to_onehot(y_va)
        model.fit(pack_input(Xr_tr, Xm_tr), y_tr_oh,
                  validation_data=(pack_input(Xr_va, Xm_va), y_va_oh),
                  epochs=hp["epochs"], batch_size=hp["batch_size"], callbacks=cbs, verbose=0)
        p_val = model.predict(pack_input(Xr_va, Xm_va), batch_size=256, verbose=0)[:,1]
        T_used, iso_model = None, None
        if args.calib == "temp":
            T = fit_temperature_scaling_grid(y_va, p_val)
            if T > args.calib_T_max: iso_model = fit_isotonic(y_va, p_val)
            else: T_used = T
        elif args.calib == "isotonic":
            iso_model = fit_isotonic(y_va, p_val)
        p_val_cal = apply_temperature_scaling(p_val, T_used) if T_used is not None else (apply_isotonic(iso_model, p_val) if iso_model is not None else np.clip(p_val, 1e-6, 1-1e-6))
        if args.ppr_target is not None:
            thr, ppr_emp = threshold_fix_ppr(p_val_cal, args.ppr_target); crit = f"PPR≈{args.ppr_target:.2f}"
        else:
            ps, rs, ts = precision_recall_curve(y_va, p_val_cal)
            thr, info = choose_threshold_by(ps, rs, ts, mode=args.decision_mode, target_p=args.precision_target)
            ppr_emp, crit = float(np.mean(p_val_cal >= thr)), info["crit"]
        all_thr.append(thr)
        if T_used is not None: all_T.append(T_used)
        folds_info.append({
            "fold": k, "n_val": int(len(y_va)), "pi_val": float(np.mean(y_va)),
            "calib": ("temp" if T_used is not None else ("isotonic" if iso_model is not None else "none")),
            "T": (None if T_used is None else float(T_used)), "thr": float(thr),
            "crit": crit, "ppr_emp": float(ppr_emp),
            "auc_pr_val": float(average_precision_score(y_va, p_val_cal)),
            "auc_roc_val": float(roc_auc_score(y_va, p_val_cal)),
        })
    T_median = (None if len(all_T)==0 else float(np.median(all_T)))
    thr_median = float(np.median(all_thr))
    return T_median, thr_median, folds_info

# ======================== CV: StratifiedGroupKFold (par projet) ========================

def sgkf_cv_threshold_and_calib(df_trva: pd.DataFrame, METRIC_COLS, RELATION_COLS, hp, args):
    """
    StratifiedGroupKFold: préserve la séparation par projet (groups=project) tout en
    gardant la stratification par label. Bon proxy LOPO, bien plus rapide si n_splits=3.
    """
    try:
        from sklearn.model_selection import StratifiedGroupKFold
    except Exception as e:
        raise SystemExit("Sklearn >= 1.1 requis pour StratifiedGroupKFold (SGKF).") from e

    Xm_all, Xr_all, y_all, ids = split_X_y_ids(df_trva, METRIC_COLS, RELATION_COLS)
    groups = ids["project"]

    sgkf = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    all_T, all_thr, folds_info = [], [], []

    for k, (idx_tr, idx_va) in enumerate(sgkf.split(Xm_all, y_all, groups=groups), start=1):
        Xm_tr, Xr_tr, y_tr = Xm_all[idx_tr], Xr_all[idx_tr], y_all[idx_tr]
        Xm_va, Xr_va, y_va = Xm_all[idx_va], Xr_all[idx_va], y_all[idx_va]

        sm, sr = make_scalers(args.scaler)
        Xm_tr = sm.fit_transform(Xm_tr) if Xm_tr.shape[1] else Xm_tr
        Xr_tr = sr.fit_transform(Xr_tr) if Xr_tr.shape[1] else Xr_tr
        Xm_va = sm.transform(Xm_va)     if Xm_tr.shape[1] else Xm_va
        Xr_va = sr.transform(Xr_va)     if Xr_tr.shape[1] else Xr_va

        Xm_tr, Xr_tr, y_tr = augment_minority_gaussian(Xm_tr, Xr_tr, y_tr, std=args.bug_aug_std, factor=args.bug_aug_factor, seed=100+k)

        model = compile_model(Xr_tr.shape[1], Xm_tr.shape[1], hp, args.use_focal)
        cbs = [
            callbacks.ReduceLROnPlateau(monitor="val_auc_pr", mode="max", factor=0.5, patience=5, verbose=0),
            callbacks.EarlyStopping(monitor="val_auc_pr", mode="max", patience=8, restore_best_weights=True, verbose=0),
        ]
        y_tr_oh, y_va_oh = to_onehot(y_tr), to_onehot(y_va)
        model.fit(pack_input(Xr_tr, Xm_tr), y_tr_oh,
                  validation_data=(pack_input(Xr_va, Xm_va), y_va_oh),
                  epochs=hp["epochs"], batch_size=hp["batch_size"], callbacks=cbs, verbose=0)

        p_val = model.predict(pack_input(Xr_va, Xm_va), batch_size=256, verbose=0)[:,1]

        T_used, iso_model = None, None
        if args.calib == "temp":
            T = fit_temperature_scaling_grid(y_va, p_val)
            if T > args.calib_T_max: iso_model = fit_isotonic(y_va, p_val)
            else: T_used = T
        elif args.calib == "isotonic":
            iso_model = fit_isotonic(y_va, p_val)

        p_val_cal = apply_temperature_scaling(p_val, T_used) if T_used is not None else (apply_isotonic(iso_model, p_val) if iso_model is not None else np.clip(p_val,1e-6,1-1e-6))

        if args.ppr_target is not None:
            thr, ppr_emp = threshold_fix_ppr(p_val_cal, args.ppr_target); crit = f"PPR≈{args.ppr_target:.2f}"
        else:
            ps, rs, ts = precision_recall_curve(y_va, p_val_cal)
            thr, info = choose_threshold_by(ps, rs, ts, mode=args.decision_mode, target_p=args.precision_target)
            ppr_emp, crit = float(np.mean(p_val_cal >= thr)), info["crit"]

        all_thr.append(thr)
        if T_used is not None: all_T.append(T_used)
        folds_info.append({
            "fold": k,
            "n_val": int(len(y_va)),
            "pi_val": float(np.mean(y_va)),
            "calib": ("temp" if T_used is not None else ("isotonic" if iso_model is not None else "none")),
            "T": (None if T_used is None else float(T_used)),
            "thr": float(thr),
            "crit": crit,
            "ppr_emp": float(ppr_emp),
            "auc_pr_val": float(average_precision_score(y_va, p_val_cal)),
            "auc_roc_val": float(roc_auc_score(y_va, p_val_cal)),
        })

    T_median = (None if len(all_T)==0 else float(np.median(all_T)))
    thr_median = float(np.median(all_thr))
    return T_median, thr_median, folds_info

# ======================== Plot TEST pack ========================

def plot_test_figs(y_te, p_te, m_te, out_dir: Path|None):
    plot_pr(y_te, p_te,  "Precision-Recall (TEST)", out_dir, "pr_test.png")
    plot_roc(y_te, p_te, "ROC (TEST)",              out_dir, "roc_test.png")
    plot_cm(np.array(m_te["cm"]), "Confusion Matrix (TEST)",             out_dir, "cm_test.png",      normalize=False)
    plot_cm(np.array(m_te["cm"]), "Confusion Matrix (TEST, normalized)", out_dir, "cm_test_norm.png", normalize=True)

# ======================== Parse/Run ========================

def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--projects_dir", required=True)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val",   nargs="+", required=True)
    ap.add_argument("--test",  nargs="+", required=True)
    ap.add_argument("--preset", choices=list(PRESETS.keys()), default="fast_cpu")
    ap.add_argument("--use_focal", action="store_true")
    ap.add_argument("--out_dir", default=None, help="Si fourni: sauve figures/rapport/test_preds.csv")
    ap.add_argument("--decision_mode", choices=["f1","f0.5","precision_target"], default="f1")
    ap.add_argument("--precision_target", type=float, default=0.60)
    ap.add_argument("--ppr_target", type=float, default=0.60)
    ap.add_argument("--scaler", choices=["standard","quantile"], default="quantile")
    ap.add_argument("--show_figs", action="store_true")
    ap.add_argument("--val_freq", type=int, default=3, help="Keras validation_freq pour LOPO")

    # Calibration & prior shift
    ap.add_argument("--calib", choices=["none","temp","isotonic"], default="temp")
    ap.add_argument("--calib_T_max", type=float, default=3.0)
    ap.add_argument("--prior_shift", action="store_true")
    ap.add_argument("--pi_test", type=float, default=None)

    # CV modes
    ap.add_argument("--cv", choices=["lopo","stratk","sgkf","none"], default="sgkf")
    ap.add_argument("--n_splits", type=int, default=3, help="K (SGKF/StratK). 3 = bon compromis CPU")

    # Augmentation bug (TRAIN only)
    ap.add_argument("--bug_aug_std", type=float, default=0.01, help="0 = désactivé; ex 0.02")
    ap.add_argument("--bug_aug_factor", type=float, default=0.5, help="0 = désactivé; ex 0.5 (+50%)")
    return ap

def run(args: argparse.Namespace):
    global SHOW_FIGS
    SHOW_FIGS = args.show_figs

    # 1) Chargement
    projects_dir = Path(args.projects_dir)
    df_tr = assemble(projects_dir, args.train)
    df_va = assemble(projects_dir, args.val)
    df_te = assemble(projects_dir, args.test)

    # Pool CV = TRAIN+VAL
    df_trva = pd.concat([df_tr, df_va], ignore_index=True)

    # 2) Colonnes communes (intersection trva vs test)
    m_trva, r_trva = infer_columns(df_trva)
    m_te,   r_te   = infer_columns(df_te)
    METRIC_COLS = sorted(set(m_trva) & set(m_te))
    RELATION_COLS = sorted(set(r_trva) & set(r_te))
    if not METRIC_COLS and not RELATION_COLS:
        raise SystemExit("[ERROR] aucune feature commune trouvée.")
    print(f"#metrics={len(METRIC_COLS)} | #relations={len(RELATION_COLS)}")

    hp = PRESETS[args.preset]

    # 3) CV pour T* & thr* (médianes)
    if args.cv == "lopo":
        T_star, thr_star, folds_info = lopo_cv_threshold_and_calib(df_trva, METRIC_COLS, RELATION_COLS, hp, args, ppr_target_default=args.ppr_target)
        print("\n[LOPO] résumé :")
        for info in folds_info:
            print(f"  - {info['project_val']}: n={info['n_val']} pi={info['pi_val']:.3f} "
                  f"calib={info['calib']} T={info['T']} thr={info['thr']:.3f} "
                  f"{info['crit']} ppr_emp={info['ppr_emp']:.3f} AUC-PR={info['auc_pr_val']:.3f}")
        print(f"[LOPO] T* (médiane) = {T_star}, thr* (médiane) = {thr_star:.3f}")

    elif args.cv == "stratk":
        T_star, thr_star, folds_info = stratk_cv_threshold_and_calib(df_trva, METRIC_COLS, RELATION_COLS, hp, args)
        print("\n[StratK] résumé :")
        for info in folds_info:
            print(f"  - fold={info['fold']}: n={info['n_val']} pi={info['pi_val']:.3f} "
                  f"calib={info['calib']} T={info['T']} thr={info['thr']:.3f} "
                  f"{info['crit']} ppr_emp={info['ppr_emp']:.3f} AUC-PR={info['auc_pr_val']:.3f}")
        print(f"[StratK] T* (médiane) = {T_star}, thr* (médiane) = {thr_star:.3f}")

    elif args.cv == "sgkf":
        T_star, thr_star, folds_info = sgkf_cv_threshold_and_calib(df_trva, METRIC_COLS, RELATION_COLS, hp, args)
        print("\n[SGKF] résumé :")
        for info in folds_info:
            print(f"  - fold={info['fold']}: n={info['n_val']} pi={info['pi_val']:.3f} "
                  f"calib={info['calib']} T={info['T']} thr={info['thr']:.3f} "
                  f"{info['crit']} ppr_emp={info['ppr_emp']:.3f} AUC-PR={info['auc_pr_val']:.3f}")
        print(f"[SGKF] T* (médiane) = {T_star}, thr* (médiane) = {thr_star:.3f}")

    else:
        T_star, thr_star, folds_info = None, None, []
        print("[CV] désactivé — fallback threshold/calibration sur TRAIN+VAL.")

    # 4) Entraînement final sur TRAIN+VAL complet
    Xm_trva, Xr_trva, y_trva, _ = split_X_y_ids(df_trva, METRIC_COLS, RELATION_COLS)
    Xm_te,   Xr_te,   y_te,   ids_te = split_X_y_ids(df_te,   METRIC_COLS, RELATION_COLS)

    sm, sr = make_scalers(args.scaler)
    Xm_trva = sm.fit_transform(Xm_trva) if Xm_trva.shape[1] else Xm_trva
    Xr_trva = sr.fit_transform(Xr_trva) if Xr_trva.shape[1] else Xr_trva
    Xm_te   = sm.transform(Xm_te)       if Xm_trva.shape[1] else Xm_te
    Xr_te   = sr.transform(Xr_te)       if Xr_trva.shape[1] else Xr_te

    Xm_trva, Xr_trva, y_trva = augment_minority_gaussian(
        Xm_trva, Xr_trva, y_trva, std=args.bug_aug_std, factor=args.bug_aug_factor, seed=2025
    )

    model = compile_model(Xr_trva.shape[1], Xm_trva.shape[1], hp, args.use_focal)

    cbs = [
        callbacks.ReduceLROnPlateau(monitor="loss", mode="min", factor=0.5, patience=5, verbose=1),
        callbacks.EarlyStopping(monitor="loss", mode="min", patience=10, restore_best_weights=True, verbose=1),
    ]
    if (not SHOW_FIGS) and args.out_dir:
        ckpt = Path(args.out_dir)/"best.keras"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        cbs.insert(0, callbacks.ModelCheckpoint(str(ckpt), monitor="loss", mode="min", save_best_only=True, verbose=1))

    y_trva_oh = to_onehot(y_trva)
    hist = model.fit(pack_input(Xr_trva, Xm_trva), y_trva_oh,
                     epochs=hp["epochs"], batch_size=hp["batch_size"], callbacks=cbs, verbose=1)

    # 5) Probas TEST (+ calibration *)
    test_prob = model.predict(pack_input(Xr_te, Xm_te), batch_size=256, verbose=1)[:,1]
    test_prob = np.clip(test_prob, 1e-6, 1-1e-6)
    if T_star is not None:
        test_prob_cal = apply_temperature_scaling(test_prob, T_star)
        print(f"[CALIB*] Temp (T* médiane) appliquée: T*={T_star:.3f}")
    else:
        test_prob_cal = test_prob

    # 6) Prior-shift (optionnel)
    if args.prior_shift:
        pi_train = float(np.mean(y_trva))
        pi_test  = float(np.mean(y_te)) if args.pi_test is None else float(args.pi_test)
        test_prob_cal = correct_prior_shift(test_prob_cal, pi_train, pi_test)
        print(f"[PRIOR] shift correction: pi_train={pi_train:.3f} -> pi_test={pi_test:.3f}")

    # 7) Seuil final
    if thr_star is not None:
        thr = float(thr_star)
        print(f"[THR*] Seuil final = médiane CV = {thr:.3f}")
    else:
        # Fallback seuil sur TRAIN+VAL
        ps, rs, ts = precision_recall_curve(y_trva, model.predict(pack_input(Xr_trva, Xm_trva), batch_size=256, verbose=0)[:,1])
        thr, crit = choose_threshold_by(ps, rs, ts, mode=args.decision_mode, target_p=args.precision_target)
        print(f"[THR] fallback (train-val): decision={crit['crit']} thr={thr:.3f}")

    # 8) Évaluation TEST
    test_pred, cm_te,  m_te  = classify_at_threshold(y_te, test_prob_cal, thr)
    print("\n=== CLASSIFICATION REPORT (TEST) ===")
    print(pd.DataFrame(m_te["report"]).T)
    print(f"\n[TEST] AUC-ROC={m_te['auc_roc']:.3f} | AUC-PR={m_te['auc_pr']:.3f}")
    print(f"[TEST] Acc={m_te['accuracy']:.3f} | P+={m_te['precision_pos']:.3f} | R+={m_te['recall_pos']:.3f} | F1+={m_te['f1_pos']:.3f}")

    # 9) Figures
    figs_out = None if (SHOW_FIGS or not args.out_dir) else Path(args.out_dir) / "figs"
    plot_history(hist, figs_out)
    plot_test_figs(y_te, test_prob_cal, m_te, figs_out)

    # 10) Sauvegardes
    if (not SHOW_FIGS) and args.out_dir:
        out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
        (out/"cv_report.json").write_text(json.dumps({
            "cv_mode": args.cv, "folds": (folds_info if len(folds_info)>0 else None),
            "T_median": (None if T_star is None else float(T_star)),
            "thr_median": (None if thr is None else float(thr))
        }, indent=2), encoding="utf-8")
        pd.DataFrame({
            "project": ids_te["project"], "version": ids_te["version"], "class": ids_te["clazz"],
            "y_true": y_te, "y_prob": test_prob_cal, "y_pred": test_pred
        }).set_index("class").to_csv(out/"test_preds.csv")
        print(f"\n[OUT] saved in {out}")

# ======================== Entrée ========================

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Démo CPU-friendly : SGKF + TempScaling + PPR + augmentation douce
        print("🚀 VSCode mode: SGKF + calibration + seuil PPR (CPU-friendly).")
        default_args = argparse.Namespace(
            projects_dir = r".\out",
            train        = ["jedit_v1","jedit_v2","camel_v1","camel_v2","xerces_v1","xerces_v2",
                            "synapse_v1","synapse_v2","velocity_v1","log4j_v1","log4j_v2",
                            "xalan_v1","xalan_v2","ant_v1","ant_v3","ant_v2",
                            "lucene_v1","lucene_v2","poi_v1","poi_v2"],
            val          = ["velocity_v2","log4j_v3","xalan_v4","camel_v4"],
            test         = ["camel_v3"],
            preset       = "balanced",
            use_focal    = True,
            out_dir      = False,
            decision_mode= "f1",
            precision_target = 0.50,
            ppr_target   = None,
            scaler       = "quantile",
            show_figs    = True,
            val_freq     = 3,
            calib        = "temp",
            calib_T_max  = 3.0,
            cv           = "sgkf",     # "sgkf" | "lopo" | "stratk" | "none"
            n_splits     = 3,
            prior_shift  = False,
            pi_test      = None,
            bug_aug_std  = 0.02,       # 0.02 si tu veux un peu plus fort
            bug_aug_factor = 0.4
        )
        run(default_args)
    else:
        parser = build_parser()
        args = parser.parse_args()
        run(args)
