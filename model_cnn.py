# ============================================================
# Cross-Project Bug Prediction (CNN) 
# - Pool 80/20 (train/val) OU mode classique train/val
# - Preprocessing: scaler fit sur TRAIN uniquement
# - Imbalance:
#     * SMOTE sur TRAIN uniquement (sur Xm=metrics seulement)
#     * Gaussian noise augmentation sur TRAIN (optionnel)
#     * class_weight (optionnel)
# - Calibration: Temperature scaling (fallback Isotonic)
# - Threshold: F1 / F0.5 / precision_target / PPR
# - W&B: courbes train, PR/ROC, hist scores, threshold-curves, confusion matrix
# ============================================================

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers as L, Model, regularizers, optimizers, callbacks

from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    precision_recall_curve, roc_curve, classification_report,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_score, recall_score, f1_score
)

# SMOTE (imbalanced-learn)
try:
    from imblearn.over_sampling import SMOTE
except Exception:
    SMOTE = None

# W&B
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint


# ======================== Presets ========================

ID_COLS = ["project", "version", "class", "bug"]

# Deux configurations d'hyperparametres pretes a l'emploi. 
PRESETS = {
    "fast_cpu": dict(d_model=96, dropout=0.30,
                    lr=2e-4, weight_decay=2e-4, label_smoothing=0.06,
                    batch_size=32, epochs=35),

    "balanced": dict(d_model=128, dropout=0.20,
                    lr=2e-4, weight_decay=5e-5, label_smoothing=0.01,
                    batch_size=32, epochs=70),
}


# ======================== Repro ========================

# Fixe la graine aleatoire partout (numpy, tensorflow, hachage Python,
# module random) pour des resultats reproductibles d'un run a l'autre.
def set_all_seeds(seed: int = 2025):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import random
        random.seed(seed)
    except Exception:
        pass


# ======================== IO ========================

# Charge un CSV en testant plusieurs encodages/separateurs
def load_csv(p: Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        for sep in (",", ";", "\t"):
            try:
                return pd.read_csv(p, encoding=enc, sep=sep)
            except Exception:
                pass
    return pd.read_csv(p)

# Cherche le fichier d'un projet en essayant plusieurs conventions de
# nommage possibles -- retourne le premier qui existe, ou None sinon.
def resolve_project_file(projects_dir: Path, name: str) -> Path | None:
    candidates = [
        projects_dir / f"data_{name}.csv",
        projects_dir / f"{name}_matrice_final.csv",
        projects_dir / f"{name}.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

# Charge et fusionne plusieurs projets en un seul tableau -- affiche un
# avertissement pour chaque fichier introuvable plutot que de planter,
# et s'arrete seulement si aucun projet n'a pu etre charge.
def assemble(projects_dir: Path, names: list[str]) -> pd.DataFrame:
    dfs = []
    for n in names:
        f = resolve_project_file(projects_dir, n)
        if f is None:
            print(f"[WARN] introuvable: {n}")
            continue
        df = load_csv(f)
        dfs.append(df)
        print(f"[OK] {n}: {len(df)} lignes ({f.name})")
    if not dfs:
        raise SystemExit("ERROR: aucun projet valide fourni.")
    return pd.concat(dfs, ignore_index=True)

# Separe automatiquement les colonnes en deux groupes : les descripteurs
# relationnels (celles qui commencent par "deg_") et les metriques
# logicielles (le reste, en gardant seulement les colonnes numeriques).
def infer_columns(df: pd.DataFrame):
    rel = [c for c in df.columns if str(c).startswith("deg_")]
    met = [c for c in df.columns if c not in set(ID_COLS) | set(rel)]
    met = [c for c in met if pd.api.types.is_numeric_dtype(df[c])]
    return met, rel

# Transforme la colonne "bug" (un compteur de defauts) en etiquette
# binaire : 1 si au moins un defaut est reference, 0 sinon.
def binarize_bug(s: pd.Series) -> np.ndarray:
    s = pd.to_numeric(s, errors="coerce").fillna(0)
    return (s.astype(float) > 0).astype(int).values

# Extrait, a partir d'un tableau complet, les trois blocs necessaires a
# l'entrainement : les metriques (Xm), les relations (Xr) et l'etiquette
# cible (y) -- garde aussi les identifiants (projet/version/classe) a
# part, pour tracabilite, sans jamais les donner au modele.
def split_X_y_ids(df: pd.DataFrame, metric_cols, relation_cols):
    y = binarize_bug(df["bug"])
    Xm = df[metric_cols].astype(float).fillna(0.0).values if metric_cols else np.zeros((len(df), 0))
    Xr = df[relation_cols].astype(float).fillna(0.0).values if relation_cols else np.zeros((len(df), 0))
    ids = dict(
        project=df["project"].astype(str).values if "project" in df.columns else np.array([""] * len(df)),
        version=df["version"].astype(str).values if "version" in df.columns else np.array([""] * len(df)),
        clazz=df["class"].astype(str).values if "class" in df.columns else np.array([""] * len(df)),
    )
    return Xm, Xr, y, ids

# Divise le pool d'entrainement en train (80%) et validation (20%),
# en preservant la proportion de classes bug/sans-bug dans les deux
# sous-ensembles (split stratifie).
def pool_split(df_pool: pd.DataFrame, val_size: float = 0.20, seed: int = 42):
    y = binarize_bug(df_pool["bug"])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    idx_tr, idx_va = next(sss.split(df_pool, y))
    return df_pool.iloc[idx_tr].reset_index(drop=True), df_pool.iloc[idx_va].reset_index(drop=True)


# ======================== Utils ========================

# Transforme les etiquettes 0/1 en format one-hot (ex: 0 -> [1,0],
# 1 -> [0,1]) -- format attendu par la couche de sortie softmax(2).
def to_onehot(y):
    y = y.astype(int)
    out = np.zeros((len(y), 2), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out

# Prepare les donnees dans le format attendu par le modele : une liste
# [relations, metriques] si les deux existent, ou une seule des deux
# sinon.
def pack_input(Xr, Xm):
    if Xr.shape[1] > 0 and Xm.shape[1] > 0:
        return [Xr, Xm]
    if Xr.shape[1] > 0:
        return Xr
    return Xm


# ======================== CNN Model ========================

# Construit le modele CNN a deux branches : une branche convolutionnelle
# pour les relations (motifs locaux via Conv1D kernel=3), et une branche
# perceptron (Dense) pour les metriques -- les deux sont ensuite
# concatenees avant la classification finale.
def build_cnn(n_rel_dims: int, n_met_dims: int, d_model=128, dropout=0.2, l2reg=1e-5):
    inputs = []
    branches = []

    # Branche relations (Conv1D) : detecte des motifs locaux entre
    # descripteurs relationnels voisins (kernel_size=3), avec deux
    # couches de convolution separees par un MaxPooling.
    if n_rel_dims > 0:
        inp_rel = L.Input(shape=(n_rel_dims,), name="relations")
        x = L.Reshape((n_rel_dims, 1), name="rel_reshape")(inp_rel)
        x = L.Conv1D(d_model // 2, 3, padding="same", activation="relu",
                     kernel_regularizer=regularizers.l2(l2reg))(x)
        x = L.MaxPooling1D(2)(x)
        x = L.Conv1D(d_model, 3, padding="same", activation="relu",
                     kernel_regularizer=regularizers.l2(l2reg))(x)
        x = L.GlobalAveragePooling1D()(x)
        x = L.Dropout(dropout)(x)
        inputs.append(inp_rel)
        branches.append(x)

    # Branche metriques (perceptron simple, sans convolution -- les
    # metriques ne forment pas une sequence spatiale).
    if n_met_dims > 0:
        inp_met = L.Input(shape=(n_met_dims,), name="metrics")
        y = L.Dense(d_model // 2, activation="relu", kernel_regularizer=regularizers.l2(l2reg))(inp_met)
        y = L.Dropout(dropout)(y)
        y = L.Dense(d_model // 2, activation="relu", kernel_regularizer=regularizers.l2(l2reg))(y)
        inputs.append(inp_met)
        branches.append(y)

    if not branches:
        raise ValueError("Aucune feature en entrée.")

    # Fusion des deux branches (deja resumees en vecteurs) puis
    # classification finale.
    z = branches[0] if len(branches) == 1 else L.Concatenate(name="concat")(branches)
    z = L.Dense(d_model, activation="relu", kernel_regularizer=regularizers.l2(l2reg))(z)
    z = L.Dropout(dropout)(z)
    out = L.Dense(2, activation="softmax", name="head")(z)
    return Model(inputs=inputs, outputs=out)


# ======================== Imbalance helpers ========================

# Calcule un poids par classe pour compenser le desequilibre bug/sans-bug
# (plus de poids sur la classe minoritaire) -- retourne None si desactive.
def get_class_weight(y, use=True):
    if not use:
        return None
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}

# Genere de nouveaux exemples "bug" en dupliquant des exemples existants
# et en ajoutant un leger bruit aleatoire -- complementaire a SMOTE.
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
    y_aug = np.concatenate([y, y_new])
    return Xm_aug, Xr_aug, y_aug

# Applique SMOTE (uniquement sur les metriques, jamais sur les relations)
# pour generer des exemples "bug" synthetiques -- les relations sont
# dupliquees depuis un exemple reel pour garder la correspondance.
# Reduit automatiquement k_neighbors si trop peu d'exemples minoritaires,
# pour eviter un plantage. Identique au script Transformer.
def apply_smote_train_only(Xm, Xr, y, seed=42, k_neighbors=5):
    if SMOTE is None:
        raise SystemExit("imblearn n'est pas installé. Fais: pip install imbalanced-learn")
    if Xm.shape[1] == 0:
        return Xm, Xr, y

    # ---- AJOUT : sécurité si trop peu d'exemples minoritaires ----
    n_minority = int(np.sum(y == 1))
    k_safe = min(k_neighbors, max(1, n_minority - 1))
    if k_safe < k_neighbors:
        print(f"[SMOTE] k_neighbors réduit de {k_neighbors} à {k_safe} "
              f"(seulement {n_minority} exemples minoritaires dans ce pool)")
    # ----------------------------------------------------------------

    sm = SMOTE(random_state=seed, k_neighbors=k_safe)
    Xm_res, y_res = sm.fit_resample(Xm, y)

    n_new = len(y_res) - len(y)
    if n_new <= 0:
        return Xm_res, Xr, y_res

    rng = np.random.default_rng(seed)
    bug_idx = np.where(y == 1)[0]
    if len(bug_idx) == 0:
        return Xm_res, Xr, y_res

    sel = rng.choice(bug_idx, size=n_new, replace=True)
    Xr_new = Xr[sel].copy() if Xr.shape[1] else Xr
    Xr_res = np.vstack([Xr, Xr_new]) if Xr.shape[1] else Xr
    return Xm_res, Xr_res, y_res

# ======================== Calibration (same as Transformer) ========================

# Convertit une probabilite en logit -- operation inverse d'une sigmoide,
# necessaire pour le temperature scaling.
def _to_logits_from_probs(p):
    eps = np.finfo(np.float32).eps
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p) - np.log(1.0 - p)

# Calcule l'erreur de calibration (log-loss binaire) entre les vraies
# etiquettes et les probabilites predites.
def _binary_nll(y_true, p):
    eps = 1e-12
    return -np.mean(
        y_true * np.log(np.clip(p, eps, 1 - eps)) +
        (1 - y_true) * np.log(np.clip(1 - p, eps, 1 - eps))
    )

# Cherche la meilleure temperature de calibration par recherche en
# grille, puis affine autour du meilleur resultat trouve.
def fit_temperature_scaling_grid(y_true, p_val):
    z = _to_logits_from_probs(p_val)

    def apply_T(T):
        return 1.0 / (1.0 + np.exp(-z / T))

    Ts = np.concatenate([
        np.linspace(0.05, 0.5, 10),
        np.linspace(0.5, 2.0, 16),
        np.linspace(2.0, 10.0, 9)
    ])

    best_T, best_nll = 1.0, 1e9
    for T in Ts:
        nll = _binary_nll(y_true, apply_T(T))
        if nll < best_nll:
            best_nll, best_T = nll, T

    around = np.linspace(max(0.05, best_T * 0.5), min(10.0, best_T * 1.5), 15)
    for T in around:
        nll = _binary_nll(y_true, apply_T(T))
        if nll < best_nll:
            best_nll, best_T = nll, T

    return float(max(0.05, min(10.0, best_T)))

# Applique la temperature deja trouvee a de nouvelles probabilites,
# sans jamais changer le classement des exemples entre eux.
def apply_temperature_scaling(p, T):
    z = _to_logits_from_probs(p)
    out = 1.0 / (1.0 + np.exp(-z / T))
    return np.clip(out, 1e-6, 1 - 1e-6)

# Ajuste une regression isotone comme methode de calibration de repli,
# utilisee quand la temperature trouvee sort d'une plage raisonnable.
def fit_isotonic(y_true, p_val):
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    ir.fit(p_val, y_true)
    return ir

# Applique la regression isotone deja ajustee a de nouvelles probabilites.
def apply_isotonic(ir, p):
    return np.clip(ir.predict(p), 1e-6, 1 - 1e-6)

# Choisit et applique la meilleure methode de calibration sur la
# validation : temperature scaling si raisonnable, sinon repli isotone.
def calibrate_probs(y_va, p_val, calib="temp", calib_T_max=3.0):
    T_used, iso_model = None, None

    if calib == "temp":
        T = fit_temperature_scaling_grid(y_va, p_val)
        if T > calib_T_max:
            iso_model = fit_isotonic(y_va, p_val)
        else:
            T_used = T
    elif calib == "isotonic":
        iso_model = fit_isotonic(y_va, p_val)

    if T_used is not None:
        p_cal = apply_temperature_scaling(p_val, T_used)
        info = {"method": "temp", "T": float(T_used)}
    elif iso_model is not None:
        p_cal = apply_isotonic(iso_model, p_val)
        info = {"method": "isotonic", "T": None}
    else:
        p_cal = np.clip(p_val, 1e-6, 1 - 1e-6)
        info = {"method": "none", "T": None}

    return p_cal, info, T_used, iso_model


# ======================== Thresholding (same as Transformer) ========================

# Trouve le seuil correspondant a une proportion cible de predictions
# positives (PPR = Predicted Positive Rate).
def threshold_fix_ppr(p, ppr_target: float):
    ppr_target = float(np.clip(ppr_target, 0.01, 0.99))
    thr = float(np.quantile(p, 1.0 - ppr_target))
    ppr_emp = float(np.mean(p >= thr))
    return thr, ppr_emp

# Teste tous les seuils possibles et retourne celui qui maximise le
# critere choisi (F1, F0.5, ou une precision minimale visee) -- calcule
# sur la validation, jamais sur le test.
def choose_threshold(y_true, y_prob, mode="f1", target_p=0.6):
    p, r, t = precision_recall_curve(y_true, y_prob)
    prec = p[:-1]
    rec = r[:-1]
    thr = t

    if mode == "f1":
        f1s = (2 * prec * rec) / (prec + rec + 1e-12)
        idx = int(np.nanargmax(f1s))
        return float(thr[idx]), {"crit": "F1", "score": float(f1s[idx])}

    if mode == "f0.5":
        beta = 0.5
        b2 = beta * beta
        fs = ((1 + b2) * prec * rec) / (b2 * prec + rec + 1e-12)
        idx = int(np.nanargmax(fs))
        return float(thr[idx]), {"crit": "F0.5", "score": float(fs[idx])}

    # precision_target
    for i in range(len(thr)):
        if prec[i] >= target_p:
            return float(thr[i]), {"crit": f"P≥{target_p:.2f}", "score": float(prec[i])}
    return 0.5, {"crit": f"P≥{target_p:.2f}", "score": 0.0}


# ======================== Evaluation ========================

# Applique un seuil de decision aux probabilites pour obtenir des
# predictions binaires, puis calcule toutes les metriques d'evaluation
# et la matrice de confusion.
def classify_at_threshold(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    rp = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": float((y_pred == y_true).mean()),
        "precision_pos": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_pos": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_pos": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "auc_pr": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "report": rp,
        "cm": cm.tolist(),
    }
    return y_pred, cm, metrics


# ======================== Scalers ========================

# Cree les deux normaliseurs (metriques et relations) : QuantileTransformer
# (rend la distribution plus normale) ou StandardScaler par defaut.
def make_scalers(kind):
    if kind == "quantile":
        sm = QuantileTransformer(output_distribution="normal", random_state=42)
        sr = QuantileTransformer(output_distribution="normal", random_state=42)
    else:
        sm = StandardScaler()
        sr = StandardScaler()
    return sm, sr


# ======================== W&B plots (same as Transformer) ========================

# Genere et envoie a wandb les courbes de diagnostic : precision-rappel,
# ROC, et l'evolution des scores selon le seuil -- purement visuel.
def wandb_log_pr_roc_threshold(y_true, y_prob, prefix="test"):
    p, r, t = precision_recall_curve(y_true, y_prob)
    fig = plt.figure()
    plt.plot(r, p)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR curve ({prefix})")
    wandb.log({f"{prefix}/pr_curve": wandb.Image(fig)})
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC curve ({prefix})")
    wandb.log({f"{prefix}/roc_curve": wandb.Image(fig)})
    plt.close(fig)

    thr = t
    prec = p[:-1]
    rec = r[:-1]
    f1 = (2 * prec * rec) / (prec + rec + 1e-12)

    table = wandb.Table(
        data=[[float(a), float(b), float(c), float(d)] for a, b, c, d in zip(thr, prec, rec, f1)],
        columns=["threshold", "precision", "recall", "f1"]
    )
    wandb.log({f"{prefix}/threshold_table": table})

    fig = plt.figure()
    plt.plot(thr, prec, label="precision")
    plt.plot(thr, rec, label="recall")
    plt.plot(thr, f1, label="f1")
    plt.xlabel("threshold"); plt.ylabel("score"); plt.title(f"Scores vs threshold ({prefix})")
    plt.legend()
    wandb.log({f"{prefix}/threshold_curves": wandb.Image(fig)})
    plt.close(fig)

# Genere un histogramme des probabilites predites, separe par vraie
# classe -- permet de visualiser si le modele separe bien les deux
# distributions de scores.
def wandb_log_hist(y_true, y_prob, prefix="test"):
    fig = plt.figure()
    plt.hist(y_prob[y_true == 0], bins=40, alpha=0.7, label="clean")
    plt.hist(y_prob[y_true == 1], bins=40, alpha=0.7, label="bug")
    plt.title(f"Score distribution ({prefix})")
    plt.xlabel("p(bug)"); plt.ylabel("count")
    plt.legend()
    wandb.log({f"{prefix}/score_hist": wandb.Image(fig)})
    plt.close(fig)


# ======================== Args ========================

# Definit tous les arguments passables en ligne de commande. 
def build_parser():
    ap = argparse.ArgumentParser()

    ap.add_argument("--projects_dir", required=True)
    ap.add_argument("--pool", nargs="+", default=None, help="Fusion projets puis split 80/20 interne (train/val).")
    ap.add_argument("--pool_val_size", type=float, default=0.20)

    ap.add_argument("--train", nargs="+", default=None)
    ap.add_argument("--val", nargs="+", default=None)
    ap.add_argument("--test", nargs="+", required=True)

    ap.add_argument("--preset", choices=list(PRESETS.keys()), default="balanced")

    ap.add_argument("--scaler", choices=["standard", "quantile"], default="quantile")

    # imbalance
    ap.add_argument("--use_class_weight", action="store_true")
    ap.add_argument("--use_smote", action="store_true")
    ap.add_argument("--smote_k", type=int, default=5)

    ap.add_argument("--bug_aug_std", type=float, default=0.02)
    ap.add_argument("--bug_aug_factor", type=float, default=0.4)

    # calibration & threshold (IDENTIQUE TRANSFORMER)
    ap.add_argument("--calib", choices=["none", "temp", "isotonic"], default="temp")
    ap.add_argument("--calib_T_max", type=float, default=3.0)

    ap.add_argument("--decision_mode", choices=["f1", "f0.5", "precision_target"], default="f1")
    ap.add_argument("--precision_target", type=float, default=0.60)
    ap.add_argument("--ppr_target", type=float, default=None)

    # output
    ap.add_argument("--out_dir", default=None)

    # wandb
    ap.add_argument("--use_wandb", action="store_true")
    ap.add_argument("--wandb_project", default="bug-prediction-cnn_final")
    ap.add_argument("--wandb_group", default=None)
    ap.add_argument("--wandb_name", default=None)
    ap.add_argument("--wandb_tags", nargs="*", default=[])
    ap.add_argument("--model_name", default="bug_cnn")

    ap.add_argument("--seed", type=int, default=42)
    return ap


# ======================== Run ========================

# Fonction principale qui execute un run complet, du chargement des
# donnees jusqu'a la sauvegarde des resultats : construit le pool sans
# fuite, normalise, gere le desequilibre, entraine le modele CNN,
# calibre les probabilites, choisit le seuil, puis evalue et enregistre
# tout sur le projet cible (jamais vu par le modele).
def run(args: argparse.Namespace):
    set_all_seeds(args.seed)
    print(f"[SEED] {args.seed}")

    projects_dir = Path(args.projects_dir)

    # 1) Chargement des donnees
    if args.pool and len(args.pool) > 0:
        print(f"[POOL] Train/Val fusionnés: {args.pool}")
        df_pool = assemble(projects_dir, args.pool)
        df_tr, df_va = pool_split(df_pool, val_size=args.pool_val_size, seed=args.seed)
        print(f"[POOL] split: train={len(df_tr)} | val={len(df_va)}")
    else:
        if not args.train or not args.val:
            raise SystemExit("[ERROR] fournir --pool OU (--train ET --val).")
        df_tr = assemble(projects_dir, args.train)
        df_va = assemble(projects_dir, args.val)

    df_te = assemble(projects_dir, args.test)

    # 2) Colonnes communes (intersection train/val et test)
    df_trva = pd.concat([df_tr, df_va], ignore_index=True)
    m_trva, r_trva = infer_columns(df_trva)
    m_te, r_te = infer_columns(df_te)

    METRIC_COLS = sorted(set(m_trva) & set(m_te))
    RELATION_COLS = sorted(set(r_trva) & set(r_te))

    if not METRIC_COLS and not RELATION_COLS:
        raise SystemExit("[ERROR] aucune feature commune trouvée.")

    print(f"#metrics={len(METRIC_COLS)} | #relations={len(RELATION_COLS)}")

    # 3) Separation en tableaux Xm/Xr/y
    Xm_tr, Xr_tr, y_tr, _ = split_X_y_ids(df_tr, METRIC_COLS, RELATION_COLS)
    Xm_va, Xr_va, y_va, _ = split_X_y_ids(df_va, METRIC_COLS, RELATION_COLS)
    Xm_te, Xr_te, y_te, ids_te = split_X_y_ids(df_te, METRIC_COLS, RELATION_COLS)

    # 4) Normalisation (ajustee sur TRAIN uniquement)
    sm, sr = make_scalers(args.scaler)
    Xm_tr = sm.fit_transform(Xm_tr) if Xm_tr.shape[1] else Xm_tr
    Xr_tr = sr.fit_transform(Xr_tr) if Xr_tr.shape[1] else Xr_tr

    Xm_va = sm.transform(Xm_va) if Xm_tr.shape[1] else Xm_va
    Xr_va = sr.transform(Xr_va) if Xr_tr.shape[1] else Xr_va

    Xm_te = sm.transform(Xm_te) if Xm_tr.shape[1] else Xm_te
    Xr_te = sr.transform(Xr_te) if Xr_tr.shape[1] else Xr_te

    # 5) Desequilibre : SMOTE (TRAIN uniquement)
    if args.use_smote:
        print("[SMOTE] actif (TRAIN only)")
        Xm_tr, Xr_tr, y_tr = apply_smote_train_only(
            Xm_tr, Xr_tr, y_tr, seed=args.seed, k_neighbors=args.smote_k
        )

    # 6) Desequilibre : augmentation gaussienne (TRAIN uniquement)
    Xm_tr, Xr_tr, y_tr = augment_minority_gaussian(
        Xm_tr, Xr_tr, y_tr,
        std=args.bug_aug_std, factor=args.bug_aug_factor, seed=args.seed
    )

    # 7) Initialisation W&B
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_name or f"{args.model_name}_test_{args.test[0]}",
            tags=args.wandb_tags + [args.model_name],
            config={
                **vars(args),
                "n_metrics": len(METRIC_COLS),
                "n_relations": len(RELATION_COLS),
                "train_len": int(len(df_tr)),
                "val_len": int(len(df_va)),
                "test_len": int(len(df_te)),
            }
        )

    # 8) Construction + compilation du modele CNN (selon le preset)
    hp = PRESETS[args.preset]
    model = build_cnn(
        n_rel_dims=Xr_tr.shape[1],
        n_met_dims=Xm_tr.shape[1],
        d_model=hp["d_model"],
        dropout=hp["dropout"],
        l2reg=1e-5,
    )

    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=hp["label_smoothing"])
    opt = optimizers.AdamW(learning_rate=hp["lr"], weight_decay=hp["weight_decay"])

    model.compile(
        optimizer=opt,
        loss=loss_fn,
        metrics=[
            tf.keras.metrics.AUC(name="auc_roc", curve="ROC"),
            tf.keras.metrics.AUC(name="auc_pr", curve="PR"),
        ],
    )

    cbs = [
        callbacks.ReduceLROnPlateau(monitor="val_auc_pr", mode="max", factor=0.5, patience=5, verbose=1),
        callbacks.EarlyStopping(monitor="val_auc_pr", mode="max", patience=10, restore_best_weights=True, verbose=1),
    ]

    if args.use_wandb:
        cbs.append(WandbMetricsLogger(log_freq="epoch"))
        if args.out_dir:
            out = Path(args.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            cbs.append(WandbModelCheckpoint(
                filepath=str(out / "best.keras"),
                monitor="val_auc_pr",
                mode="max",
                save_best_only=True,
                verbose=1
            ))

    cw = get_class_weight(y_tr, use=args.use_class_weight)
    if cw is not None:
        print(f"[CLASS_WEIGHT] {cw}")

    model.fit(
        pack_input(Xr_tr, Xm_tr), to_onehot(y_tr),
        validation_data=(pack_input(Xr_va, Xm_va), to_onehot(y_va)),
        epochs=hp["epochs"],
        batch_size=hp["batch_size"],
        callbacks=cbs,
        class_weight=cw,
        verbose=1
    )

    # 9) Calibration sur validation + choix du seuil
    p_val = model.predict(pack_input(Xr_va, Xm_va), batch_size=256, verbose=0)[:, 1]
    p_val = np.clip(p_val, 1e-6, 1 - 1e-6)

    p_val_cal, calib_info, T_used, iso_model = calibrate_probs(
        y_va, p_val, calib=args.calib, calib_T_max=args.calib_T_max
    )

    if args.ppr_target is not None:
        thr, ppr_emp = threshold_fix_ppr(p_val_cal, args.ppr_target)
        thr_info = {"crit": f"PPR≈{args.ppr_target:.2f}", "score": ppr_emp}
    else:
        thr, thr_info = choose_threshold(y_va, p_val_cal, mode=args.decision_mode, target_p=args.precision_target)

    print(f"[VAL] calib={calib_info} | thr={thr:.3f} ({thr_info['crit']})")

    # 10) Probabilites sur test + application de la calibration
    p_te = model.predict(pack_input(Xr_te, Xm_te), batch_size=256, verbose=0)[:, 1]
    p_te = np.clip(p_te, 1e-6, 1 - 1e-6)

    if calib_info["method"] == "temp" and T_used is not None:
        p_te_cal = apply_temperature_scaling(p_te, T_used)
    elif calib_info["method"] == "isotonic" and iso_model is not None:
        p_te_cal = apply_isotonic(iso_model, p_te)
    else:
        p_te_cal = p_te

    # 11) Evaluation finale
    y_pred, cm, m = classify_at_threshold(y_te, p_te_cal, thr)
    print("\n=== CLASSIFICATION REPORT (TEST) ===")
    print(pd.DataFrame(m["report"]).T)
    print(f"\n[TEST] AUC-ROC={m['auc_roc']:.3f} | AUC-PR={m['auc_pr']:.3f}")
    print(f"[TEST] Acc={m['accuracy']:.3f} | P+={m['precision_pos']:.3f} | R+={m['recall_pos']:.3f} | F1+={m['f1_pos']:.3f}")

    # 12) Envoi des resultats a W&B
    if args.use_wandb:
        wandb.log({
            "val/calib_method": calib_info["method"],
            "val/calib_T": calib_info["T"] if calib_info["T"] is not None else -1,
            "val/threshold": thr,
            "val/threshold_crit": thr_info["crit"],

            "test/accuracy": m["accuracy"],
            "test/precision_pos": m["precision_pos"],
            "test/recall_pos": m["recall_pos"],
            "test/f1_pos": m["f1_pos"],
            "test/auc_pr": m["auc_pr"],
            "test/auc_roc": m["auc_roc"],

            "test/confusion_matrix": wandb.plot.confusion_matrix(
                y_true=y_te,
                preds=y_pred,
                class_names=["clean", "bug"]
            )
        })
        wandb_log_pr_roc_threshold(y_te, p_te_cal, prefix="test")
        wandb_log_hist(y_te, p_te_cal, prefix="test")

    # 13) Sauvegarde des resultats (optionnel)
    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)

        # sauvegarde des predictions
        pd.DataFrame({
            "project": ids_te["project"],
            "version": ids_te["version"],
            "class": ids_te["clazz"],
            "y_true": y_te,
            "y_prob": p_te_cal,
            "y_pred": y_pred,
        }).to_csv(out / "test_preds.csv", index=False)

        # sauvegarde du rapport
        (out / "report.json").write_text(json.dumps({
            "val": {"calib": calib_info, "thr": thr, "thr_info": thr_info},
            "test": {k: v for k, v in m.items() if k != "report"},
            "test_report": m["report"]
        }, indent=2), encoding="utf-8")

        print(f"[OUT] saved: {out}")

    if args.use_wandb:
        wandb.finish()


# ======================== Main ========================

if __name__ == "__main__":
    parser = build_parser()
    if len(sys.argv) == 1:
        # Configuration par defaut utilisee si le script est lance sans
        # aucun argument -- ignoree des qu'on passe des arguments (comme
        # le fait run_all.py). Note : "velocity_v2" (le test) n'apparait
        # pas dans le pool -- pas de fuite dans ce bloc precis.
        args = argparse.Namespace(
            projects_dir=r".\données_final",
            pool=[
                "jedit_v1", "jedit_v2", "jedit_v3",
                "camel_v1", "camel_v2",
                "xerces_v1", "xerces_v2",
                "synapse_v1", "synapse_v3",
                "ant_v1", "ant_v2", "ant_v3",
                "log4j_v1", "log4j_v3",
                "xalan_v1", "xalan_v2",
                "lucene_v2", "lucene_v1", "poi_v2", "poi_v3"
            ],
            pool_val_size=0.20,
            train=None, val=None,
            test=["velocity_v2"],

            preset="balanced",

            scaler="quantile",
            use_class_weight=True,
            use_smote=True,
            smote_k=5,
            bug_aug_std=0.02,
            bug_aug_factor=0.4,
            calib="temp",
            calib_T_max=3.0,
            decision_mode="f1",
            precision_target=0.70,
            ppr_target=None,
            out_dir="runs/CNN",
            use_wandb=True,
            wandb_project="bug-prediction-cnn_final",
            wandb_group="Memoire_2026",
            wandb_name="test_velocity_v2_F1",
            wandb_tags=["smote", "temp_scaling", "threshold_f1"],
            model_name="bug_cnn",
            seed=2025,
        )
        run(args)
    else:
        # Cas normal : les arguments passes en ligne de commande (via
        # generate_pools.py ou run_all.py) remplacent la configuration
        # par defaut ci-dessus.
        run(parser.parse_args())