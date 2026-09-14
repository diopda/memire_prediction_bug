import wandb
import pandas as pd
from pathlib import Path

# =========================================================
# CONFIGURATION
# =========================================================

ENTITY = "dadah95diop-universit-du-qu-bec-trois-rivi-res"

PROJECTS = {
    "CNN": "bug-prediction-cnn_final",
    "Transformer": "bug-prediction-transformer_final",
}

OUT_DIR = Path("wandb_results_by_model")
OUT_DIR.mkdir(exist_ok=True)

ONLY_FINISHED = True

# Normalisation des noms de decision_mode
DECISION_MODE_MAP = {
    "f1": "F1",
    "f0.5": "F0.5",
    "f05": "F0.5",
    "f_0.5": "F0.5",
    "f_05": "F0.5",
    "f0_5": "F0.5",
}

# critère de sélection de la meilleure run par projet
# ici on prend la meilleure F1 mesurée sur le test
BEST_BY = "F1_bug"

# =========================================================
# OUTILS
# =========================================================

def safe_get(d, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def normalize_decision_mode(value):
    if value is None:
        return None
    v = str(value).strip().lower()
    return DECISION_MODE_MAP.get(v, value)


def extract_test_project(cfg, summary, run_name):
    # cas 1
    test_cfg = cfg.get("test")
    if isinstance(test_cfg, list) and len(test_cfg) > 0:
        return test_cfg[0]

    # cas 2
    if isinstance(cfg.get("test_project"), str):
        return cfg["test_project"]

    # cas 3
    if isinstance(summary.get("test_project"), str):
        return summary["test_project"]

    # cas 4 : chercher dans le nom du run
    if isinstance(run_name, str):
        candidates = [
            "ant_v1", "ant_v2", "ant_v3", "ant_v4",
            "camel_v1", "camel_v2", "camel_v3", "camel_v4",
            "jedit_v1", "jedit_v2", "jedit_v3", "jedit_v4",
            "log4j_v1", "log4j_v2", "log4j_v3",
            "lucene_v1", "lucene_v2", "lucene_v3",
            "poi_v1", "poi_v2", "poi_v3", "poi_v4",
            "synapse_v1", "synapse_v2", "synapse_v3",
            "velocity_v1", "velocity_v2",
            "xalan_v1", "xalan_v2", "xalan_v3", "xalan_v4",
            "xerces_v1", "xerces_v2", "xerces_v3", "xerces_v4",
        ]
        low = run_name.lower()
        for c in candidates:
            if c in low:
                return c

    return None


def extract_run_row(run, model_name):
    cfg = run.config
    summary = run.summary._json_dict if hasattr(run.summary, "_json_dict") else dict(run.summary)

    decision_mode = normalize_decision_mode(
        safe_get(cfg, "decision_mode", "threshold_mode", default=safe_get(summary, "decision_mode"))
    )

    row = {
        "model": model_name,
        "project_wandb": run.project,
        "run_id": run.id,
        "run_name": run.name,
        "state": run.state,
        "test_project": extract_test_project(cfg, summary, run.name),
        "decision_mode": decision_mode,
        "seed": cfg.get("seed"),
        "use_smote": cfg.get("use_smote"),
        "smote_k": cfg.get("smote_k"),
        "calibration": safe_get(cfg, "calib", "calibration"),

        # métriques test
        "AUC_PR": safe_get(summary, "test/auc_pr", "test_auc_pr"),
        "AUC_ROC": safe_get(summary, "test/auc_roc", "test_auc_roc"),
        "Precision_bug": safe_get(summary, "test/precision_pos", "test_precision_pos"),
        "Recall_bug": safe_get(summary, "test/recall_pos", "test_recall_pos"),
        "F1_bug": safe_get(summary, "test/f1_pos", "test_f1_pos"),
        "Accuracy": safe_get(summary, "test/accuracy", "test_acc"),
        "Threshold": safe_get(summary, "val/threshold", "best_threshold", "threshold"),

        "url": run.url,
        "created_at": str(run.created_at),
    }

    return row


def export_latex_table(df, out_path, caption, label):
    latex = df.to_latex(
        index=False,
        float_format="%.3f",
        caption=caption,
        label=label,
        escape=False
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex)


# =========================================================
# EXTRACTION GLOBALE
# =========================================================

api = wandb.Api()
all_rows = []

for model_name, project_name in PROJECTS.items():
    print(f"\n📥 Lecture du projet W&B : {project_name}")
    runs = api.runs(f"{ENTITY}/{project_name}")

    for run in runs:
        if ONLY_FINISHED and run.state != "finished":
            continue

        row = extract_run_row(run, model_name)
        all_rows.append(row)

df_all = pd.DataFrame(all_rows)

# garder seulement les runs exploitables
df_all = df_all.dropna(subset=["test_project", "decision_mode", "F1_bug"]).copy()

# arrondir pour exports lisibles
metric_cols = ["AUC_PR", "AUC_ROC", "Precision_bug", "Recall_bug", "F1_bug", "Accuracy", "Threshold"]
for c in metric_cols:
    if c in df_all.columns:
        df_all[c] = pd.to_numeric(df_all[c], errors="coerce")

df_all = df_all.sort_values(["model", "decision_mode", "test_project", BEST_BY], ascending=[True, True, True, False])
df_all.to_csv(OUT_DIR / "all_runs_raw.csv", index=False)
print("✅ all_runs_raw.csv généré")

# =========================================================
# EXPORT PAR MODELE
# =========================================================

for model_name in PROJECTS.keys():
    df_model = df_all[df_all["model"] == model_name].copy()
    model_dir = OUT_DIR / model_name.lower()
    model_dir.mkdir(exist_ok=True)

    # 1) tout le modèle
    df_model.to_csv(model_dir / f"{model_name.lower()}_all_runs.csv", index=False)
    print(f"✅ {model_name.lower()}_all_runs.csv généré")

    # 2) meilleure run par projet et decision_mode
    df_best_model = (
        df_model.sort_values(BEST_BY, ascending=False)
                .groupby(["test_project", "decision_mode"], as_index=False)
                .first()
                .sort_values(["decision_mode", "test_project"])
                .reset_index(drop=True)
    )

    df_best_model.to_csv(model_dir / f"{model_name.lower()}_best_per_project_and_mode.csv", index=False)
    print(f"✅ {model_name.lower()}_best_per_project_and_mode.csv généré")

    # 3) séparer F1 et F0.5
    for mode in ["F1", "F0.5"]:
        df_mode = df_best_model[df_best_model["decision_mode"] == mode].copy()

        table = df_mode[[
            "test_project", "AUC_PR", "AUC_ROC",
            "Precision_bug", "Recall_bug", "F1_bug",
            "Accuracy", "Threshold"
        ]].rename(columns={
            "test_project": "Projet_test",
            "AUC_PR": "AUC_PR",
            "AUC_ROC": "AUC_ROC",
            "Precision_bug": "Precision",
            "Recall_bug": "Rappel",
            "F1_bug": "F1",
            "Accuracy": "Accuracy",
            "Threshold": "Seuil"
        })

        table = table.round(3)

        csv_name = f"{model_name.lower()}_{mode.lower().replace('.', '').replace('0', '0')}_table.csv"
        tex_name = f"{model_name.lower()}_{mode.lower().replace('.', '').replace('0', '0')}_table.tex"

        # noms plus lisibles
        if mode == "F1":
            csv_name = f"{model_name.lower()}_f1_table.csv"
            tex_name = f"{model_name.lower()}_f1_table.tex"
        elif mode == "F0.5":
            csv_name = f"{model_name.lower()}_f05_table.csv"
            tex_name = f"{model_name.lower()}_f05_table.tex"

        table.to_csv(model_dir / csv_name, index=False)

        export_latex_table(
            table,
            model_dir / tex_name,
            caption=f"Performances en test du modèle {model_name} avec seuil optimisé selon {mode}",
            label=f"tab:{model_name.lower()}_{'f1' if mode == 'F1' else 'f05'}"
        )

        print(f"✅ {csv_name} généré")
        print(f"✅ {tex_name} généré")

print("\n🎓 Extraction terminée.")