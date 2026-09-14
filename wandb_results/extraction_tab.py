import pandas as pd
import numpy as np
from pathlib import Path

# ====== CHEMIN DU CSV ======
CSV_PATH = "wandb_results/results_best_per_project.csv"  # <-- adapte si besoin
OUT_DIR = Path("tables_latex")
OUT_DIR.mkdir(exist_ok=True)

# ====== LECTURE ======
df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]
df["decision_mode"] = df["decision_mode"].str.lower().str.strip()

# ====== NUMERIQUE ======
num_cols = ["AUC_PR","AUC_ROC","Precision_bug","Recall_bug","F1_bug"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ====== CALCUL F0.5 ======
beta = 0.5
df["F0_5_bug"] = (1+beta**2)*df["Precision_bug"]*df["Recall_bug"] / (
    beta**2*df["Precision_bug"] + df["Recall_bug"]
)

# ====== TABLE F1 ======
df_f1 = df[df["decision_mode"] == "f1"]
cols = ["test_project","model","AUC_PR","AUC_ROC",
        "Precision_bug","Recall_bug","F1_bug"]
t_f1 = df_f1[cols].round(3)

latex_f1 = t_f1.to_latex(
    index=False,
    caption="Performances en test avec seuil optimisé selon F1.",
    label="tab:results_f1"
)

(Path("tables_latex") / "table_results_f1.tex").write_text(latex_f1, encoding="utf-8")

# ====== TABLE F0.5 ======
df_f05 = df[df["decision_mode"] == "f0.5"]
cols = ["test_project","model","AUC_PR","AUC_ROC",
        "Precision_bug","Recall_bug","F0_5_bug"]
t_f05 = df_f05[cols].round(3)

latex_f05 = t_f05.to_latex(
    index=False,
    escape=False,
    caption="Performances en test avec seuil optimisé selon $F_{0.5}$.",
    label="tab:results_f05"
)

(Path("tables_latex") / "table_results_f05.tex").write_text(latex_f05, encoding="utf-8")

print("✅ Tables générées dans le dossier tables_latex/")
