import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
CSV_PATH = "wandb_results/results_best_per_project.csv"   # <-- mets ici ton fichier
DECISION_MODE = "F0.5"                        # on fait les 3 figures sur F1
OUT_DIR = "."                               # dossier de sortie (ex: "figures")

# =========================
# LOAD
# =========================
df = pd.read_csv(CSV_PATH)

# Normalise quelques noms possibles
df["model"] = df["model"].astype(str).str.strip()
df["test_project"] = df["test_project"].astype(str).str.strip()
df["decision_mode"] = df["decision_mode"].astype(str).str.strip()

# =========================
# 1) Filtrer sur F1
# =========================
df_f1 = df[df["decision_mode"].str.upper() == DECISION_MODE].copy()

required_cols = {"test_project","model","Precision_bug","Recall_bug","F1_bug"}
missing = required_cols - set(df_f1.columns)
if missing:
    raise ValueError(f"Colonnes manquantes dans le CSV: {missing}")

# =========================
# 2) Garder projets communs aux deux modèles
# =========================
# (on suppose que les modèles s'appellent exactement "CNN" et "Transformer")
wanted_models = ["CNN", "Transformer"]
df_f1 = df_f1[df_f1["model"].isin(wanted_models)].copy()

counts = df_f1.groupby("test_project")["model"].nunique()
common_projects = counts[counts == 2].index.tolist()

df_common = df_f1[df_f1["test_project"].isin(common_projects)].copy()

if len(common_projects) == 0:
    raise ValueError("Aucun projet commun trouvé entre CNN et Transformer (en mode F0.5).")

# Pour avoir un ordre stable (facultatif)
common_projects = sorted(common_projects)

# =========================
# FIGURE 1 — F1 moyen (bar plot + std)
# =========================
stats = df_common.groupby("model")["F1_bug"].agg(["mean", "std", "count"]).reindex(wanted_models)

plt.figure()
plt.bar(stats.index, stats["mean"])
plt.errorbar(
    x=np.arange(len(stats.index)),
    y=stats["mean"].values,
    yerr=stats["std"].fillna(0).values,
    fmt="none",
    capsize=5
)
plt.ylabel("F1 (classe bug) — moyenne sur projets communs")
plt.title("Comparaison globale (F1 moyen) : CNN vs Transformer")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig1_f1_moyen_cnn_vs_transformer.png", dpi=300)
plt.close()

# =========================
# FIGURE 2 — F1 par projet (grouped bars)
# =========================
pivot_f1 = df_common.pivot_table(
    index="test_project",
    columns="model",
    values="F1_bug",
    aggfunc="first"
).reindex(common_projects)

x = np.arange(len(pivot_f1.index))
width = 0.38

plt.figure(figsize=(max(10, len(x)*0.6), 5))
plt.bar(x - width/2, pivot_f1["CNN"], width, label="CNN")
plt.bar(x + width/2, pivot_f1["Transformer"], width, label="Transformer")

plt.xticks(x, pivot_f1.index, rotation=45, ha="right")
plt.ylabel("F1 (classe bug)")
plt.title("Comparaison par projet (F1) — projets communs")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig2_f1_par_projet_cnn_vs_transformer.png", dpi=300)
plt.close()

# =========================
# FIGURE 3 — Precision vs Recall (scatter)
# =========================
plt.figure(figsize=(7, 5))
for m in wanted_models:
    sub = df_common[df_common["model"] == m]
    plt.scatter(sub["Recall_bug"], sub["Precision_bug"], label=m)

# Optionnel : annoter les projets (utile si pas trop de points)
for _, row in df_common.iterrows():
    plt.annotate(row["test_project"], (row["Recall_bug"], row["Precision_bug"]),
                 fontsize=8, alpha=0.7)

plt.xlabel("Rappel (classe bug)")
plt.ylabel("Précision (classe bug)")
plt.title("Compromis précision–rappel (F1) — CNN vs Transformer")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig3_scatter_precision_recall_cnn_vs_transformer.png", dpi=300)
plt.close()

print("✅ Figures générées :")
print(" - fig1_f1_moyen_cnn_vs_transformer.png")
print(" - fig2_f1_par_projet_cnn_vs_transformer.png")
print(" - fig3_scatter_precision_recall_cnn_vs_transformer.png")
print(f"Projets communs utilisés ({len(common_projects)}):", common_projects)
