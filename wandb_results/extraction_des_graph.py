import pandas as pd
import matplotlib.pyplot as plt

# ====== CONFIG ======
CSV_PATH = "wandb_results/results_best_per_project.csv"  # adapte si besoin
OUT_FIG = "fig_velocity_v2_cnn_vs_transformer_f1.png"

# ====== LOAD ======
df = pd.read_csv(CSV_PATH)

# On filtre: projet + mode décision
d = df[(df["test_project"] == "velocity_v2") & (df["decision_mode"].str.lower() == "f1")].copy()

# On garde CNN + Transformer
d = d[d["model"].isin(["CNN", "Transformer"])]

if d.empty or d["model"].nunique() < 2:
    raise ValueError("Je ne trouve pas les 2 modèles (CNN et Transformer) pour velocity_v2 en mode F1 dans le CSV.")

# ====== VALUES ======
d = d.set_index("model")
metrics = {
    "AUC-PR": float(d.loc["CNN", "AUC_PR"]),  # CNN
    "Recall (bug)": float(d.loc["CNN", "Recall_bug"]),
    "F1 (bug)": float(d.loc["CNN", "F1_bug"]),
}
metrics_t = {
    "AUC-PR": float(d.loc["Transformer", "AUC_PR"]),  # Transformer
    "Recall (bug)": float(d.loc["Transformer", "Recall_bug"]),
    "F1 (bug)": float(d.loc["Transformer", "F1_bug"]),
}

labels = list(metrics.keys())
cnn_vals = [metrics[k] for k in labels]
tr_vals  = [metrics_t[k] for k in labels]

# ====== PLOT ======
x = range(len(labels))
bar_w = 0.38

plt.figure(figsize=(9, 4.8))
plt.bar([i - bar_w/2 for i in x], cnn_vals, width=bar_w, label="CNN")
plt.bar([i + bar_w/2 for i in x], tr_vals,  width=bar_w, label="Transformer")

plt.ylim(0, 1.0)
plt.xticks(list(x), labels)
plt.ylabel("Score")
plt.title("Velocity v2 — Comparaison CNN vs Transformer (seuil optimisé selon F1)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300)
print(f"✅ Figure enregistrée: {OUT_FIG}")
