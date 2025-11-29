# -*- coding: utf-8 -*-
"""
Build final training data from folders laid out as:
root/
  <projectA>/
    <version1>/
      matrix_multilabel.csv   # columns: src, dst, INHERITS, CONTAINS, CALLS, REFERENCES, IMPLEMENTS, ASSOCIATES, DEPENDS_ON, COMPOSES, ...
      matrics_bug.csv         # columns: class|name + metrics OO + bug (count or 0/1)
    <version2>/
      ...
  <projectB>/
    <versionX>/
      ...

Outputs in --out:
  <project>_matrice_final.csv      # one per project, all versions concatenated (rows = classes-per-version)
  global_tabular.csv               # all projects/versions concatenated

Notes:
- `bug` is binarized (>0 -> 1).
- `name` / `classname` normalized to `class` for joins.
- deg_in_* / deg_out_* computed from matrix_multilabel.csv per relation type.
- Adds aggregates: total_deg_out, total_deg_in, ratio_out_in, nb_types_rel_out, nb_types_rel_in.
- Keeps `project` and `version` columns (useful for splits/reporting; your model will ignore them as features).
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import pandas as pd

REL_DEFAULT_ORDER = [
    "INHERITS","CONTAINS","CALLS","REFERENCES","IMPLEMENTS","ASSOCIATES","DEPENDS_ON","COMPOSES"
]

# --- IO helpers ---

def safe_read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8","utf-8-sig","latin1"):
        for sep in (",",";","\t"):
            try:
                return pd.read_csv(path, encoding=enc, sep=sep)
            except Exception:
                pass
    return pd.read_csv(path)

def normalize_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ren = {}
    for c in df.columns:
        cl = str(c).lower().strip()
        if cl in {"name","classname","classes"}:
            ren[c] = "class"
    if ren:
        df = df.rename(columns=ren)
    return df

def detect_files(folder: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Find 'matrix_multilabel.csv' (relations) and 'matrics_bug.csv' (metrics+bug) in the given folder.
    Accepts a few flexible variants.
    """
    labels, metrics_bug = None, None
    for p in folder.iterdir():
        if not p.is_file():
            continue
        n = p.name.lower()

        # --- relations (labels) ---
        if labels is None and (
            n == "matrix_multilabel.csv"
            or ("matrix" in n and "multilabel" in n and n.endswith(".csv"))
            or ("matrice" in n and "label" in n and n.endswith(".csv"))  # tolérance
            or (n == "matrice_labels.csv")  # tolérance
        ):
            labels = p

        # --- metrics + bug ---
        if metrics_bug is None and (
            n == "matrics_bug.csv"                      # ton nom spécifique
            or ("matrics" in n and "bug" in n and n.endswith(".csv"))
            or (n == "metrics_bug.csv")                # tolérance
            or (n == "metrics-bug.csv")                # tolérance
            or (n == "metrics.csv")                    # fallback (si bug column existe)
        ):
            metrics_bug = p

    # Fallbacks directs
    if labels is None and (folder / "matrix_multilabel.csv").exists():
        labels = folder / "matrix_multilabel.csv"
    if metrics_bug is None and (folder / "matrics_bug.csv").exists():
        metrics_bug = folder / "matrics_bug.csv"

    return labels, metrics_bug

# --- Feature engineering (relations -> deg_*) ---

def compute_deg_from_labels(labels_df: pd.DataFrame) -> pd.DataFrame:
    L = labels_df.copy()
    L.columns = [str(c).strip() for c in L.columns]
    if not {"src","dst"}.issubset(set(L.columns)):
        raise ValueError("matrix_multilabel.csv doit contenir les colonnes 'src' et 'dst'.")

    L["src"] = L["src"].astype(str)
    L["dst"] = L["dst"].astype(str)

    rel_cols = [c for c in L.columns if c not in ("src","dst")]
    ordered  = [c for c in REL_DEFAULT_ORDER if c in rel_cols]
    extras   = [c for c in rel_cols if c not in REL_DEFAULT_ORDER]
    rel_cols = ordered + extras
    if not rel_cols:
        raise ValueError("Aucune colonne de relation détectée dans matrix_multilabel.csv")

    classes = sorted(set(L["src"]) | set(L["dst"]))
    deg = pd.DataFrame({"class": classes}).set_index("class")

    for rel in rel_cols:
        vals = L[rel].fillna(0)
        try:
            mask = vals.astype(float) != 0
        except Exception:
            mask = vals.astype(str).str.lower().str.strip().isin(("1","true","t","yes","y","bug","vrai","oui"))
        sub = L[mask]

        out_counts = sub.groupby("src").size().rename(f"deg_out_{rel.lower()}")
        in_counts  = sub.groupby("dst").size().rename(f"deg_in_{rel.lower()}")

        d = pd.concat([out_counts, in_counts], axis=1)
        deg = deg.join(d, how="left")

    deg = deg.fillna(0).astype(int).reset_index()

    out_cols = [c for c in deg.columns if c.startswith("deg_out_")]
    in_cols  = [c for c in deg.columns if c.startswith("deg_in_")]
    deg["total_deg_out"]     = deg[out_cols].sum(axis=1) if out_cols else 0
    deg["total_deg_in"]      = deg[in_cols].sum(axis=1) if in_cols else 0
    deg["ratio_out_in"]      = (deg["total_deg_out"] + 1) / (deg["total_deg_in"] + 1)
    deg["nb_types_rel_out"]  = (deg[out_cols] > 0).sum(axis=1) if out_cols else 0
    deg["nb_types_rel_in"]   = (deg[in_cols] > 0).sum(axis=1) if in_cols else 0
    return deg

# --- Label normalization ---

def binarize_bug(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return (s.astype(float) > 0).astype(int)
    low = s.astype(str).str.lower().str.strip()
    return low.isin({"1","true","t","yes","y","oui","vrai","bug","buggy","defect"}).astype(int)

def pick_bug_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        cl = str(c).lower()
        if cl in {"bug","bugs","defect","nb_bug","n_bug","nb_bugs","num_bugs","count_bug","count_bugs"}:
            return c
    return None

# --- Build one version ---

def build_one_version(project: str, version: str, folder: Path) -> Optional[pd.DataFrame]:
    labels_path, metrics_bug_path = detect_files(folder)
    if labels_path is None or metrics_bug_path is None:
        print(f"[WARN] Fichiers manquants sous {folder} (labels={labels_path}, metrics_bug={metrics_bug_path})")
        return None

    labels      = safe_read_csv(labels_path)
    metrics_bug = safe_read_csv(metrics_bug_path)

    # 1) Relations -> deg_*
    deg = compute_deg_from_labels(labels)

    # 2) Metrics + bug (binarize)
    metrics_bug = normalize_ids(metrics_bug)
    if "class" not in metrics_bug.columns:
        raise ValueError(f"'{metrics_bug_path.name}' ne contient pas la colonne 'class'/'name'.")
    bug_col = pick_bug_column(metrics_bug)
    if bug_col is None:
        raise ValueError(f"Aucune colonne bug détectée dans '{metrics_bug_path.name}'")
    metrics_bug = metrics_bug.rename(columns={bug_col: "bug"})
    metrics_bug["bug"] = binarize_bug(metrics_bug["bug"])
    metrics_bug["class"] = metrics_bug["class"].astype(str)

    # 3) Merge
    merged = deg.merge(metrics_bug, on="class", how="inner")

    # 4) Project/version (pour split/rapports)
    merged["project"] = project
    merged["version"] = version

    # 5) Colonnes en ordre lisible
    rel_out = [c for c in merged.columns if c.startswith("deg_out_")]
    rel_in  = [c for c in merged.columns if c.startswith("deg_in_")]
    aggs    = ["total_deg_out","total_deg_in","ratio_out_in","nb_types_rel_out","nb_types_rel_in"]
    exclude = set(["class","bug","project","version"]) | set(rel_out) | set(rel_in) | {"src","dst"}
    metric_candidates = [c for c in merged.columns if c not in exclude]

    front = ["project","version","class"] + metric_candidates + rel_out + rel_in + aggs + ["bug"]
    front = [c for c in front if c in merged.columns]
    merged = merged[front].copy()

    merged = merged.drop_duplicates(subset=["project","version","class"], keep="first").reset_index(drop=True)
    return merged

# --- Main (batch over <project>/<version>) ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Racine: <project>/<version>/ (avec matrix_multilabel.csv + matrics_bug.csv)")
    ap.add_argument("--out",  required=True, help="Dossier de sortie")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # Ignore folders that are obviously not data roots
    SKIP_DIRS = {".venv","runs","wandb","src","__pycache__","data","output","outputs"}

    all_rows: List[pd.DataFrame] = []
    per_project: Dict[str, List[pd.DataFrame]] = {}

    for pdir in sorted([p for p in root.iterdir() if p.is_dir() and p.name not in SKIP_DIRS]):
        project = pdir.name
        # versions under project
        version_dirs = [v for v in pdir.iterdir() if v.is_dir() and v.name not in SKIP_DIRS]

        # Cas sans sous-dossiers version → on traite le dossier projet comme une version unique "v0"
        if not version_dirs:
            print(f"[INFO] Traitement {project} (sans sous-dossier version) ...")
            df = build_one_version(project, "v0", pdir)
            if df is None or df.empty:
                print(f"[WARN] {project}: aucun enregistrement.")
            else:
                all_rows.append(df)
                per_project.setdefault(project, []).append(df)
            continue

        # Cas normal avec versions
        for vdir in sorted(version_dirs):
            version = vdir.name
            print(f"[INFO] Traitement {project}/{version} ...")
            try:
                df = build_one_version(project, version, vdir)
            except Exception as e:
                print(f"[WARN] {project}/{version} ignoré: {e}")
                df = None
            if df is None or df.empty:
                print(f"[WARN] {project}/{version}: aucun enregistrement.")
                continue
            all_rows.append(df)
            per_project.setdefault(project, []).append(df)

    if not all_rows:
        raise SystemExit("[ERROR] Aucun jeu construit. Vérifie l'arborescence et les noms de fichiers.")

    global_df = pd.concat(all_rows, ignore_index=True)

    # 1) Un fichier par projet, regroupant toutes les versions
    written = []
    for project, dfs in per_project.items():
        dfp = pd.concat(dfs, ignore_index=True)
        out_path = out_dir / f"{project}_matrice_final.csv"
        dfp.to_csv(out_path, index=False)
        written.append(out_path.name)

    # 2) Fichier global (tous projets/versions)
    global_df.to_csv(out_dir / "global_tabular.csv", index=False)

    print("[DONE] Fichiers par projet écrits :", ", ".join(sorted(written)))
    print(f"[DONE] global_tabular.csv : {len(global_df)} lignes, {len(global_df.columns)} colonnes")
    print("Remarque: 'project' et 'version' sont conservés pour les splits/rapports, "
          "mais votre modèle ne les utilisera pas comme features.")

if __name__ == "__main__":
    main()

