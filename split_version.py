import pandas as pd
from pathlib import Path

def split_project_versions_in_folder(
    folder: str,
    all_suffix: str = "_matrice_final.csv",
    version_col: str = "version",
    project_col: str = "project",
):
    """
    Parcourt un dossier, trouve les fichiers '*_all.csv',
    et crée un CSV par version : 'ant_v1.csv', 'camel_v2.csv', etc.
    """

    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Dossier introuvable: {folder}")

    # On cherche tous les fichiers du type ant_all.csv, camel_all.csv, etc.
    all_files = list(folder_path.glob(f"*{all_suffix}"))
    if not all_files:
        print(f"[WARN] Aucun fichier '*{all_suffix}' trouvé dans {folder_path.resolve()}")
        return

    print(f"[INFO] {len(all_files)} fichier(s) '*{all_suffix}' trouvé(s) dans {folder_path.resolve()}")

    for csv_path in all_files:
        print(f"\n[INFO] Traitement de: {csv_path.name}")
        df = pd.read_csv(csv_path)

        # Vérification colonnes
        if version_col not in df.columns:
            print(f"  [SKIP] Colonne '{version_col}' absente, on ignore ce fichier.")
            continue
        if project_col not in df.columns:
            print(f"  [SKIP] Colonne '{project_col}' absente, on ignore ce fichier.")
            continue

        # On récupère le nom du projet :
        # - soit à partir de la colonne 'project'
        # - soit, en fallback, à partir du nom du fichier ant_all -> 'ant'
        unique_projects = df[project_col].dropna().unique()
        if len(unique_projects) == 1:
            project_name = str(unique_projects[0])
        else:
            # Si plusieurs projets dans le même fichier, on découpe par projet aussi
            print(f"  [WARN] Plusieurs projets trouvés dans {csv_path.name}: {unique_projects}")
            print("        On va traiter chaque (project, version) séparément.")
            project_name = None  # on le gérera dans la boucle

        # Groupby par (project, version) si plusieurs projets, sinon par version uniquement
        if project_name is None:
            group_cols = [project_col, version_col]
        else:
            group_cols = [version_col]

        grouped = df.groupby(group_cols)

        for keys, df_sub in grouped:
            if project_name is None:
                proj, ver = keys  # (project, version)
                proj_str = str(proj)
                ver_str = str(ver)
            else:
                proj_str = project_name
                ver_str = str(keys)  # ici keys = version

            # On nettoie un peu la version pour le nom de fichier
            ver_str_clean = ver_str.replace(" ", "")
            # Exemple: ant_v1.csv
            out_name = f"{proj_str}_{ver_str_clean}.csv"
            out_path = folder_path / out_name

            df_sub.to_csv(out_path, index=False)
            print(f"  [OK] Version extraite -> {out_name} ({len(df_sub)} lignes)")

    print("\n[FIN] Extraction des versions terminée ✅")


if __name__ == "__main__":
    # 🔧 MODIFIE ce dossier si besoin
    folder = r".\out"

    split_project_versions_in_folder(folder)
