"""
Lanceur automatique - execute tous les runs (Transformer/CNN x F1/F0.5)
pour tous les projets cibles, les uns apres les autres, sans intervention.

Usage :
    python run_all.py                  -> lance TOUT (100 runs, tres long)
    python run_all.py xerces_v3        -> lance seulement les 4 runs de ce projet
    python run_all.py xerces_v3 lucene_v1   -> plusieurs projets a la fois

Chaque run sauvegarde sa sortie complete dans un fichier .log separe,
dans le dossier logs/. Si un run plante, le script continue avec le
suivant au lieu de tout arreter (l'erreur est notee dans logs/erreurs.txt).
"""

import subprocess
import sys
import time
from pathlib import Path
from generate_pools import build_pool  # reutilise la fonction sans fuite deja testee

# Dossier ou seront sauvegardes tous les logs, cree s'il n'existe pas deja.
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Les 20 projets cibles retenus pour le protocole final (2 versions par
# projet, en maximisant le contraste en taille et taux de defauts).
TARGETS = [
    "ant_v3", "ant_v4",
    "camel_v3", "camel_v4",
    "jedit_v1", "jedit_v4",
    "log4j_v2", "log4j_v3",
    "lucene_v1", "lucene_v3",
    "poi_v1", "poi_v4",
    "synapse_v2", "synapse_v3",
    "velocity_v1", "velocity_v2",
    "xalan_v1", "xalan_v4",
    "xerces_v3", "xerces_v4",
]

# Associe chaque modele a son script et a son propre projet wandb, pour
# ne jamais melanger les resultats du CNN et du Transformer.
MODELS = {
    "transformer": ("model_transformer.py", "prediction-bug-transformer"),
    "cnn": ("model_cnn.py", "prediction-bug-cnn"),
}
THRESHOLDS = ["f1", "f0.5"]

PROJECTS_DIR = r".\données_final"
SEED = 2025


# Execute un seul run complet : calcule le pool sans fuite, verifie une
# deuxieme fois qu'aucune version du projet cible ne s'y trouve, lance
# le script d'entrainement correspondant, et sauvegarde toute sa sortie
# dans un fichier .log dedie.
def run_one(target: str, model_name: str, script: str, wandb_project: str, mode: str) -> bool:
    pool = build_pool(target)
    # jamais de version du meme projet dans le pool
    proj_prefix = target.split("_v")[0] + "_"
    assert not any(v.startswith(proj_prefix) for v in pool), \
        f"FUITE DETECTEE pour {target} !"

    run_name = f"{target}_{model_name}_{mode}"
    log_file = LOG_DIR / f"{run_name}.log"

    cmd = [
        sys.executable, "-u", script,   # sys.executable  meme interpreteur/venv que ce script
        "--projects_dir", PROJECTS_DIR,
        "--pool", *pool,
        "--test", target,
        "--decision_mode", mode,
        "--seed", str(SEED),
        "--use_smote", "--use_class_weight",
        "--use_wandb",
        "--wandb_project", wandb_project,
        "--wandb_name", run_name,
        "--wandb_group", "protocole_strict_2026",
        "--wandb_tags", model_name, mode, "inter_projets_strict",
    ]

    print(f"[LANCEMENT] {run_name} ...", flush=True)
    t0 = time.time()

    # Execute reellement la commande et redirige toute sa sortie vers le fichier log.
    with open(log_file, "w", encoding="utf-8") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    elapsed = time.time() - t0
    ok = result.returncode == 0   # 0 = succes, autre chiffre = erreur
    status = "OK" if ok else "ECHEC"
    print(f"[{status}] {run_name} ({elapsed:.0f}s) -> {log_file}", flush=True)
    return ok


# Fonction principale : enchaine automatiquement toutes les combinaisons
# (projets x modeles x seuils), continue meme si un run echoue, et
# sauvegarde la liste des echecs a la fin pour pouvoir les relancer.
def main():
    # Sans argument : traite tous les TARGETS. Avec des arguments : ne
    # traite que les projets passes en ligne de commande.
    targets = sys.argv[1:] if len(sys.argv) > 1 else TARGETS

    total = len(targets) * len(MODELS) * len(THRESHOLDS)
    print(f"=== {len(targets)} projet(s) x 2 modeles x 2 seuils = {total} runs ===\n")

    errors = []
    done = 0

    for target in targets:
        for model_name, (script, wandb_project) in MODELS.items():
            for mode in THRESHOLDS:
                done += 1
                print(f"--- Run {done}/{total} ---")
                try:
                    ok = run_one(target, model_name, script, wandb_project, mode)
                    if not ok:
                        errors.append(f"{target} / {model_name} / {mode}")
                except Exception as e:
                    # Un run qui plante ne doit jamais arreter les autres.
                    print(f"[ERREUR] {target}/{model_name}/{mode}: {e}")
                    errors.append(f"{target} / {model_name} / {mode} -> {e}")

    print(f"\n=== Termine : {done - len(errors)}/{total} reussis ===")
    if errors:
        print(f"{len(errors)} run(s) en echec :")
        for e in errors:
            print("  -", e)
        (LOG_DIR / "erreurs.txt").write_text("\n".join(errors), encoding="utf-8")
        print(f"\nListe sauvegardee dans {LOG_DIR / 'erreurs.txt'}")


if __name__ == "__main__":
    main()