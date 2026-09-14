"""
Générateur de pools d'entraînement pour notre protocole inter-projets strict.

Principe : pour chaque projet cible, le pool d'entraînement exclut
TOUTES les versions du même projet (peu importe qu'elles soient
antérieures ou postérieures à la version testée), conformément à
l'Option retenue dans le memoire.
"""

import sys

# Le corpus complet : chaque projet et son nombre de versions disponibles.
CORPUS = {
    "ant": 4, "camel": 4, "jedit": 4, "log4j": 3, "lucene": 3,
    "poi": 4, "velocity": 2, "xalan": 4, "xerces": 4, "synapse": 3,
}

# Genere automatiquement la liste des 35 versions (ex: "ant_v1", "ant_v2"...)
# a partir du dictionnaire ci-dessus, plutot que de les taper a la main.
ALL_VERSIONS = [
    f"{proj}_v{v}"
    for proj, n in CORPUS.items()
    for v in range(1, n + 1)
]

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


# Retourne le pool d'entrainement pour un projet cible donne, en
# excluant toutes les versions de ce meme projet (peu importe si elles
# sont anterieures ou posterieures) -- c'est ce qui garantit l'absence
# de fuite de donnees entre versions d'un meme systeme.
def build_pool(target: str) -> list[str]:
    target_proj = target.split("_v")[0]
    return [v for v in ALL_VERSIONS if not v.startswith(target_proj + "_")]


# Associe chaque script a son propre projet wandb, pour ne jamais
# melanger les resultats du CNN et du Transformer.
WANDB_PROJECTS = {
    "model_transformer.py": "prediction-bug-transformer",
    "model_cnn.py": "prediction-bug-cnn",
}


# Construit et affiche (sans l'executer) la commande complete a lancer
# pour un projet cible, un modele et un seuil donnes -- pratique pour
# verifier ou copier une commande a la main avant de s'engager.
def print_command(target: str, model_script: str = "model_transformer.py",
                   decision_mode: str = "f1", use_wandb: bool = True,
                   seed: int = 2025):
    pool = build_pool(target)
    pool_str = " ".join(pool)
    model_name = "transformer" if "transformer" in model_script else "cnn"
    run_name = f"{target}_{model_name}_{decision_mode}"
    cmd = (
        f'python -u "{model_script}" '
        f'--projects_dir ".\\données_final" '
        f'--pool {pool_str} '
        f'--test {target} '
        f'--decision_mode {decision_mode} '
        f'--seed {seed} '
        f'--use_smote --use_class_weight'
    )
    if use_wandb:
        wandb_project = WANDB_PROJECTS.get(model_script, "bug-prediction-strict")
        cmd += (
            f' --use_wandb'
            f' --wandb_project {wandb_project}'
            f' --wandb_name {run_name}'
            f' --wandb_group protocole_strict_2026'
            f' --wandb_tags {model_name} {decision_mode} inter_projets_strict'
        )
    print(cmd)
    print()


if __name__ == "__main__":
    # Sans argument : affiche les commandes pour tous les TARGETS.
    # Avec un argument : affiche seulement celles du projet donne.
    if len(sys.argv) > 1:
        targets_to_run = [sys.argv[1]]
    else:
        targets_to_run = TARGETS

    print(f"# {len(targets_to_run)} cible(s) x 2 modeles x 2 seuils = "
          f"{len(targets_to_run) * 4} runs au total\n")

    for t in targets_to_run:
        pool = build_pool(t)
        # Deuxieme verification, independante de build_pool(), qu'aucune
        # fuite ne s'est glissee -- si elle echoue, le script plante
        # volontairement plutot que d'afficher une commande contaminee.
        assert not any(v.startswith(t.split('_v')[0] + '_') for v in pool), \
            f"ERREUR: fuite detectee pour {t}!"

        print(f"# ===== {t} ({len(pool)} projets dans le pool) =====")
        for script in ["model_transformer.py", "model_cnn.py"]:
            for mode in ["f1", "f0.5"]:
                print_command(t, model_script=script, decision_mode=mode)