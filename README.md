# Analyse de la qualité logicielle : vers une représentation matricielle des relations inter et intra-classe

Mémoire de maîtrise (MMIA) — UQTR
Auteur : Dadah Diop
Directeur de recherche : Fadel Touré

## Vue d'ensemble

Ce dépôt contient le code source du pipeline expérimental développé dans le cadre de ce mémoire, portant sur la **prédiction de défauts logiciels en contexte inter-projets**. Contrairement aux approches classiques qui réduisent chaque classe à un vecteur de métriques scalaires, ce travail propose une **représentation matricielle explicite des relations structurelles** entre classes (héritage, appels de méthodes, dépendances, etc.), extraite automatiquement du code source Java et combinée à des métriques logicielles classiques.

Deux architectures d'apprentissage profond sont entraînées sur cette représentation hybride et comparées :

- **CNN à double entrée** — convolutions 1D sur les descripteurs relationnels, perceptron sur les métriques
- **Transformer multi-tokens** — chaque descripteur relationnel devient un token, avec encodage positionnel appris et attention multi-têtes

## Contributions principales

1. **Plugin IntelliJ IDEA** — extraction automatique et reproductible des relations inter-classes (8 types de relations) à partir du code source Java
2. **Représentation matricielle enrichie** — descripteurs relationnels bruts, combinés à des mesures de connectivité (degrés, diversité relationnelle, ratio d'asymétrie) et à des métriques logicielles classiques
3. **Évaluation comparative rigoureuse** — CNN et Transformer, entraînés et évalués selon un **protocole inter-projets strict** (aucune version du projet cible, antérieure ou postérieure, ne figure dans le pool d'entraînement)
4. **Analyse orientée maintenance préventive** — les prédictions sont interprétées pour identifier les composants structurellement à risque avant toute manifestation d'un défaut

## Résultats clés

- Évaluation sur **20 versions cibles**, réparties sur 10 projets Java open source (écosystème Apache)
- Score F1 moyen : **0.591 (CNN)** contre **0.584 (Transformer)** — les deux architectures obtiennent des performances globalement équivalentes, sans qu'un facteur structurel unique (taille, connectivité, complexité) ne permette de prédire systématiquement laquelle sera la plus performante sur un projet donné
- L'encodage positionnel appris s'est révélé déterminant pour la stabilité du Transformer, corrigeant notamment un cas d'échec complet (AUC-ROC de 0.448 à 0.818 sur un projet cible)

## Structure du dépôt

```
├── model_transformer.py    # Architecture et entraînement du Transformer
├── model_cnn.py             # Architecture et entraînement du CNN
├── generate_pools.py        # Calcul des pools d'entraînement sans fuite de données
├── run_all.py                # Lanceur automatique des 80 runs (20 projets x 2 modeles x 2 seuils)
└── README.md
```

## Protocole expérimental

Pour chaque projet cible, le pool d'entraînement **exclut systématiquement toutes les versions du même projet**, peu importe leur position chronologique par rapport à la version testée — une contrainte plus stricte que la simple exclusion de la version testée, nécessaire pour garantir une mesure fiable de la capacité de généralisation inter-projets. Cette règle est vérifiée automatiquement (via assertion) avant chaque exécution.

Chaque configuration est entraînée avec :
- Normalisation ajustée uniquement sur l'ensemble d'entraînement (`StandardScaler`/`QuantileTransformer`)
- Gestion du déséquilibre de classes (`SMOTE` sur les métriques, augmentation gaussienne, pondération de classe)
- Calibration des probabilités (temperature scaling, repli sur régression isotone)
- Seuil de décision optimisé sur l'ensemble de validation selon deux critères : $F_1$ et $F_{0.5}$

## Utilisation

```bash
# Lancer un seul projet cible (test rapide)
python run_all.py xerces_v3

# Lancer l'ensemble des 80 runs (20 projets x 2 modeles x 2 seuils)
python run_all.py
```

Chaque run est journalisé individuellement dans `logs/`, et suivi via [Weights & Biases](https://wandb.ai/).

## Dépendances principales

- TensorFlow / Keras
- scikit-learn
- imbalanced-learn (SMOTE)
- pandas, numpy
- wandb

## Citation

Si vous utilisez ce travail, merci de citer :

```
Diop, D. (2026). Analyse de la qualité logicielle : vers une représentation
matricielle des relations inter et intra-classe. Mémoire de maîtrise,
Université du Québec à Trois-Rivières.
```
