# TrOCR-HTR pour la correspondance de Constance de Salm

Système de fine-tuning TrOCR (Reconnaissance optique de caractères basée sur les Transformers) pour la reconnaissance de l’écriture manuscrite dans la correspondance de Constance de Salm.

## Vue d’ensemble

TrOCR est un modèle Transformer de bout en bout pour l’OCR, combinant ViT (Vision Transformer) pour encoder l’image et GPT-2 pour décoder le texte. Ce projet réalise un fine-tuning de TrOCR sur des données d’écriture manuscrite historique de Constance de Salm.

## Structure du répertoire

```
trocr-htr/
├── README.md                    # Ce document
├── requirements.txt             # Dépendances
├── config/
│   ├── base_config.yaml        # Configuration de base
│   └── training_config.yaml    # Configuration d’entraînement
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Gestion des données d’entraînement
│   ├── model.py                # Wrapper du modèle TrOCR
│   ├── trainer.py              # Pipeline d’entraînement
│   ├── evaluator.py            # Évaluation du modèle
│   └── utils.py                # Fonctions utilitaires
├── scripts/
│   ├── prepare_data.py         # Préparation des données
│   ├── train.py                # Script d’entraînement
│   ├── evaluate.py             # Script d’évaluation
│   └── inference.py            # Script d’inférence
├── notebooks/
│   ├── 01_Data_Exploration.ipynb     # Exploration des données
│   ├── 02_Training_TrOCR.ipynb       # Entraînement de TrOCR
│   ├── 03_Evaluation.ipynb           # Évaluation des résultats
│   └── 04_Inference_Demo.ipynb       # Démonstration d’inférence
├── data/                       # Données prétraitées
│   ├── train/
│   ├── val/
│   └── test/
├── models/                     # Modèles entraînés
│   ├── checkpoints/
│   └── final/
└── outputs/                    # Résultats d’inférence
    ├── predictions/
    └── evaluations/
```

## Installation

1. Créer un environnement virtuel Python :
```bash
python -m venv trocr_env
source trocr_env/bin/activate  # Linux/Mac
# ou
trocr_env\Scripts\activate     # Windows
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

### 1. Préparer les données
```bash
python scripts/prepare_data.py --source_dir ../htr/verite-terrain/ --output_dir data/
```

### 2. Entraînement
```bash
python scripts/train.py --config config/training_config.yaml
```

### 3. Évaluation
```bash
python scripts/evaluate.py --model_path models/final/trocr_cds_model --test_dir data/test/
```

### 4. Inférence
```bash
python scripts/inference.py --model_path models/final/trocr_cds_model --image_path path/to/image.jpg
```

## Notebooks

1. **01_Data_Exploration.ipynb** : Exploration et analyse des données d’entraînement
2. **02_Training_TrOCR.ipynb** : Guide pas à pas pour le fine-tuning de TrOCR
3. **03_Evaluation.ipynb** : Évaluation détaillée du modèle
4. **04_Inference_Demo.ipynb** : Démonstration sur de nouvelles images

## Intégration avec le pipeline actuel

Le modèle TrOCR après entraînement peut :
- Remplacer ou compléter les modèles Kraken existants
- Produire des résultats au format ALTO XML compatibles avec le pipeline TEI
- Comparer les performances avec les modèles Kraken existants

## Performances attendues

- **Base TrOCR** : ~70-80% de précision sur texte manuscrit
- **TrOCR fine-tuné** : Objectif >85% de précision sur les données Constance de Salm
- **Vitesse** : ~1-2 secondes/image sur GPU, ~5-10 secondes/image sur CPU

## Configuration système requise

- **RAM** : Minimum 8 Go, recommandé 16 Go+
- **GPU** : Recommandé NVIDIA GPU avec support CUDA
- **Stockage** : ~10 Go pour le modèle et le cache
- **Python** : 3.8+
