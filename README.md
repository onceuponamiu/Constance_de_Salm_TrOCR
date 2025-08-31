# 🖋️ TrOCR-HTR pour la correspondance de Constance de Salm

Système de fine-tuning **TrOCR** (OCR basé sur Transformers) pour la reconnaissance de l’écriture manuscrite dans la correspondance de Constance de Salm.

---

## 📖 Vue d’ensemble

- **TrOCR** est un modèle *Transformer end-to-end* pour l’OCR :
  - **Vision Transformer (ViT)** → encodeur d’images
  - **GPT-2** → décodeur de texte
- Ce projet réalise un fine-tuning de TrOCR sur des données d’écriture manuscrite **historique (XIXe siècle)** issues de la correspondance de Constance de Salm.

👉 Données d’entraînement uploadées sur Hugging Face :
🔗 [onceuponamiu/trocr_constance_de_salm_finetune](https://huggingface.co/onceuponamiu/trocr_constance_de_salm_finetune)

---

## 📊 Exploration des données

### Résumé
- **Total image-text pairs**: 33
- **Longueur moyenne du texte**: 2360.2 caractères
- **Longueur min / max**: 1180 / 3927
- **Textes <10 caractères**: 0

### Exemple
- **Sample text**:
  `Chacun connait cet apologue : "Un peintre avait exposé un tableau représentant un lion terrassé "par...`
- **Sample image**:
  `/training-data/verite-terrain/CdS02_Konv002-01_0034.jpg`

### Vérification des dossiers
- ✅ `training-data/verite-terrain/`
  - 📷 Images: 33
  - 📄 XML: 45
  - 🔗 Matching pairs: 33
- ✅ `training-data/sample-images/`
  - 📷 Images: 107
  - 📄 XML: 0
  - 🔗 Matching pairs: 0
- ✅ `training-data/predic-corrigees/`
  - 📷 Images: 0
  - 📄 XML: 20
  - 🔗 Matching pairs: 0

---

## 📂 Structure du dépôt

```
trocr-htr/
├── README.md                 # Ce document
├── requirements.txt          # Dépendances
├── config/                   # Configurations
│   ├── base_config.yaml
│   └── training_config.yaml
├── src/                      # Code source
│   ├── data_loader.py
│   ├── model.py
│   ├── trainer.py
│   ├── evaluator.py
│   └── utils.py
├── scripts/                  # Scripts principaux
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── notebooks/                # Jupyter notebooks
│   ├── 01_Data_Exploration.ipynb
│   ├── 02_Training_TrOCR.ipynb
│   ├── 03_Evaluation.ipynb
│   └── 04_Inference_Demo.ipynb
├── data/                     # Données (prétraitées + train/val/test)
├── edition_numerique/        # Exemples de sorties numériques
└── Google_Colab_Setup.ipynb  # Setup rapide pour Colab
```

---

## ⚙️ Installation

1.  Créer un environnement virtuel Python :
    ```bash
    python -m venv trocr_env
    source trocr_env/bin/activate  # Linux/Mac
    trocr_env\Scripts\activate     # Windows
    ```

2.  Installer les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Utilisation

1.  **Préparer les données**
    ```bash
    python scripts/prepare_data.py --source_dir training-data/verite-terrain/ --output_dir data/
    ```

2.  **Entraînement**
    ```bash
    python scripts/train.py --config config/training_config.yaml
    ```

3.  **Évaluation**
    ```bash
    python scripts/evaluate.py --model_path models/final/trocr_cds_model --test_dir data/test/
    ```

4.  **Inférence locale**
    ```bash
    python scripts/inference.py --model_path models/final/trocr_cds_model --image_path path/to/image.jpg
    ```

### 📓 Notebooks

-   `01_Data_Exploration.ipynb` : Exploration et analyse des données
-   `02_Training_TrOCR.ipynb` : Fine-tuning pas à pas
-   `03_Evaluation.ipynb` : Évaluation détaillée
-   `04_Inference_Demo.ipynb` : Démo d’inférence sur de nouvelles images

---

## 🔧 Chargement du modèle fine-tuné depuis Hugging Face

Le modèle fine-tuné est disponible publiquement :
🔗 **[onceuponamiu/trocr_constance_de_salm_finetune](https://huggingface.co/onceuponamiu/trocr_constance_de_salm_finetune)**

### Exemple en Python
```python
from transformers import VisionEncoderDecoderModel, AutoProcessor
from PIL import Image

# Charger le modèle et le processor depuis Hugging Face
model_id = "onceuponamiu/trocr_constance_de_salm_finetune"
model = VisionEncoderDecoderModel.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

# Charger une image manuscrite
image = Image.open("path/to/your/image.jpg").convert("RGB")

# Préparer les inputs
pixel_values = processor(images=image, return_tensors="pt").pixel_values

# Générer la prédiction
generated_ids = model.generate(pixel_values, max_length=512, num_beams=4)
predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

print("🔮 Transcription OCR:", predicted_text)
```

**Notes**
- `AutoProcessor` gère à la fois preprocessing image (ViT) et tokenisation texte (GPT-2).
- Toujours convertir les images en `RGB`.
- Ajustez `max_length` et `num_beams` pour un meilleur compromis qualité/temps.

Pour accélérer sur GPU :
```python
model.to("cuda")
pixel_values = pixel_values.to("cuda")
```

---

### 🔗 Intégration avec le pipeline
- Peut remplacer ou compléter les modèles Kraken.
- Export possible au format ALTO XML compatible TEI.
- Comparaison des performances avec les pipelines existants.

### 📈 Performances attendues
- **TrOCR base** : ~70–80% précision sur manuscrits historiques
- **TrOCR fine-tuné** : objectif >85% précision sur données CdS
- **Vitesse** : ~1–2 sec/image (GPU), ~5–10 sec/image (CPU)

### 💻 Configuration système
- **RAM** : min. 8 Go (16 Go+ recommandé)
- **GPU** : NVIDIA avec CUDA (fortement recommandé)
- **Stockage** : ~10 Go (modèle + cache)
- **Python** : 3.8+
