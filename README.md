# Plant Disease Detection

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![Stars](https://img.shields.io/github/stars/VenuBandi/plant-disease-detection?style=social)](https://github.com/VenuBandi/plant-disease-detection)

End-to-end plant disease detection system — ResNet9 image classifier across **38 disease classes**, SQL Server database, SSIS ETL pipeline, and Gradio interface.

---

## Overview

This project combines deep learning and database engineering to detect plant diseases from leaf images. A custom ResNet9 model classifies 38 disease conditions across 14 crop types. Predictions are stored in a SQL Server database with an SSIS ETL pipeline for structured disease tracking and a Power BI dashboard for analysis.

## Dataset

[New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) — 87,000+ augmented leaf images across 38 classes.

| Split | Images |
|---|---|
| Train | ~70,295 |
| Validation | ~17,572 |

## Model

**Architecture:** ResNet9 — a lightweight 9-layer residual network with batch normalization, dropout, and max pooling.

**Training:** 1Cycle learning rate policy with early stopping, Adam optimizer, cross-entropy loss.

**38 Classes across 14 crops:**
Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato — including both diseased and healthy variants.

## Interface

Gradio web interface (`notebooks/gradio_interface.ipynb`) — upload a leaf image, get predicted disease class with probability scores.

## Database & ETL

| Component | Description |
|---|---|
| SQL Server | Separate tables per crop (Tomato, Potato, Peach, Otherplants) in `ETL_VENU` DB |
| SSIS Package | `etl/DATA604_plant_disease.ispac` — ETL pipeline loading predictions into SQL Server |
| Power BI | `reports/plant_disease_dashboard.pbix` — disease distribution and severity dashboard |

## Repo Structure

```
plant-disease-detection/
├── notebooks/
│   ├── plant_disease_classification_resnet9.ipynb   # Model training
│   └── gradio_interface.ipynb                       # Prediction UI
├── sql/
│   ├── query_tomato.sql
│   ├── query_potato.sql
│   ├── query_peach.sql
│   └── query_otherplants.sql
├── etl/
│   └── DATA604_plant_disease.ispac                  # SSIS ETL package
├── reports/
│   └── plant_disease_dashboard.pbix                 # Power BI dashboard
├── examples/
│   ├── apple_black_rot.jpg
│   ├── tomato_black_spot.jpg
│   └── Tomato_early_blight.jpg
├── class_names.txt
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/VenuBandi/plant-disease-detection
cd plant-disease-detection
pip install -r requirements.txt
jupyter lab notebooks/plant_disease_classification_resnet9.ipynb
```

Download the dataset from Kaggle (`vipoooool/new-plant-diseases-dataset`) and place it as `plant_diseases/` in the project root.

> Model weights (`.pth`) are excluded from the repo due to size. Run the training notebook to generate them.

## Stack

| Layer | Tools |
|---|---|
| Model | PyTorch · ResNet9 · 1Cycle LR |
| Interface | Gradio |
| Database | SQL Server · T-SQL |
| ETL | SSIS (SQL Server Integration Services) |
| Visualization | Power BI |

## Related Work

- [histopathologic-cancer-detection](https://github.com/VenuBandi/histopathologic-cancer-detection) — medical imaging CNN
- [structural-crack-detection](https://github.com/VenuBandi/structural-crack-detection) — infrastructure defect detection
- [intel-image-classification](https://github.com/VenuBandi/intel-image-classification) — ResNet50 scene classification
