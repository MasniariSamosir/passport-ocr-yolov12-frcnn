# Dataset Information

This document provides information about the dataset used in this research, including its source, preparation process, and usage guidelines.

---

## Dataset Source

- **Dataset Name:** Generated Passports Segmentation
- **Dataset Used:** Generated USA Passport
- **Platform:** Kaggle
- **Author:** TrainingDataPro
- **URL:** https://www.kaggle.com/datasets/trainingdatapro/generated-passports-segmentation

---

## Dataset Description

The project uses synthetic USA passport document images obtained from the Kaggle dataset.

The images were manually annotated using **Label Studio** to generate object detection labels for passport information fields. The annotations were then converted into the formats required for training and evaluating the **YOLOv12** and **Faster R-CNN** models.

The annotated dataset was used for automatic passport information extraction based on Optical Character Recognition (OCR).

---

## Annotation Classes

The following passport information fields were annotated:

- Photo
- Passport Number
- Surname
- Given Names
- Nationality
- Date of Birth
- Place of Birth
- Sex
- Date of Issue
- Date of Expiry
- Machine Readable Zone (MRZ)

---

## Dataset Preparation

The dataset preparation process consisted of the following steps:

1. Download synthetic passport images from Kaggle.
2. Perform manual annotation using Label Studio.
3. Export annotation results.
4. Convert annotations into YOLOv12 and Faster R-CNN formats.
5. Split the dataset into training and validation sets.
6. Train and evaluate the object detection models.

---

## Repository Notice

The original dataset is **not included** in this repository.

Please download the dataset directly from the official Kaggle page and comply with its license and terms of use.

---

## Acknowledgement

The author gratefully acknowledges **TrainingDataPro** for providing the passport document dataset through Kaggle.
