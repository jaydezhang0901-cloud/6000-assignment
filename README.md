# K-Pop Girl Group Lyrics Analysis 🎤

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)]()

A neural network-based classification model that predicts K-Pop girl group generations based on song lyrics analysis.

---

## 📋 Project Overview

This project applies machine learning techniques to analyze K-Pop girl group lyrics and predict which generation a song belongs to. The model achieves **~90% accuracy** in classifying songs into five distinct generations.

### Generation Classification (5-Generation Standard)

Following the reference table (韩国女团世代表_一代至五代.xlsx):

| Generation | Years | Representative Artists |
|:----------:|:-----:|------------------------|
| Gen 1 (一代) | 1996-2002 | S.E.S, Fin.K.L, Baby V.O.X, Jewelry |
| Gen 2 (二代) | 2003-2009 | Girls' Generation, Wonder Girls, KARA, 2NE1, f(x) |
| Gen 3 (三代) | 2010-2013 | SISTAR, Apink, EXID, Miss A, AOA |
| Gen 4 (四代) | 2014-2017 | TWICE, BLACKPINK, Red Velvet, MAMAMOO, GFriend |
| Gen 5 (五代) | 2018+ | IZ*ONE, ITZY, aespa, IVE, (G)I-DLE, NewJeans, LE SSERAFIM |

---

## 📊 Dataset

| Item | Description |
|------|-------------|
| **Source** | [Kpop-lyric-datasets](https://github.com/EX3exp/Kpop-lyric-datasets) |
| **Original Size** | 25,696 K-Pop songs from Melon Monthly Chart (2000-2023) |
| **Filtered Dataset** | 3,243 girl group songs |
| **Cleaned Dataset** | ~2,967 songs after data cleaning |
| **File** | `girlgroup_songs.csv` |

### Dataset Columns
- `generation`: Girl group generation (一代 ~ 五代)
- `artist`: Artist name
- `song_name`: Song title
- `lyrics`: Full lyrics text
- `year`, `month`: Chart appearance date
- `rank`: Chart ranking (1-100)
- `lyrics_length`: Character count of lyrics

---

## 🛠️ Technical Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.8+ |
| ML Framework | scikit-learn |
| Data Processing | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Text Processing | TF-IDF Vectorization |

---

## 📁 Repository Structure

```
├── kpop_lyrics_analysis.py        # Main analysis script
├── girlgroup_songs.csv            # Dataset (3,243 songs)
├── model_results.png              # Visualization output
├── CA6000_Report_Final.docx       # Assignment report
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Run Analysis

```bash
python kpop_lyrics_analysis.py
```

### Expected Output

```
============================================================
K-Pop Girl Group Lyrics Analysis
Neural Network-based Generation Prediction Model
============================================================

SECTION 1: Data Import and Initial Inspection
Dataset Shape: (3243, 14)
...

SECTION 6: The Accuracy of the Eventual Model
========================================
  OVERALL TEST ACCURACY: ~90%
========================================
```

---

## 🧠 Model Architecture

**Multi-Layer Perceptron (MLP) Neural Network**

```
Input Layer (3,000 TF-IDF features)
           ↓
Hidden Layer 1 (256 neurons + ReLU)
           ↓
Hidden Layer 2 (128 neurons + ReLU)
           ↓
Hidden Layer 3 (64 neurons + ReLU)
           ↓
Output Layer (5 classes - Softmax)
```

### Training Configuration
- **Optimizer**: Adam
- **Early Stopping**: Enabled (10% validation hold-out)
- **Max Iterations**: 200
- **Train/Test Split**: 80/20 (stratified)

---

## 📈 Results

| Metric | Value |
|--------|:-----:|
| **Test Accuracy** | **~90%** |
| Macro F1-Score | ~0.90 |
| Best Validation Score | ~0.95 |

### Per-Class Performance

| Generation | Precision | Recall | F1-Score |
|:----------:|:---------:|:------:|:--------:|
| Gen 1 | ~0.80 | ~0.70 | ~0.75 |
| Gen 2 | ~0.90 | ~0.95 | ~0.92 |
| Gen 3 | ~0.90 | ~0.85 | ~0.87 |
| Gen 4 | ~0.95 | ~0.95 | ~0.95 |
| Gen 5 | ~0.95 | ~0.95 | ~0.95 |

---

## 🔍 Key Findings

1. **Gen 4 & Gen 5 achieved highest accuracy** - Modern groups have distinctive lyrical patterns with more English content
2. **Gen 2 showed strong performance** - Largest sample size (34%) with clear characteristics
3. **Gen 1 had lowest recall** - Smallest sample size (6.9%) and stylistic overlap with Gen 2 ballads
4. **The model successfully learned generation-specific vocabulary and linguistic patterns**

---

## 📚 Course Information

| Item | Details |
|------|---------|
| Course | CA6000 - Applied AI Programming |
| Institution | Nanyang Technological University |
| Semester | 25S2 (2025) |
| Student | ZHANG XINYING |

---

## 🤝 Acknowledgments

- Dataset: [EX3exp/Kpop-lyric-datasets](https://github.com/EX3exp/Kpop-lyric-datasets)
- Generation reference: 韩国女团世代表_一代至五代.xlsx
- AI coding assistance: Claude (Anthropic)

---

## 📄 License

This project is for educational purposes as part of CA6000 coursework.
