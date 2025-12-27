# K-Pop Girl Group Lyrics Analysis 

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A neural network-based classification model that predicts K-pop girl group generations (1996-2023) based on song lyrics analysis.

## 📋 Project Overview

This project applies machine learning techniques to analyze K-pop girl group lyrics and predict which generation a song belongs to. The model achieves **92.26% accuracy** in classifying songs into six distinct generations.

### Generation Classification

| Generation | Years | Representative Artists |
|------------|-------|------------------------|
| 一代 (Gen 1) | 1996-2002 | S.E.S, Fin.K.L, Baby V.O.X |
| 二代 (Gen 2) | 2003-2009 | Girls' Generation, Wonder Girls, 2NE1 |
| 三代 (Gen 3) | 2010-2013 | SISTAR, Apink, EXID |
| 四代 (Gen 4) | 2014-2017 | TWICE, BLACKPINK, Red Velvet |
| 五代 (Gen 5) | 2018-2021 | ITZY, aespa, (G)I-DLE |
| 六代 (Gen 6) | 2022-present | NewJeans, LE SSERAFIM, NMIXX |

## 📊 Dataset

- **Source**: [Kpop-lyric-datasets](https://github.com/EX3exp/Kpop-lyric-datasets)
- **Original Size**: 25,696 K-pop songs from Melon Monthly Chart (2000-2023)
- **Filtered Dataset**: 3,243 girl group songs
- **Features**: Lyrics, artist, composer, lyricist, release date, chart rank

## 🛠️ Technical Stack

- **Language**: Python 3.8+
- **ML Framework**: scikit-learn
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Text Processing**: TF-IDF Vectorization

## 📁 Repository Structure

```
├── CA6000_Submission.zip          # Complete submission package
├── Kpop_Girl_Groups_Gen1_to_Gen5.xlsx  # Girl groups reference data
├── extract_girlgroup_data.py      # Data extraction script
├── girlgroup_songs.csv            # Processed dataset
└── README.md
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Run Analysis

```python
# Load dataset
import pandas as pd
df = pd.read_csv('girlgroup_songs.csv', encoding='utf-8-sig')

# Run the complete analysis
python kpop_lyrics_analysis.py
```

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
Output Layer (6 classes - Softmax)
```

## 📈 Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 92.26% |
| Macro F1-Score | 0.9172 |
| Training Iterations | 22 |

### Per-Class Performance

| Generation | Accuracy |
|------------|----------|
| Gen 1 | 78.0% |
| Gen 2 | 88.9% |
| Gen 3 | 95.0% |
| Gen 4 | 86.5% |
| Gen 5 | 100.0% |
| Gen 6 | 95.4% |

## 🔍 Key Findings

1. **Gen 6 achieved 100% accuracy** - Modern groups like NewJeans have distinctive lyrical patterns with high English content
2. **Gen 2 & Gen 4 show high accuracy** - Large sample sizes and clear generational characteristics
3. **Gen 1 had the lowest accuracy** - Smaller sample size and stylistic overlap with Gen 2 ballads

## 📚 Course Information

- **Course**: CA6000 - Applied AI Programming
- **Institution**: Nanyang Technological University
- **Semester**: 25S2 (2025)

## 🤝 Acknowledgments

- Dataset from [EX3exp/Kpop-lyric-datasets](https://github.com/EX3exp/Kpop-lyric-datasets)
- AI coding assistance from Claude (Anthropic)

## 📄 License

This project is for educational purposes as part of the CA6000 coursework.
