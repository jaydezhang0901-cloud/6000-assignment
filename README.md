# K-Pop Girl Group Lyrics Analysis 🎤

A neural network-based classification model that predicts K-Pop girl group generations (1996-2021) based on song lyrics analysis.

---

## 📋 Project Overview

This project applies machine learning techniques to analyze K-Pop girl group lyrics and predict which generation a song belongs to. The model achieves **92.26% accuracy** in classifying songs into five distinct generations.

### Generation Classification

| Generation | Years | Representative Artists |
|:----------:|:-----:|------------------------|
| 一代 (Gen 1) | 1996-2002 | S.E.S, Fin.K.L, Baby V.O.X, Jewelry |
| 二代 (Gen 2) | 2003-2009 | Girls' Generation, Wonder Girls, KARA, 2NE1, f(x) |
| 三代 (Gen 3) | 2010-2013 | SISTAR, Apink, EXID, Miss A, AOA |
| 四代 (Gen 4) | 2014-2017 | TWICE, BLACKPINK, Red Velvet, MAMAMOO, GFriend |
| 五代 (Gen 5) | 2018-2021 | IZ*ONE, ITZY, aespa, IVE, (G)I-DLE, STAYC |

---

## 📊 Dataset

| Item | Description |
|------|-------------|
| **Source** | [Kpop-lyric-datasets](https://github.com/EX3exp/Kpop-lyric-datasets) |
| **Original Size** | 25,696 K-Pop songs from Melon Monthly Chart (2000-2023) |
| **Filtered Dataset** | 3,243 girl group songs |
| **Features** | Lyrics, artist, composer, lyricist, release date, chart rank |

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
├── CA6000_Submission.zip              # Complete submission package
├── Kpop_Girl_Groups_Gen1_to_Gen5.xlsx # Girl groups generation reference (Gen 1-5)
├── extract_girlgroup_data.py          # Data extraction script
├── girlgroup_songs.csv                # Processed dataset
└── README.md
```

---

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
```

```bash
# Run the complete analysis
python kpop_lyrics_analysis.py
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

---

## 📈 Results

| Metric | Value |
|--------|:-----:|
| **Test Accuracy** | **92.26%** |
| Macro F1-Score | 0.9172 |
| Training Iterations | 22 |

### Per-Class Performance

| Generation | Accuracy |
|:----------:|:--------:|
| 一代 (Gen 1) | 78.0% |
| 二代 (Gen 2) | 88.9% |
| 三代 (Gen 3) | 95.0% |
| 四代 (Gen 4) | 86.5% |
| 五代 (Gen 5) | 95.4% |

---

## 🔍 Key Findings

1. **Gen 3 & Gen 5 achieved highest accuracy (~95%)** - These generations have distinctive lyrical patterns and styles
2. **Gen 2 & Gen 4 show high accuracy** - Large sample sizes and clear generational characteristics  
3. **Gen 1 had the lowest accuracy (78%)** - Smaller sample size and stylistic overlap with Gen 2 ballads

---

## 📚 Course Information

| Item | Details |
|------|---------|
| Course | CA6000 - Applied AI Programming |
| Institution | Nanyang Technological University |
| Semester | 25S2 (2025) |

---

## 🤝 Acknowledgments

- Dataset from [EX3exp/Kpop-lyric-datasets](https://github.com/EX3exp/Kpop-lyric-datasets)
- AI coding assistance from Claude (Anthropic)

---

## 📄 License

This project is for educational purposes as part of CA6000 coursework.
