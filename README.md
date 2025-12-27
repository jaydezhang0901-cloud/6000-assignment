# K-Pop Girl Group Lyrics Analysis 

A neural network-based classification model that predicts K-Pop girl group generations based on song lyrics analysis.

---

## 📋 Project Overview

This project applies machine learning techniques to analyze K-Pop girl group lyrics and predict which generation a song belongs to. The model achieves **89.90% accuracy** in classifying songs into five distinct generations.

### Generation Classification (5-Generation Standard)

| Generation | Years | Representative Artists |
|:----------:|:-----:|------------------------|
| 一代 (Gen 1) | 1996-2002 | S.E.S, Fin.K.L, Baby V.O.X, Jewelry |
| 二代 (Gen 2) | 2003-2009 | Girls' Generation, Wonder Girls, KARA, 2NE1, f(x) |
| 三代 (Gen 3) | 2010-2013 | SISTAR, Apink, EXID, Miss A, AOA |
| 四代 (Gen 4) | 2014-2017 | TWICE, BLACKPINK, Red Velvet, MAMAMOO, GFriend |
| 五代 (Gen 5) | 2018+ | IZ*ONE, ITZY, aespa, IVE, (G)I-DLE, NewJeans, LE SSERAFIM |

---

## 📊 Dataset

| Item | Description |
|------|-------------|
| **Source** | [Kpop-lyric-datasets](https://github.com/EX3exp/Kpop-lyric-datasets) |
| **Original Size** | 25,696 K-Pop songs from Melon Monthly Chart (2000-2023) |
| **Filtered Dataset** | 3,243 girl group songs |
| **File** | `girlgroup_songs.csv` |
| **Features** | Lyrics, artist, generation, song_name, album, release_date, genre, chart rank |

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
├── 韩国女团世代表_一代至五代.xlsx      # Girl groups generation reference
├── extract_girlgroup_data.py          # Data extraction script
├── girlgroup_songs.csv                # Processed dataset (3,243 songs)
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
import pandas as pd
df = pd.read_csv('girlgroup_songs.csv', encoding='utf-8-sig')
print(df['generation'].value_counts())
```

```bash
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
| **Test Accuracy** | **89.90%** |
| Macro F1-Score | 0.88 |
| Best Validation Score | 0.9538 |
| Training Iterations | 17 |

### Per-Class Performance

| Generation | Precision | Recall | F1-Score |
|:----------:|:---------:|:------:|:--------:|
| Gen 1 | 0.90 | 0.68 | 0.78 |
| Gen 2 | 0.84 | 0.93 | 0.88 |
| Gen 3 | 0.87 | 0.82 | 0.84 |
| Gen 4 | 0.95 | 0.94 | 0.95 |
| Gen 5 | 0.96 | 0.92 | 0.94 |

---

## 🔍 Key Findings

1. **Gen 4 & Gen 5 achieved highest accuracy (~94-95%)** - Modern groups have distinctive lyrical patterns
2. **Gen 2 showed strong performance (88%)** - Largest sample size (34%) with clear characteristics
3. **Gen 1 had lowest recall (68%)** - Smallest sample size (6.8%) and stylistic overlap with Gen 2

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
- Generation reference from 韩国女团世代表_一代至五代.xlsx
- AI coding assistance from Claude (Anthropic)

---

## 📄 License

This project is for educational purposes as part of CA6000 coursework.
