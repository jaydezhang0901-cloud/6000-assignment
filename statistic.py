# dataset_tfidf_by_generation.py
# TF-IDF 差异词分析（按 generation）

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer


# =========================
# 1. 停用词（英文 + 简单韩文）
# =========================
EN_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "that", "this",
    "to", "of", "in", "on", "for", "with", "as", "by", "at",
    "is", "are", "was", "were", "be", "been", "being",
    "oh", "yeah", "uh", "ah", "la", "na",
    "oh", "ohh", "ooh", "oohh",
    "yeah", "yea", "yah", "ya",
    "uh", "uhh", "um", "umm",
    "ah", "aha", "ha", "haha",
    "la", "lalala", "na", "nanana",
    "baby", "babe",
    "hey", "hi", "yo",
    "woo", "woah", "whoa",
    "mmm", "mm"
}

KO_STOPWORDS = {
    "이", "것", "수", "내", "네",
    "너무", "정말", "왜", "좀", "더", "다",
    "하고", "해서",
    "아", "아아", "어", "어어", "오", "오오",
    "야", "얘", "에", "응", "흥",
    "하", "하하", "헤", "헤헤",
    "음", "으음", "음음",
    "라", "라라", "나나",
    "와", "워", "워워",
    "아이", "에이",
    "그냥", "진짜", "완전",
    "막", "또"
}

STOPWORDS = EN_STOPWORDS.union(KO_STOPWORDS)


# =========================
# 2. 文本清洗
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s\uac00-\ud7a3]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize(text):
    return [
        w for w in text.split()
        if w not in STOPWORDS and len(w) > 1
    ]


# =========================
# 3. 加载数据
# =========================
def load_dataset(filepath):
    df = pd.read_csv(filepath, encoding='utf-8-sig')

    df = df[['lyrics', 'generation', 'artist', 'lyric_writer']]
    df = df.dropna(subset=['lyrics', 'generation'])

    df['cleaned_lyrics'] = df['lyrics'].apply(clean_text)
    df['tokens'] = df['cleaned_lyrics'].apply(tokenize)
    df['processed_text'] = df['tokens'].apply(lambda x: " ".join(x))

    return df


# =========================
# 4. 标签分布统计
# =========================
def label_distribution(df, col):
    return (
        df[col]
        .value_counts()
        .reset_index()
        .rename(columns={'index': col, col: 'count'})
    )


# =========================
# 5. TF-IDF 差异词分析（按 generation）
# =========================
def tfidf_by_generation(df, top_n=15, min_samples=5):
    """
    对每个 generation 计算最具区分性的 TF-IDF 词
    """
    results = {}

    for gen, sub_df in df.groupby("generation"):
        if len(sub_df) < min_samples:
            continue

        vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 1)
        )

        tfidf = vectorizer.fit_transform(sub_df['processed_text'])
        feature_names = vectorizer.get_feature_names_out()

        mean_tfidf = tfidf.mean(axis=0).A1
        top_indices = mean_tfidf.argsort()[::-1][:top_n]

        results[gen] = [
            (feature_names[i], round(mean_tfidf[i], 4))
            for i in top_indices
        ]

    return results


# =========================
# 6. 打印工具
# =========================
def print_tfidf_results(results):
    for gen, words in results.items():
        print(f"\n=== Generation: {gen} ===")
        for w, score in words:
            print(f"{w:15s} {score}")


# =========================
# 7. 主程序
# =========================
if __name__ == "__main__":

    DATASET_PATH = "girlgroup_songs.csv"   # ← 改路径

    print("Loading dataset...")
    df = load_dataset(DATASET_PATH)

    # ========= 基础分布 =========
    print("\n=== Generation Distribution ===")
    print(label_distribution(df, 'generation'))

    print("\n=== Artist Distribution (Top 10) ===")
    print(label_distribution(df, 'artist').head(10))

    print("\n=== Lyric Writer Distribution (Top 10) ===")
    print(label_distribution(df, 'lyric_writer').head(10))

    # =========================
    # 类别数量统计
    # =========================
    print("\n=== Number of unique classes ===")
    print(f"Number of generations: {df['generation'].nunique()}")
    print(f"Number of artists: {df['artist'].nunique()}")
    print(f"Number of lyric writers: {df['lyric_writer'].nunique()}")

    # ========= TF-IDF 差异词 =========
    print("\n=== TF-IDF Distinctive Words by Generation ===")
    tfidf_results = tfidf_by_generation(df, top_n=15)
    print_tfidf_results(tfidf_results)


