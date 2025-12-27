# pip install tensorflow==2.10.0 pandas scikit-learn

import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout


# =========================
# 1. 文本清洗
# =========================
def clean_text(text):
    """
    仅保留英文、数字、韩文和空格
    """
    text = str(text).lower()
    text = re.sub(r'[^\w\s\uac00-\ud7a3]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# =========================
# 2. 数据加载与预处理
# =========================
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath, encoding='utf-8-sig')

    # 只保留需要的列
    df = df[['lyrics', 'generation', 'artist', 'lyric_writer']]

    # 删除缺失标签的样本
    df = df.dropna(subset=['lyrics', 'generation', 'artist', 'lyric_writer'])

    # 清洗歌词
    df['cleaned_lyrics'] = df['lyrics'].apply(clean_text)

    return df


# =========================
# 3. 特征与标签准备
# =========================
def prepare_features(df, max_words=5000, max_len=100):
    # 文本 → 序列
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['cleaned_lyrics'])

    X = tokenizer.texts_to_sequences(df['cleaned_lyrics'])
    X = pad_sequences(X, maxlen=max_len, padding='post', truncating='post')

    # 多标签编码
    gen_encoder = LabelEncoder()
    artist_encoder = LabelEncoder()
    writer_encoder = LabelEncoder()

    y_gen = gen_encoder.fit_transform(df['generation'])
    y_artist = artist_encoder.fit_transform(df['artist'])
    y_writer = writer_encoder.fit_transform(df['lyric_writer'])

    return (
        X,
        y_gen, y_artist, y_writer,
        tokenizer,
        gen_encoder, artist_encoder, writer_encoder,
        max_len
    )


# =========================
# 4. 多输出 LSTM 模型
# =========================
def build_multi_output_model(
    max_words,
    max_len,
    gen_classes,
    artist_classes,
    writer_classes
):
    inputs = Input(shape=(max_len,))

    x = Embedding(
        input_dim=max_words,
        output_dim=128,
        input_length=max_len
    )(inputs)

    x = LSTM(64)(x)
    x = Dropout(0.5)(x)

    gen_output = Dense(
        gen_classes,
        activation='softmax',
        name='generation'
    )(x)

    artist_output = Dense(
        artist_classes,
        activation='softmax',
        name='artist'
    )(x)

    writer_output = Dense(
        writer_classes,
        activation='softmax',
        name='lyric_writer'
    )(x)

    model = Model(
        inputs=inputs,
        outputs=[gen_output, artist_output, writer_output]
    )

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# =========================
# 5. 主流程
# =========================
if __name__ == "__main__":

    DATASET_PATH = "girlgroup_songs.csv"   # ← 改成你的 CSV 路径

    MAX_WORDS = 5000
    MAX_LEN = 120
    EPOCHS = 10
    BATCH_SIZE = 32

    print("Loading and preprocessing data...")
    df = load_and_preprocess_data(DATASET_PATH)

    print("Preparing features and labels...")
    (
        X,
        y_gen, y_artist, y_writer,
        tokenizer,
        gen_encoder, artist_encoder, writer_encoder,
        max_len
    ) = prepare_features(df, MAX_WORDS, MAX_LEN)

    # 划分训练 / 测试
    X_train, X_test, \
    ygen_train, ygen_test, \
    yartist_train, yartist_test, \
    ywriter_train, ywriter_test = train_test_split(
        X, y_gen, y_artist, y_writer,
        test_size=0.2,
        random_state=42
    )

    print("Building model...")
    model = build_multi_output_model(
        max_words=MAX_WORDS,
        max_len=max_len,
        gen_classes=len(gen_encoder.classes_),
        artist_classes=len(artist_encoder.classes_),
        writer_classes=len(writer_encoder.classes_)
    )

    model.summary()

    print("\nStarting training...\n")
    model.fit(
        X_train,
        {
            'generation': ygen_train,
            'artist': yartist_train,
            'lyric_writer': ywriter_train
        },
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1
    )

    print("\nEvaluating on test set...\n")
    results = model.evaluate(
        X_test,
        {
            'generation': ygen_test,
            'artist': yartist_test,
            'lyric_writer': ywriter_test
        }
    )

    print("Test results:", results)


    # =========================
    # 6. 预测函数
    # =========================
    def predict_all(lyrics_text):
        cleaned = clean_text(lyrics_text)
        seq = tokenizer.texts_to_sequences([cleaned])
        pad = pad_sequences(seq, maxlen=max_len, padding='post')

        pred_gen, pred_artist, pred_writer = model.predict(pad)

        return {
            'generation': gen_encoder.inverse_transform([np.argmax(pred_gen)])[0],
            'artist': artist_encoder.inverse_transform([np.argmax(pred_artist)])[0],
            'lyric_writer': writer_encoder.inverse_transform([np.argmax(pred_writer)])[0]
        }


    # 示例预测
    sample_lyrics = "그대는 나의 햇살 너만 있으면 돼"
    prediction = predict_all(sample_lyrics)

    print("\nPrediction Example:")
    for k, v in prediction.items():
        print(f"{k}: {v}")
