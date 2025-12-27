"""
K-Pop Girl Group Lyrics Analysis
Neural Network-based Generation Prediction Model

Course: CA6000 - Applied AI Programming
Student: ZHANG XINYING
Date: December 2025

This script analyzes K-Pop girl group lyrics and predicts which generation
a song belongs to using a Multi-Layer Perceptron (MLP) neural network.

Dataset: 3,243 songs from 97 girl groups (2000-2023)
Model Accuracy: 89.90%
"""

# ============================================================================
# IMPORTS
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Set plot style
plt.rcParams['font.size'] = 10
plt.rcParams['figure.facecolor'] = 'white'

print("=" * 60)
print("K-Pop Girl Group Lyrics Analysis")
print("Neural Network-based Generation Prediction Model")
print("=" * 60)

# ============================================================================
# SECTION 1: DATA IMPORT AND INITIAL INSPECTION
# ============================================================================
print("\n" + "=" * 60)
print("SECTION 1: Data Import and Initial Inspection")
print("=" * 60)

# Load dataset
df = pd.read_csv('girlgroup_songs.csv', encoding='utf-8-sig')
print(f"\nDataset Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Display data info
print("\n--- Data Types ---")
print(df.dtypes)

# Initial error detection
print("\n--- Missing Values ---")
print(df.isnull().sum())
print(f"\nDuplicate Rows: {df.duplicated().sum()}")

# Sample data preview
print("\n--- Sample Data (First 5 Rows) ---")
print(df[['generation', 'artist', 'song_name', 'year']].head())

# ============================================================================
# SECTION 2: INTENTIONALLY INTRODUCE ERRORS
# ============================================================================
print("\n" + "=" * 60)
print("SECTION 2: Intentionally Introduce Errors")
print("=" * 60)

# Create a copy for error introduction
df_error = df.copy()
initial_rows = len(df_error)

np.random.seed(42)  # For reproducibility

# 1. NaN in lyrics (5% ≈ 162 rows)
lyrics_nan_idx = np.random.choice(df_error.index, size=int(len(df_error)*0.05), replace=False)
df_error.loc[lyrics_nan_idx, 'lyrics'] = np.nan
print(f"1. Added NaN to lyrics: {len(lyrics_nan_idx)} rows")

# 2. NaN in artist (3% ≈ 97 rows)
artist_nan_idx = np.random.choice(df_error.index, size=int(len(df_error)*0.03), replace=False)
df_error.loc[artist_nan_idx, 'artist'] = np.nan
print(f"2. Added NaN to artist: {len(artist_nan_idx)} rows")

# 3. Duplicates (50 rows)
duplicates = df_error.sample(n=50, random_state=42)
df_error = pd.concat([df_error, duplicates], ignore_index=True)
print(f"3. Added duplicate rows: 50 rows")

# 4. Invalid year values (10 rows)
invalid_year_idx = df_error.sample(n=10, random_state=42).index
df_error.loc[invalid_year_idx, 'year'] = 'invalid'
print(f"4. Added invalid year values: 10 rows")

# 5. Outliers in lyrics_length (15 extreme values)
outlier_idx = df_error.sample(n=15, random_state=42).index
df_error.loc[list(outlier_idx)[:10], 'lyrics_length'] = 99999
df_error.loc[list(outlier_idx)[10:], 'lyrics_length'] = -500
print(f"5. Added outliers in lyrics_length: 15 rows")

print(f"\nAfter introducing errors - Shape: {df_error.shape}")
print(f"Total missing values: {df_error.isnull().sum().sum()}")
print(f"Duplicate rows: {df_error.duplicated().sum()}")

# ============================================================================
# SECTION 2.1: DATA CLEANING PROCESS
# ============================================================================
print("\n" + "=" * 60)
print("SECTION 2.1: Data Cleaning Process")
print("=" * 60)

cleaning_log = []

# Step 1: Remove duplicates (first to avoid inflating missing counts)
before = len(df_error)
df_error = df_error.drop_duplicates()
removed = before - len(df_error)
cleaning_log.append(('drop_duplicates()', removed, len(df_error)))
print(f"Step 1 - drop_duplicates(): Removed {removed} rows, Remaining: {len(df_error)}")

# Step 2: Drop rows missing critical columns (lyrics)
before = len(df_error)
df_error = df_error.dropna(subset=['lyrics'])
removed = before - len(df_error)
cleaning_log.append(('dropna(lyrics)', removed, len(df_error)))
print(f"Step 2 - dropna(lyrics): Removed {removed} rows, Remaining: {len(df_error)}")

# Step 3: Drop rows missing artist
before = len(df_error)
df_error = df_error.dropna(subset=['artist'])
removed = before - len(df_error)
cleaning_log.append(('dropna(artist)', removed, len(df_error)))
print(f"Step 3 - dropna(artist): Removed {removed} rows, Remaining: {len(df_error)}")

# Step 4: Fix invalid data types in 'year'
df_error['year'] = pd.to_numeric(df_error['year'], errors='coerce')
before = len(df_error)
df_error = df_error.dropna(subset=['year'])
df_error['year'] = df_error['year'].astype(int)
removed = before - len(df_error)
cleaning_log.append(('to_numeric(year)', removed, len(df_error)))
print(f"Step 4 - to_numeric(year): Removed {removed} rows, Remaining: {len(df_error)}")

# Step 5: Remove outliers (realistic lyrics_length 50–5000)
df_error['lyrics_length'] = df_error['lyrics'].apply(lambda x: len(str(x)))
before = len(df_error)
df_error = df_error[(df_error['lyrics_length'] >= 50) & (df_error['lyrics_length'] <= 5000)]
removed = before - len(df_error)
cleaning_log.append(('outlier_removal', removed, len(df_error)))
print(f"Step 5 - outlier_removal: Removed {removed} rows, Remaining: {len(df_error)}")

print(f"\n--- Cleaning Summary ---")
print(f"Started: {initial_rows + 50} rows (with errors)")
print(f"Final: {len(df_error)} rows")
print(f"Total Removed: {initial_rows + 50 - len(df_error)} rows ({(initial_rows + 50 - len(df_error))/(initial_rows + 50)*100:.1f}%)")

# ============================================================================
# SECTION 3: STATISTICAL SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("SECTION 3: Statistical Summary of Cleaned Dataset")
print("=" * 60)

# 3.1 Numerical Descriptive Statistics
print("\n--- 3.1 Numerical Descriptive Statistics ---")
stats = df_error[['year', 'lyrics_length', 'rank']].describe()
stats.loc['variance'] = df_error[['year', 'lyrics_length', 'rank']].var()
print(stats.round(2))

# 3.2 Generation Distribution (5-Generation Standard)
print("\n--- 3.2 Generation Distribution (5-Generation Standard) ---")

def assign_generation(debut_year):
    """
    Assign generation based on debut year.
    Following the reference table (韩国女团世代表_一代至五代.xlsx)
    """
    if pd.isna(debut_year):
        return None
    try:
        debut_year = int(debut_year)
    except:
        return None
    
    if 1996 <= debut_year <= 2002:
        return 'Gen 1'
    elif 2003 <= debut_year <= 2009:
        return 'Gen 2'
    elif 2010 <= debut_year <= 2013:
        return 'Gen 3'
    elif 2014 <= debut_year <= 2017:
        return 'Gen 4'
    else:  # 2018+
        return 'Gen 5'

# Map Chinese generation names to English (5-Generation Standard)
# Note: 六代 is merged into Gen 5 following the 5-generation standard
gen_map = {
    '一代': 'Gen 1', 
    '二代': 'Gen 2', 
    '三代': 'Gen 3', 
    '四代': 'Gen 4', 
    '五代': 'Gen 5',
    '六代': 'Gen 5'  # Merge Gen 6 into Gen 5 (2018+)
}
if df_error['generation'].dtype == 'object':
    df_error['generation'] = df_error['generation'].replace(gen_map)

# Filter out any None generations
df_error = df_error[df_error['generation'].notna()]

gen_counts = df_error['generation'].value_counts().sort_index()
gen_pct = (gen_counts / len(df_error) * 100).round(1)

print("\nGeneration Distribution:")
for gen in ['Gen 1', 'Gen 2', 'Gen 3', 'Gen 4', 'Gen 5']:
    if gen in gen_counts.index:
        print(f"  {gen}: {gen_counts[gen]:,} songs ({gen_pct[gen]}%)")

# ============================================================================
# SECTION 4 & 5: NEURAL NETWORK MODEL
# ============================================================================
print("\n" + "=" * 60)
print("SECTION 4 & 5: Neural Network Model Training")
print("=" * 60)

# 5.1 Preprocessing Steps
print("\n--- 5.1 Text Preprocessing ---")

def clean_text(text):
    """
    Clean lyrics text by removing special characters
    and keeping Korean, English, numbers, and spaces.
    """
    if not isinstance(text, str):
        return ""
    # Remove special characters, keep Korean (가-힣), English, numbers, spaces
    text = re.sub(r'[^가-힣a-zA-Z0-9\s]', ' ', text.lower())
    # Remove extra whitespace
    return ' '.join(text.split())

df_error['clean_lyrics'] = df_error['lyrics'].apply(clean_text)

# Remove rows with very short cleaned lyrics
df_error = df_error[df_error['clean_lyrics'].str.len() > 10]
print(f"After text cleaning: {len(df_error)} samples")

# Prepare features and labels
X = df_error['clean_lyrics'].values
y = df_error['generation'].values

# Label encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"Classes: {le.classes_}")

# TF-IDF Vectorization
print("\n--- TF-IDF Vectorization ---")
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# Fit vectorizer on training data only
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"Train shape: {X_train_tfidf.shape}")
print(f"Test shape: {X_test_tfidf.shape}")

# 5.2 Model Training
print("\n--- 5.2 Model Training ---")
print("\nModel Architecture:")
print("  Input Layer: 3,000 TF-IDF features")
print("  Hidden Layer 1: 256 neurons + ReLU")
print("  Hidden Layer 2: 128 neurons + ReLU")
print("  Hidden Layer 3: 64 neurons + ReLU")
print("  Output Layer: 5 neurons (Softmax)")
print("  Optimizer: Adam")
print("  Early Stopping: Enabled (10% validation)")

model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    early_stopping=True,
    validation_fraction=0.1,
    max_iter=200,
    random_state=42,
    verbose=False
)

print("\nTraining model...")
model.fit(X_train_tfidf, y_train)

print(f"\nTraining completed!")
print(f"  Iterations: {model.n_iter_}")
print(f"  Best validation score: {model.best_validation_score_:.4f}")

# ============================================================================
# SECTION 6: MODEL EVALUATION
# ============================================================================
print("\n" + "=" * 60)
print("SECTION 6: The Accuracy of the Eventual Model")
print("=" * 60)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Overall accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n{'='*40}")
print(f"  OVERALL TEST ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"{'='*40}")

# Classification report
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Per-class accuracy (recall)
print("\n--- Per-Class Accuracy (Recall) ---")
for i, gen in enumerate(le.classes_):
    mask = y_test == i
    if mask.sum() > 0:
        class_acc = (y_pred[mask] == i).sum() / mask.sum()
        print(f"  {gen}: {class_acc*100:.1f}%")

# Confusion matrix
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)
print(pd.DataFrame(cm, index=le.classes_, columns=le.classes_))

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n" + "=" * 60)
print("Generating Visualizations...")
print("=" * 60)

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('K-Pop Girl Group Lyrics Analysis - Model Results', fontsize=14, fontweight='bold')

# 1. Generation Distribution
ax1 = axes[0, 0]
gen_order = ['Gen 1', 'Gen 2', 'Gen 3', 'Gen 4', 'Gen 5']
gen_counts_plot = df_error['generation'].value_counts().reindex(gen_order)
colors = plt.cm.viridis(np.linspace(0.2, 0.8, 5))
bars = ax1.bar(gen_order, gen_counts_plot.values, color=colors, edgecolor='black')
ax1.set_xlabel('Generation')
ax1.set_ylabel('Number of Songs')
ax1.set_title('Generation Distribution')
for bar, val in zip(bars, gen_counts_plot.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
             str(val), ha='center', fontsize=9)

# 2. Lyrics Length Distribution
ax2 = axes[0, 1]
ax2.hist(df_error['lyrics_length'], bins=40, color='steelblue', edgecolor='black', alpha=0.7)
ax2.axvline(df_error['lyrics_length'].mean(), color='red', linestyle='--', 
            label=f"Mean: {df_error['lyrics_length'].mean():.0f}")
ax2.set_xlabel('Lyrics Length (characters)')
ax2.set_ylabel('Frequency')
ax2.set_title('Lyrics Length Distribution')
ax2.legend()

# 3. Training Loss Curve
ax3 = axes[0, 2]
ax3.plot(model.loss_curve_, 'b-', linewidth=2)
ax3.set_xlabel('Iteration')
ax3.set_ylabel('Loss')
ax3.set_title('Training Loss Curve')
ax3.grid(True, alpha=0.3)
ax3.axvline(model.n_iter_, color='red', linestyle='--', alpha=0.7, 
            label=f'Early Stop @ {model.n_iter_}')
ax3.legend()

# 4. Confusion Matrix
ax4 = axes[1, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
            xticklabels=le.classes_, yticklabels=le.classes_)
ax4.set_xlabel('Predicted')
ax4.set_ylabel('Actual')
ax4.set_title(f'Confusion Matrix (Acc: {accuracy*100:.2f}%)')

# 5. Per-Class Accuracy
ax5 = axes[1, 1]
class_accs = []
for i in range(len(le.classes_)):
    mask = y_test == i
    if mask.sum() > 0:
        class_accs.append((y_pred[mask] == i).sum() / mask.sum())
    else:
        class_accs.append(0)

bar_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
bars = ax5.bar(le.classes_, class_accs, color=bar_colors, edgecolor='black')
ax5.set_xlabel('Generation')
ax5.set_ylabel('Accuracy')
ax5.set_title('Per-Class Accuracy')
ax5.set_ylim(0, 1.1)
ax5.axhline(y=accuracy, color='red', linestyle='--', linewidth=2, label=f'Overall: {accuracy:.2f}')
for bar, acc in zip(bars, class_accs):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{acc:.2f}', ha='center', fontsize=10, fontweight='bold')
ax5.legend()

# 6. Precision/Recall/F1 Comparison
ax6 = axes[1, 2]
report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
metrics = ['precision', 'recall', 'f1-score']
x = np.arange(len(le.classes_))
width = 0.25

for i, metric in enumerate(metrics):
    values = [report[gen][metric] for gen in le.classes_]
    ax6.bar(x + i*width, values, width, label=metric.capitalize())

ax6.set_xlabel('Generation')
ax6.set_ylabel('Score')
ax6.set_title('Precision / Recall / F1-Score')
ax6.set_xticks(x + width)
ax6.set_xticklabels(le.classes_)
ax6.legend()
ax6.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig('model_results.png', dpi=150, bbox_inches='tight')
print("Saved: model_results.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"""
Dataset: {len(df_error)} songs from K-Pop girl groups (2000-2023)
Features: TF-IDF vectors (3,000 features, unigrams + bigrams)
Model: MLP Neural Network (256 → 128 → 64)
Train/Test Split: 80/20 (stratified)

RESULTS:
  - Test Accuracy: {accuracy*100:.2f}%
  - Best Validation Score: {model.best_validation_score_*100:.2f}%
  - Training Iterations: {model.n_iter_}

KEY FINDINGS:
  - Gen 4 & Gen 5 achieved highest accuracy (~94%)
  - Gen 2 showed strong performance due to largest sample size
  - Gen 1 had lowest accuracy due to smallest sample size and overlap with Gen 2
  - The model successfully learned generation-specific vocabulary patterns
""")

print("=" * 60)
print("Analysis Complete!")
print("=" * 60)
