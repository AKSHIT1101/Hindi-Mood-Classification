"""
Hindi Song Mood Classifier - v2
=================================
Improvements over v1:
  1. OpenSMILE      -> eGeMAPS emotion features (valence/arousal proxies)
  2. Vosk STT       -> Hindi transcript
  3. XLM-RoBERTa   -> multilingual sentiment scores (neg / neu / pos)
  4. LaBSE          -> 768-dim sentence embedding (full semantic meaning)
  5. Middle-30s     -> features from center of song only
  6. Feature selection -> drop low-variance + highly correlated features

SETUP (run once):
----------------------------------
pip install opensmile vosk pydub librosa soundfile pandas scikit-learn \
            joblib tqdm transformers torch sentence-transformers

Download Vosk Hindi model (~40MB):
  https://alphacephei.com/vosk/models -> vosk-model-small-hi-0.22
  Unzip it, set VOSK_MODEL_PATH below.

HuggingFace models download automatically on first run:
  cardiffnlp/twitter-xlm-roberta-base-sentiment  (~300MB)
  sentence-transformers/LaBSE                     (~700MB)

FOLDER STRUCTURE:
  audio/                                   <- all .mp3 / .wav files
  3_moods_with_final_mood_edited.xlsx      <- labelled dataset
  vosk-model-small-hi-0.22/               <- unzipped vosk model
"""

import tempfile
import os, json, warnings
import numpy as np
import pandas as pd
import joblib, librosa
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── CONFIG ───────────────────────────────────────────────────────────────────
AUDIO_DIR       = "songs_folder"
EXCEL_PATH      = "3 moods with final mood_edited.xlsx"
VOSK_MODEL_PATH = "vosk-model-small-hi-0.22"
CACHE_DIR       = "feature_cache"
OUTPUT_DIR      = "output_v2"
MIDDLE_SECONDS  = 30
TARGET_COL      = "FINAL MOOD"
RANDOM_STATE    = 42
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# SECTION 1 - NLP MODELS
# =============================================================================

_sentiment_pipe = None
_labse_model    = None

_vosk_model = None

def get_vosk_model(path):
    global _vosk_model
    if _vosk_model is None:
        from vosk import Model
        print("  Loading Vosk model (once)...")
        _vosk_model = Model(path)
    return _vosk_model


def get_sentiment_pipe():
    """
    XLM-RoBERTa multilingual sentiment model.
    Trained on Twitter data across 8 languages, generalises well to Hindi.
    Returns 3 scores: negative / neutral / positive (always sum to 1.0).
    These are real sentiment numbers, not class labels.
    """
    global _sentiment_pipe
    if _sentiment_pipe is None:
        from transformers import pipeline
        print("  Loading XLM-RoBERTa sentiment model (~300MB on first run)...")
        _sentiment_pipe = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
            tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment",
            top_k=None,        # return ALL label scores, not just the top one
            truncation=True,
            max_length=512,
        )
    return _sentiment_pipe


def get_labse():
    """
    LaBSE: Language-agnostic BERT Sentence Embeddings.
    Supports 109 languages including Hindi and Romanised Hindi.
    Encodes the full meaning of a transcript into a 768-dim vector.
    Words like 'ishq', 'dard', 'khushi' each pull the vector in
    different directions that the classifier can learn from.
    """
    global _labse_model
    if _labse_model is None:
        from sentence_transformers import SentenceTransformer
        print("  Loading LaBSE model (~700MB on first run)...")
        _labse_model = SentenceTransformer("sentence-transformers/LaBSE")
    return _labse_model


def extract_nlp_features(transcript: str) -> dict:
    """
    From a raw Hindi transcript string extract:
      nlp_sent_negative  (float 0-1)
      nlp_sent_neutral   (float 0-1)
      nlp_sent_positive  (float 0-1)
      nlp_emb_0 ... nlp_emb_767  (768 floats)

    Total: 771 NLP features per song.
    Empty transcripts get zeroed-out neutral values.
    """
    feats = {}
    empty = not transcript or not transcript.strip()

    # --- Sentiment scores (3 features) -------------------------------------
    if empty:
        feats["nlp_sent_negative"] = 0.0
        feats["nlp_sent_neutral"]  = 1.0
        feats["nlp_sent_positive"] = 0.0
    else:
        try:
            pipe    = get_sentiment_pipe()
            results = pipe(transcript)[0]   # list of {label, score}
            smap    = {r["label"].lower(): r["score"] for r in results}
            feats["nlp_sent_negative"] = float(smap.get("negative", 0.0))
            feats["nlp_sent_neutral"]  = float(smap.get("neutral",  0.0))
            feats["nlp_sent_positive"] = float(smap.get("positive", 0.0))
        except Exception as e:
            print(f"    Sentiment error: {e}")
            feats["nlp_sent_negative"] = 0.0
            feats["nlp_sent_neutral"]  = 1.0
            feats["nlp_sent_positive"] = 0.0

    # --- LaBSE sentence embedding (768 features) ---------------------------
    if empty:
        for i in range(768):
            feats[f"nlp_emb_{i}"] = 0.0
    else:
        try:
            model = get_labse()
            vec   = model.encode(transcript, normalize_embeddings=True)
            for i, v in enumerate(vec):
                feats[f"nlp_emb_{i}"] = float(v)
        except Exception as e:
            print(f"    LaBSE error: {e}")
            for i in range(768):
                feats[f"nlp_emb_{i}"] = 0.0

    return feats


# =============================================================================
# SECTION 2 - AUDIO FEATURE EXTRACTION
# =============================================================================

def load_middle_segment(path: str, duration: int = MIDDLE_SECONDS, sr: int = 22050):
    """Load `duration` seconds centred on the song midpoint."""
    try:
        total = librosa.get_duration(path=path)
        if total == 0:
            raise ValueError("Empty audio")

        start = max(0.0, (total / 2) - (duration / 2))

        y, _ = librosa.load(
            path,
            sr=sr,
            offset=start,
            duration=duration,
            mono=True
        )

        if y is None or len(y) == 0:
            raise ValueError("Loaded empty waveform")

        return y, sr

    except Exception as e:
        raise RuntimeError(f"Audio load failed: {e}")


def extract_librosa_features(y, sr) -> dict:
    """51 librosa features extracted from the middle segment."""
    feats = {}

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        feats[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc_{i+1}_std"]  = float(np.std(mfcc[i]))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(12):
        feats[f"chroma_{i+1}_mean"] = float(np.mean(chroma[i]))

    feats["spectral_centroid"]      = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    feats["spectral_bandwidth"]     = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    feats["spectral_rolloff"]       = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    sc = librosa.feature.spectral_contrast(y=y, sr=sr)
    feats["spectral_contrast_mean"] = float(np.mean(sc))
    feats["spectral_contrast_std"]  = float(np.std(sc))
    feats["spectral_flatness"]      = float(np.mean(librosa.feature.spectral_flatness(y=y)))

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, np.ndarray):
        tempo = tempo.item() if tempo.size == 1 else np.mean(tempo)
    feats["tempo"]         = float(tempo)
    feats["onset_rate"]    = float(np.mean(librosa.onset.onset_strength(y=y, sr=sr)))
    feats["beat_strength"] = float(len(beats) / (len(y) / sr)) if len(beats) > 0 else 0.0

    rms = librosa.feature.rms(y=y)[0]
    feats["rms_mean"]      = float(np.mean(rms))
    feats["rms_std"]       = float(np.std(rms))
    feats["zcr_mean"]      = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
    feats["dynamic_range"] = float(np.max(rms) - np.min(rms))

    return feats


def extract_opensmile_features(path: str) -> dict:
    """
    eGeMAPS v02 via opensmile: ~88 features including F0 contours,
    loudness, shimmer, jitter, HNR - direct emotion/arousal proxies.
    """
    import opensmile
    smile  = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    result = smile.process_file(path)
    return {f"osm_{k}": float(v) for k, v in result.iloc[0].to_dict().items()}


def vosk_transcribe(path: str, model_path: str) -> str:
    """Convert audio to 16kHz mono WAV then run Vosk Hindi STT."""
    import wave
    from vosk import Model, KaldiRecognizer
    from pydub import AudioSegment

    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    AudioSegment.from_file(path).set_channels(1).set_frame_rate(16000).export(tmp, format="wav")

    model = get_vosk_model(model_path)
    rec   = KaldiRecognizer(model, 16000)
    parts = []
    with wave.open(tmp, "rb") as wf:
        while True:
            data = wf.readframes(4000)
            if not data:
                break
            if rec.AcceptWaveform(data):
                parts.append(json.loads(rec.Result()).get("text", ""))
    parts.append(json.loads(rec.FinalResult()).get("text", ""))
    return " ".join(parts).strip()


# =============================================================================
# SECTION 3 - FEATURE SELECTION
# =============================================================================

def drop_low_variance(df: pd.DataFrame, threshold: float = 0.01) -> list:
    from sklearn.preprocessing import MinMaxScaler
    scaled = pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns)
    return [c for c in scaled.columns if scaled[c].var() > threshold]


def drop_correlated(df: pd.DataFrame, threshold: float = 0.90) -> list:
    corr  = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop  = {c for c in upper.columns if any(upper[c] > threshold)}
    return [c for c in df.columns if c not in drop]


def select_features(X_df: pd.DataFrame) -> pd.DataFrame:
    # NOTE: skip correlation filter on embedding dims (768 dims x 768 dims
    # is a 768x768 matrix - slow and unnecessary since LaBSE dims are
    # already decorrelated by design). Only apply to audio features.
    audio_cols = [c for c in X_df.columns if not c.startswith("nlp_emb_")]
    nlp_cols   = [c for c in X_df.columns if c.startswith("nlp_emb_")]

    print(f"Audio features before selection : {len(audio_cols)}")
    audio_df = X_df[audio_cols]
    keep     = drop_low_variance(audio_df)
    audio_df = audio_df[keep]
    print(f"After low-variance filter       : {len(audio_df.columns)}")
    keep     = drop_correlated(audio_df)
    audio_df = audio_df[keep]
    print(f"After correlation filter        : {len(audio_df.columns)}")
    print(f"NLP embedding dims kept as-is   : {len(nlp_cols)}")
    print(f"Total features                  : {len(audio_df.columns) + len(nlp_cols)}")

    return pd.concat([audio_df, X_df[nlp_cols]], axis=1)


# =============================================================================
# SECTION 4 - MAIN FEATURE EXTRACTION LOOP
# =============================================================================

def build_feature_matrix(df_meta: pd.DataFrame) -> pd.DataFrame:
    audio_dir = Path(AUDIO_DIR)

    if not audio_dir.exists():
        print(f"\n[ERROR] Audio folder not found: {audio_dir.resolve()}")
        print("  Set AUDIO_DIR = 'songs_folder'  (or full path if needed)")
        raise SystemExit(1)

    # Index all audio files on disk by their stem (filename without extension),
    # lowercased for case-insensitive matching
    audio_files = {
        p.stem.lower(): p
        for p in audio_dir.glob("*")
        if p.suffix.lower() in (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".webm", ".opus")
    }

    print(f"Audio folder     : {audio_dir.resolve()}")
    print(f"Audio files found: {len(audio_files)}")

    if len(audio_files) == 0:
        print("[ERROR] No audio files found. Check AUDIO_DIR and file extensions.")
        raise SystemExit(1)

    # Quick sanity check on first 3 rows
    print("\nMatching check (first 3 rows):")
    for fn in df_meta["File name"].dropna().head(3):
        fn_str = str(fn).strip()
        match  = audio_files.get(fn_str.lower())
        print(f"  {'FOUND' if match else 'MISSING'} <- '{fn_str}'")
    print()

    rows, skipped = [], 0

    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc="Extracting features"):
        song_id   = str(row["ID"])
        file_name = str(row["File name"]).strip()
        target    = row[TARGET_COL]

        # Direct match: File name column value IS the filename (without extension)
        audio_path = audio_files.get(file_name.lower())

        if audio_path is None:
            skipped += 1
            continue

        cache_file = Path(CACHE_DIR) / f"{song_id}.json"

        if cache_file.exists():
            with open(cache_file) as f:
                feats = json.load(f)
        else:
            feats = {"song_id": song_id, "target": target}

            # Librosa middle-30s features
            try:
                y, sr = load_middle_segment(str(audio_path))
                feats.update(extract_librosa_features(y, sr))
            except Exception as e:
                print(f"  Librosa error [{file_name}]: {e}")

            # OpenSMILE eGeMAPS features
            try:
                feats.update(extract_opensmile_features(str(audio_path)))
            except Exception as e:
                print(f"  OpenSMILE error [{file_name}]: {e}")

            # Vosk STT -> raw Hindi transcript string
            transcript = ""
            try:
                transcript = vosk_transcribe(str(audio_path), VOSK_MODEL_PATH)
                feats["transcript"] = transcript
            except Exception as e:
                print(f"  Vosk error [{file_name}]: {e}")
                feats["transcript"] = ""

            # XLM-RoBERTa sentiment scores + LaBSE embedding
            try:
                feats.update(extract_nlp_features(transcript))
            except Exception as e:
                print(f"  NLP error [{file_name}]: {e}")

            with open(cache_file, "w") as f:
                json.dump(feats, f)

        rows.append(feats)

    print(f"\nProcessed {len(rows)} songs, skipped {skipped} (audio not found).")
    return pd.DataFrame(rows)


# =============================================================================
# SECTION 5 - TRAINING & EVALUATION
# =============================================================================

def train_and_evaluate(X, y, feature_names, label_encoder):
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    models = {
        "Random Forest": Pipeline([
            ("sc", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                           random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("sc", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                               max_depth=5, random_state=RANDOM_STATE)),
        ]),
        "SVM (RBF)": Pipeline([
            ("sc", StandardScaler()),
            ("clf", SVC(C=10, gamma="scale", class_weight="balanced",
                        probability=True, random_state=RANDOM_STATE)),
        ]),
        "Logistic Regression": Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                                       random_state=RANDOM_STATE)),
        ]),
    }

    print("\n" + "="*60)
    print("5-FOLD CROSS-VALIDATION")
    print("="*60)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train,
                                 cv=cv, scoring="f1_macro", n_jobs=-1)
        print(f"  {name:<25}: {scores.mean():.4f} +/- {scores.std():.4f}")

    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    results, trained = {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc  = accuracy_score(y_test, pred)
        f1   = f1_score(y_test, pred, average="macro")
        results[name] = {"accuracy": acc, "f1_macro": f1, "pred": pred}
        trained[name] = model
        print(f"  {name:<25}: Acc={acc:.4f}  F1={f1:.4f}")

    ensemble = VotingClassifier(
        estimators=[(n.replace(" ", "_"), m) for n, m in trained.items()],
        voting="soft", n_jobs=-1
    )
    ensemble.fit(X_train, y_train)
    ens_pred = ensemble.predict(X_test)
    ens_acc  = accuracy_score(y_test, ens_pred)
    ens_f1   = f1_score(y_test, ens_pred, average="macro")
    print(f"  {'Ensemble':<25}: Acc={ens_acc:.4f}  F1={ens_f1:.4f}")

    # Pick best overall
    best_name, best_pred, best_f1 = "Ensemble", ens_pred, ens_f1
    for name, r in results.items():
        if r["f1_macro"] > best_f1:
            best_name, best_pred, best_f1 = name, r["pred"], r["f1_macro"]

    final_model = ensemble if best_name == "Ensemble" else trained[best_name]

    print(f"\nBest model: {best_name}  (F1-macro={best_f1:.4f})")
    print("\n" + "="*60)
    print(f"CLASSIFICATION REPORT - {best_name}")
    print("="*60)
    print(classification_report(y_test, best_pred, target_names=label_encoder.classes_))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, best_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, fmt, title in zip(
            axes,
            [cm, cm.astype(float) / cm.sum(axis=1, keepdims=True)],
            ["d", ".2f"], ["Counts", "Normalised"]):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_, ax=ax)
        ax.set_title(f"Confusion Matrix ({title})")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.suptitle(f"Best Model: {best_name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Model comparison bar chart
    names_all = list(results.keys()) + ["Ensemble"]
    f1_all    = [results[n]["f1_macro"] for n in results] + [ens_f1]
    acc_all   = [results[n]["accuracy"]  for n in results] + [ens_acc]
    x = np.arange(len(names_all))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.2, acc_all, 0.35, label="Accuracy", color="steelblue")
    ax.bar(x + 0.2, f1_all,  0.35, label="F1-macro",  color="coral")
    ax.set_xticks(x); ax.set_xticklabels(names_all, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1); ax.set_ylabel("Score")
    ax.set_title("Model Comparison - v2 (OpenSMILE + Vosk + XLM-RoBERTa + LaBSE)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Feature importance - audio/NLP meta features only (skip embedding dims)
    rf  = trained["Random Forest"]
    imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=feature_names)
    imp_non_emb = imp[[c for c in imp.index if not c.startswith("nlp_emb_")]].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(13, 7))
    imp_non_emb.head(25).plot(kind="bar", color="teal", ax=ax)
    ax.set_title("Top 25 Feature Importances - Audio + Sentiment (Random Forest)")
    ax.set_ylabel("Importance")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_importance_audio.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Separate chart: top 25 embedding dims by importance
    imp_emb = imp[[c for c in imp.index if c.startswith("nlp_emb_")]].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(13, 5))
    imp_emb.head(25).plot(kind="bar", color="mediumpurple", ax=ax)
    ax.set_title("Top 25 LaBSE Embedding Dims by Importance (Random Forest)")
    ax.set_ylabel("Importance")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_importance_embeddings.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nCharts saved to {OUTPUT_DIR}/")

    joblib.dump({
        "model": final_model,
        "label_encoder": label_encoder,
        "feature_names": feature_names,
    }, f"{OUTPUT_DIR}/mood_classifier_v2.pkl")
    print(f"Model saved to {OUTPUT_DIR}/mood_classifier_v2.pkl")

    return final_model


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    from sklearn.preprocessing import LabelEncoder

    print("="*60)
    print("HINDI MOOD CLASSIFIER - v2")
    print("="*60)

    df_meta = pd.read_excel(EXCEL_PATH)
    print(f"Loaded {len(df_meta)} songs from Excel.")

    # Step 1: extract all features (each song cached to JSON)
    df_feats = build_feature_matrix(df_meta)

    # Step 2: ensure target column present
    if TARGET_COL not in df_feats.columns:
        id_to_target = df_meta.set_index("ID")[TARGET_COL].to_dict()
        df_feats[TARGET_COL] = df_feats["song_id"].astype(int).map(id_to_target)
    df_feats = df_feats.dropna(subset=[TARGET_COL])

    # Step 3: isolate feature columns
    NON_FEATURE = {
        "song_id", "target", TARGET_COL, "transcript",
        "ID", "Song Title", "Film / Album", "Year",
        "ChatGPT Mood", "Claude Mood", "deepseek/copilot Mood",
        "Test", "File name"
    }
    feat_cols = [c for c in df_feats.columns if c not in NON_FEATURE]
    X_df = df_feats[feat_cols].copy()

    # Step 4: feature selection (audio only; embeddings kept as-is)
    print("\n" + "="*60)
    print("FEATURE SELECTION")
    print("="*60)
    X_df = select_features(X_df)

    # Step 5: encode labels
    le = LabelEncoder()
    y  = le.fit_transform(df_feats[TARGET_COL].values)
    print(f"\nClasses : {list(le.classes_)}")
    print(f"Samples : {len(y)}")

    # Step 6: train and evaluate
    train_and_evaluate(X_df.values, y, X_df.columns.tolist(), le)

    print("\nPipeline complete.")