
import pandas as pd
import numpy as np
import librosa
import os
import re
import subprocess
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

import shutil
import sys

def ensure_ffmpeg():
    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg is not installed or not in PATH.")
        print("Install it and try again.")
        sys.exit(1)

    print("ffmpeg found:", shutil.which("ffmpeg"))

ensure_ffmpeg()

# =========================
# SETTINGS
# =========================

INPUT_EXCEL = "songs.xlsx"
OUTPUT_EXCEL = "cdjn cienvevocvojk.xlsx"
MISSING_EXCEL = "missing_files_v1.xlsx"

SONG_FOLDER = "songs_folder"

START_ID = 1
END_ID = 1800

TARGET_SR = 22050
MAX_DURATION = 60   # seconds of audio to analyse


# =========================
# CLEAN FILENAMES
# =========================

def clean_name(name):

    name = str(name).lower()
    name = name.replace(".mp3","")

    name = re.sub(r'[^a-z0-9]', '', name)

    return name


# =========================
# BUILD FILE LOOKUP
# =========================

print("\nBuilding song lookup...")

file_lookup = {}

for f in os.listdir(SONG_FOLDER):

    if f.lower().endswith(".mp3"):

        key = clean_name(f)

        file_lookup[key] = f

print("MP3 files detected:", len(file_lookup))


# =========================
# FFMPEG AUDIO LOADER
# =========================

def load_audio_ffmpeg(path):

    command = [
        "ffmpeg",
        "-i", path,
        "-t", str(MAX_DURATION),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "1",
        "-ar", str(TARGET_SR),
        "-"
    ]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    try:
        out, _ = process.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        raise ValueError("ffmpeg_timeout")

    audio = np.frombuffer(out, np.float32)

    if len(audio) < 1000:
        raise ValueError("audio_decode_failed")

    return audio, TARGET_SR


# =========================
# FEATURE EXTRACTION
# =========================

def extract_features(file_path):

    y, sr = load_audio_ffmpeg(file_path)

    features = {}

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    for i in range(13):

        features[f"mfcc_{i+1}_mean"] = np.mean(mfcc[i])
        features[f"mfcc_{i+1}_std"] = np.std(mfcc[i])


    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    for i in range(12):

        features[f"chroma_{i+1}_mean"] = np.mean(chroma[i])


    # Spectral
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    features["spectral_centroid"] = np.mean(centroid)
    features["spectral_bandwidth"] = np.mean(bandwidth)
    features["spectral_rolloff"] = np.mean(rolloff)


    # Rhythm
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

    features["tempo"] = tempo


    # Energy
    rms = librosa.feature.rms(y=y)

    features["rms_mean"] = np.mean(rms)
    features["rms_std"] = np.std(rms)

    zcr = librosa.feature.zero_crossing_rate(y)

    features["zcr_mean"] = np.mean(zcr)


    return features


# =========================
# LOAD DATASET
# =========================

print("\nLoading dataset...")

df = pd.read_excel(INPUT_EXCEL)

df.columns = df.columns.str.strip()

print("Total rows:", len(df))

df_batch = df[(df["ID"] >= START_ID) & (df["ID"] <= END_ID)]

print("Rows in selected ID range:", len(df_batch))


# =========================
# REMOVE ALREADY PROCESSED
# =========================

if os.path.exists(OUTPUT_EXCEL):

    existing_df = pd.read_excel(OUTPUT_EXCEL)

    existing_df["File name"] = existing_df["File name"].astype(str).str.lower().str.strip()
    df_batch["File name"] = df_batch["File name"].astype(str).str.lower().str.strip()

    processed = set(existing_df["File name"])

    before = len(df_batch)

    df_batch = df_batch[~df_batch["File name"].isin(processed)]

    print("Rows already processed:", before - len(df_batch))

else:

    existing_df = pd.DataFrame()

print("Rows remaining to process:", len(df_batch))


# =========================
# PROCESS SONGS
# =========================

new_rows = []
missing_rows = []

for _, row in tqdm(df_batch.iterrows(), total=len(df_batch)):

    original_name = str(row["File name"]).strip()

    cleaned = clean_name(original_name)

    if cleaned not in file_lookup:

        missing_rows.append({
            "Song Title": row["Song Title"],
            "File name": original_name,
            "Reason": "file_not_found"
        })

        continue


    actual_file = file_lookup[cleaned]

    file_path = os.path.join(SONG_FOLDER, actual_file)

    try:

        features = extract_features(file_path)

        row_data = row.to_dict()

        row_data.update(features)

        new_rows.append(row_data)

    except Exception as e:

        missing_rows.append({
            "Song Title": row["Song Title"],
            "File name": actual_file,
            "Reason": str(e)
        })


# =========================
# SAVE FEATURES
# =========================

if new_rows:

    print("\nSaving features...")

    new_df = pd.DataFrame(new_rows)

    if os.path.exists(OUTPUT_EXCEL):

        final_df = pd.concat([existing_df, new_df], ignore_index=True)

    else:

        final_df = new_df

    final_df.drop_duplicates(subset=["File name"], inplace=True)

    final_df.to_excel(OUTPUT_EXCEL, index=False)


# =========================
# SAVE MISSING FILES
# =========================

if missing_rows:

    print("\nSaving missing files...")

    missing_df = pd.DataFrame(missing_rows)

    if os.path.exists(MISSING_EXCEL):

        old_missing = pd.read_excel(MISSING_EXCEL)

        missing_df = pd.concat([old_missing, missing_df], ignore_index=True)

    missing_df.drop_duplicates(subset=["File name"], inplace=True)

    missing_df.to_excel(MISSING_EXCEL, index=False)


print("\nBatch finished successfully.")
