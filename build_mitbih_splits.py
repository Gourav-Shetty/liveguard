import os
import glob
import re
import numpy as np
import pandas as pd

RAW_DIR = r"C:\LiveGuard\archive\mitbih_database"
OUT_DIR = r"C:\LiveGuard\data"
os.makedirs(OUT_DIR, exist_ok=True)

# 48 MIT-BIH Patient Records
# Allocate 38 to Train, 5 to Val, 5 to Test (Standard inter-patient split)
TEST_PATIENTS = {"100", "119", "207", "214", "222"}
VAL_PATIENTS = {"105", "111", "200", "213", "231"}

# Standard AAMI heartbeat mapping
AAMI_MAPPING = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,       # Normal (N)
    'A': 1, 'a': 1, 'J': 1, 'S': 1,               # Supraventricular Ectopic (S)
    'V': 2, 'E': 2,                               # Ventricular Ectopic (V)
    'F': 3,                                       # Fusion (F)
    '/': 4, 'f': 4, 'Q': 4                        # Paced / Unknown (Q)
}

BEAT_WINDOW = 180
PRE_R = 70
POST_R = 110

def parse_annotations(annot_path):
    beats = []
    with open(annot_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    sample_idx = int(parts[1])
                    beat_type = parts[2]
                    if beat_type in AAMI_MAPPING:
                        beats.append((sample_idx, AAMI_MAPPING[beat_type]))
                except ValueError:
                    continue
    return beats

def process_record(csv_path, annot_path):
    record_id = os.path.basename(csv_path).replace(".csv", "")
    df = pd.read_csv(csv_path)
    
    # Select lead II (col 1: MLII)
    ecg_signal = df.iloc[:, 1].values.astype(np.float32)
    beats = parse_annotations(annot_path)

    X_list, y_list, id_list = [], [], []
    sig_len = len(ecg_signal)

    for r_peak, label in beats:
        start = r_peak - PRE_R
        end = r_peak + POST_R
        if start >= 0 and end <= sig_len:
            segment = ecg_signal[start:end]
            std = np.std(segment)
            if std > 1e-6:
                segment = (segment - np.mean(segment)) / std
            else:
                segment = segment - np.mean(segment)

            X_list.append(segment.reshape(BEAT_WINDOW, 1))
            y_list.append(label)
            id_list.append(record_id)

    return X_list, y_list, id_list

def main():
    print(f"Reading raw MIT-BIH patient files from: {RAW_DIR}")
    csv_files = sorted(glob.glob(os.path.join(RAW_DIR, "[0-9][0-9][0-9].csv")))

    splits = {
        "train": {"X": [], "y": [], "record_id": []},
        "val": {"X": [], "y": [], "record_id": []},
        "test": {"X": [], "y": [], "record_id": []}
    }

    patient_counts = {"train": 0, "val": 0, "test": 0}

    for csv_file in csv_files:
        rec_id = os.path.basename(csv_file).replace(".csv", "")
        annot_file = os.path.join(RAW_DIR, f"{rec_id}annotations.txt")
        if not os.path.exists(annot_file):
            continue

        if rec_id in TEST_PATIENTS:
            split_name = "test"
        elif rec_id in VAL_PATIENTS:
            split_name = "val"
        else:
            split_name = "train"

        patient_counts[split_name] += 1
        X_rec, y_rec, id_rec = process_record(csv_file, annot_file)
        splits[split_name]["X"].extend(X_rec)
        splits[split_name]["y"].extend(y_rec)
        splits[split_name]["record_id"].extend(id_rec)
        print(f"Processed Patient {rec_id} -> {split_name.upper()} ({len(X_rec)} beats)")

    print("\n--- Summary of Generated Dataset Splits ---")
    for s_name in ["train", "val", "test"]:
        X_arr = np.array(splits[s_name]["X"], dtype=np.float32)
        y_arr = np.array(splits[s_name]["y"], dtype=np.int64)
        id_arr = np.array(splits[s_name]["record_id"])

        out_file = os.path.join(OUT_DIR, f"mitbih_{s_name}_ready.npz")
        np.savez_compressed(out_file, X_cnn=X_arr, y=y_arr, record_id=id_arr)
        
        # Also copy to quickstart-pytorch so Mirza's scripts locate them immediately
        alt_out = os.path.join(r"C:\LiveGuard\quickstart-pytorch", f"mitbih_{s_name}_ready.npz")
        np.savez_compressed(alt_out, X_cnn=X_arr, y=y_arr, record_id=id_arr)

        abnormal_count = int(np.sum(y_arr != 0))
        normal_count = int(np.sum(y_arr == 0))
        print(f"[{s_name.upper()}] {patient_counts[s_name]} patients | Total: {len(y_arr)} beats "
              f"(Normal: {normal_count}, Abnormal: {abnormal_count}) -> {out_file}")

if __name__ == "__main__":
    main()
