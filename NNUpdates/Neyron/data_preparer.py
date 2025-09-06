# Updates/Neyron/data_preparer.py

import os
import csv
import time
import hashlib
import pandas as pd
from typing import List, Optional, Tuple
from utils.common.FileStructureManager import FileStructureManager

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def _find_time_column(cols: List[str]) -> Optional[str]:
    candidates = ["timestamp", "time", "date", "datetime"]
    for c in candidates:
        if c in cols:
            return c
    return None

def _file_fingerprint(path: str) -> Tuple[int, float]:
    st = os.stat(path)
    return st.st_size, st.st_mtime

def _ensure_dir(d: str):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _load_manifest(manifest_path: str) -> pd.DataFrame:
    # Підтримка як .txt, так і спадково .csv
    try_candidates = [manifest_path]
    base, ext = os.path.splitext(manifest_path)
    if ext.lower() != ".txt":
        try_candidates.append(base + ".txt")
    try_candidates.append(base + ".csv")

    for p in try_candidates:
        if os.path.isfile(p):
            try:
                return pd.read_csv(p)
            except Exception:
                pass
    return pd.DataFrame(columns=[
        "source_path", "base_name", "start_idx", "end_idx", "length",
        "file_size", "file_mtime", "season_path", "created_at"
    ])

def _save_manifest(manifest: pd.DataFrame, manifest_path: str):
    tmp = manifest.copy()
    tmp.sort_values(["source_path", "start_idx"], inplace=True, ignore_index=True)
    # Записуємо CSV-контент із розширенням .txt (вимога)
    tmp.to_csv(manifest_path, index=False)

def _archive_existing(seasons_dir: str, base_name: Optional[str] = None) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    archive_dir = os.path.join(seasons_dir, "archive", ts)
    _ensure_dir(archive_dir)

    def _match(p: str) -> bool:
        if base_name is None:
            return True
        return os.path.basename(p).startswith(base_name + "_season_")

    for root, _, files in os.walk(seasons_dir):
        if os.path.abspath(root).startswith(os.path.abspath(os.path.join(seasons_dir, "archive"))):
            continue
        for f in files:
            if f.endswith(".csv"):
                p = os.path.join(root, f)
                if _match(p):
                    rel = os.path.relpath(p, seasons_dir)
                    dst = os.path.join(archive_dir, rel)
                    _ensure_dir(os.path.dirname(dst))
                    try:
                        os.rename(p, dst)
                    except Exception:
                        pass
    return archive_dir

def create_market_seasons(
    source_directory_key: str,
    season_length_candles: int = 720,
    stride: Optional[int] = None,
    output_dir: str = "data/Updates/Neyron/seasons",
    mode: str = "append",  # "append" | "overwrite" | "archive"
    validate: bool = True,
) -> List[str]:
    """
    Ріже історичні CSV на сезони та зберігає окремими файлами.
    Маніфест тепер створюється як seasons_manifest.txt.
    """
    assert season_length_candles > 0, "season_length_candles має бути > 0"
    if stride is None or stride <= 0:
        stride = season_length_candles

    fsm = FileStructureManager()
    source_path = fsm.get_path(source_directory_key, is_file=False)
    if not source_path or not os.path.isdir(source_path):
        print(f"❌ Помилка: директорію за ключем '{source_directory_key}' не знайдено.")
        return []

    _ensure_dir(output_dir)
    # НОВЕ: маніфест як .txt
    manifest_path = os.path.join(output_dir, "seasons_manifest.txt")
    manifest = _load_manifest(manifest_path)

    # Зібрати всі CSV
    all_files = fsm.get_all_files_in_directory(source_directory_key, extensions=[".csv"])
    raw_files = [f for f in all_files if "_processing" not in f]

    created_paths: List[str] = []

    if mode in ("overwrite", "archive"):
        print(f"🧹 Режим '{mode}': очищення існуючих сезонів цього джерела...")
        seen_bases = set(os.path.splitext(os.path.basename(p))[0] for p in raw_files)
        for base in seen_bases:
            if mode == "archive":
                _archive_existing(output_dir, base_name=base)
            else:
                for root, _, files in os.walk(output_dir):
                    for f in files:
                        if f.endswith(".csv") and f.startswith(base + "_season_"):
                            try:
                                os.remove(os.path.join(root, f))
                            except Exception:
                                pass
            manifest = manifest[manifest["base_name"] != base].reset_index(drop=True)

    print(f"\n🔪 Нарізаємо дані на сезони: length={season_length_candles}, stride={stride}, mode={mode}")
    for file_path in raw_files:
        try:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            sz, mt = _file_fingerprint(file_path)

            df = pd.read_csv(file_path)
            df = _normalize_columns(df)

            if "close" not in df.columns:
                print(f"⚠️ Пропуск {file_path}: немає колонки 'close' (після lowercase).")
                continue

            if validate:
                tcol = _find_time_column(list(df.columns))
                if tcol is not None:
                    try:
                        if not df[tcol].is_monotonic_increasing:
                            df = df.sort_values(tcol).reset_index(drop=True)
                    except Exception:
                        pass

            n = len(df)
            if n < season_length_candles:
                continue

            start = 0
            while start + season_length_candles <= n:
                end = start + season_length_candles

                if mode == "append":
                    dup = manifest[
                        (manifest["source_path"] == file_path)
                        & (manifest["start_idx"] == start)
                        & (manifest["length"] == season_length_candles)
                    ]
                    if not dup.empty:
                        start += stride
                        continue

                season_df = df.iloc[start:end].reset_index(drop=True)
                season_filename = f"{base_name}_season_{start}_{end}.csv"
                season_output_path = os.path.join(output_dir, season_filename)

                season_df.to_csv(season_output_path, index=False)
                created_paths.append(season_output_path)

                manifest = pd.concat([
                    manifest,
                    pd.DataFrame([{
                        "source_path": file_path,
                        "base_name": base_name,
                        "start_idx": start,
                        "end_idx": end,
                        "length": season_length_candles,
                        "file_size": sz,
                        "file_mtime": mt,
                        "season_path": season_output_path,
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }])
                ], ignore_index=True)

                start += stride

        except Exception as e:
            print(f"⚠️ Помилка обробки файлу {file_path}: {e}")

    _save_manifest(manifest, manifest_path)
    print(f"✅ Успішно створено {len(created_paths)} нових сезонів. Маніфест (txt): {manifest_path}")
    return created_paths
